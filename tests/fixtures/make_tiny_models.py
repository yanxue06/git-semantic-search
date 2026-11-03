"""Build the two tiny embedding models the custom-model tests load.

Each is a real ONNX graph with a real tokenizer, small enough to commit, so the
tests run the same load-and-encode path a Hugging Face model does without
downloading one.

    uv run --no-project --with onnx==1.17.0 --with tokenizers==0.21.1 --with numpy \
        python tests/fixtures/make_tiny_models.py

tiny-mean   16 dims. Returns one vector per token; git-semantic averages them.
            Files sit flat, the way a hand-made model directory would.
tiny-cls    24 dims. Returns the token average at every position, so taking
            the first token works. Laid out like a cloned Hugging Face repo:
            the graph in onnx/, plus prompts and a context limit to detect.
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

HERE = Path(__file__).parent

SPECIAL = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]

# Words the tests' commit messages and queries use. Everything else, including
# the author name and diff text, maps to [UNK], whose vector is zero, so only
# these words move an embedding.
WORDS = [
    "fix", "memory", "leak", "parser", "add", "login", "page", "update",
    "readme", "install", "steps", "speed", "startup", "caching", "config",
]


def build_tokenizer(path):
    vocab = {}
    for token in SPECIAL + WORDS:
        vocab[token] = len(vocab)

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


def word_vectors(dim, seed):
    rng = np.random.default_rng(seed)
    table = rng.standard_normal((len(SPECIAL) + len(WORDS), dim)).astype(np.float32)
    table[: len(SPECIAL)] = 0.0
    return table


def mask_as_column(nodes):
    # attention_mask [batch, seq] -> float [batch, seq, 1], ready to scale rows.
    nodes.append(helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Unsqueeze", ["mask_f", "last_axis"], ["mask_col"]))


def save(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def token_input(name):
    return helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", "seq"])


def build_mean_model(path, dim):
    table = word_vectors(dim, seed=7)
    nodes = [helper.make_node("Gather", ["table", "input_ids"], ["tokens"])]
    mask_as_column(nodes)
    nodes.append(helper.make_node("Mul", ["tokens", "mask_col"], ["last_hidden_state"]))

    graph = helper.make_graph(
        nodes,
        "tiny-mean",
        [token_input("input_ids"), token_input("attention_mask")],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch", "seq", dim])],
        initializer=[
            numpy_helper.from_array(table, "table"),
            numpy_helper.from_array(np.array([-1], dtype=np.int64), "last_axis"),
        ],
    )
    save(graph, path)


def build_cls_model(path, dim):
    table = word_vectors(dim, seed=11)
    nodes = [
        helper.make_node("Gather", ["table", "input_ids"], ["words"]),
        # token_type_ids is consumed so the input is real, but its table is zero.
        helper.make_node("Gather", ["type_table", "token_type_ids"], ["types"]),
        helper.make_node("Add", ["words", "types"], ["tokens"]),
    ]
    mask_as_column(nodes)
    nodes += [
        helper.make_node("Mul", ["tokens", "mask_col"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "seq_axis"], ["total"], keepdims=1),
        helper.make_node("ReduceSum", ["mask_col", "seq_axis"], ["count"], keepdims=1),
        helper.make_node("Div", ["total", "count"], ["average"]),
        helper.make_node("Shape", ["masked"], ["token_shape"]),
        helper.make_node("Expand", ["average", "token_shape"], ["last_hidden_state"]),
    ]

    graph = helper.make_graph(
        nodes,
        "tiny-cls",
        [token_input("input_ids"), token_input("attention_mask"), token_input("token_type_ids")],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch", "seq", dim])],
        initializer=[
            numpy_helper.from_array(table, "table"),
            numpy_helper.from_array(np.zeros((2, dim), dtype=np.float32), "type_table"),
            numpy_helper.from_array(np.array([-1], dtype=np.int64), "last_axis"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "seq_axis"),
        ],
    )
    save(graph, path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def pooling_config(dim, mode):
    return {
        "word_embedding_dimension": dim,
        "pooling_mode_cls_token": mode == "cls",
        "pooling_mode_mean_tokens": mode == "mean",
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }


def main():
    mean_dir = HERE / "tiny-mean"
    build_mean_model(mean_dir / "model.onnx", dim=16)
    build_tokenizer(mean_dir / "tokenizer.json")
    write_json(mean_dir / "1_Pooling" / "config.json", pooling_config(16, "mean"))

    cls_dir = HERE / "tiny-cls"
    build_cls_model(cls_dir / "onnx" / "model.onnx", dim=24)
    build_tokenizer(cls_dir / "tokenizer.json")
    write_json(cls_dir / "1_Pooling" / "config.json", pooling_config(24, "cls"))
    write_json(cls_dir / "config.json", {"max_position_embeddings": 128, "hidden_size": 24})
    write_json(
        cls_dir / "config_sentence_transformers.json",
        {"prompts": {"query": "query: ", "document": "passage: "}},
    )


if __name__ == "__main__":
    main()
