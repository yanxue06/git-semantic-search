//! Code-aware tokenizer for commit text.
//!
//! A whitespace-and-lowercase tokenizer is wrong for this corpus. Commit
//! messages and diffs are full of identifiers — `src/auth.rs`, `encode_text`,
//! `CVE-2024-1234`, `HnswIndex` — and a user searching `auth` should find a
//! commit that only ever wrote `src/auth.rs`.
//!
//! So each identifier is emitted twice: once whole, once split at the
//! boundaries programmers actually use (path separators, underscores, hyphens,
//! camelCase humps, letter/digit transitions). The whole form keeps exact
//! matches strong; the parts make substring intent findable. BM25 divides by
//! document length, so the extra tokens don't inflate a document's score.

/// Longest token kept. 40 covers a full SHA-1; beyond that is base64 blobs and
/// minified payloads, which only add noise and index size.
const MAX_TOKEN_LEN: usize = 40;

/// Shortest token kept, whole or split. Single characters carry no retrieval
/// signal, appear in nearly every diff, and would bloat every posting list.
const MIN_TOKEN_LEN: usize = 2;

/// Tokenize `text` into lowercase terms, duplicates preserved so term
/// frequencies stay meaningful.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut word = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() {
            word.push(ch);
        } else {
            flush(&mut word, &mut tokens);
        }
    }
    flush(&mut word, &mut tokens);

    tokens
}

/// Emit the accumulated word, then its sub-parts if it is compound.
fn flush(word: &mut String, tokens: &mut Vec<String>) {
    if word.is_empty() {
        return;
    }

    let whole = std::mem::take(word);
    if whole.len() > MAX_TOKEN_LEN || whole.len() < MIN_TOKEN_LEN {
        return;
    }

    let lowered = whole.to_lowercase();
    let parts = split_identifier(&whole);

    tokens.push(lowered.clone());

    // Only worth adding parts when the split actually found structure.
    if parts.len() > 1 {
        for part in parts {
            if part.len() >= MIN_TOKEN_LEN && part != lowered {
                tokens.push(part);
            }
        }
    }
}

/// Split an identifier at camelCase humps and letter/digit transitions.
///
/// Runs of capitals are kept together up to the last one, so `HTTPServer`
/// yields `http` and `server` rather than `h`, `t`, `t`, `p`, `server`.
fn split_identifier(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let mut parts = Vec::new();
    let mut current = String::new();

    for (i, &ch) in chars.iter().enumerate() {
        let previous = if i > 0 { Some(chars[i - 1]) } else { None };
        let next = chars.get(i + 1).copied();

        let boundary = match previous {
            None => false,
            Some(prev) => {
                // lower|digit -> Upper, e.g. "parseHTTP"
                let hump = !prev.is_uppercase() && ch.is_uppercase();
                // UPPER -> Upperlower, e.g. "HTTPServer" splits before "Server"
                let acronym_end = prev.is_uppercase()
                    && ch.is_uppercase()
                    && next.is_some_and(|n| n.is_lowercase());
                // letter <-> digit, e.g. "utf8" / "v2parse"
                let alpha_digit = prev.is_alphabetic() && ch.is_numeric();
                let digit_alpha = prev.is_numeric() && ch.is_alphabetic();

                hump || acronym_end || alpha_digit || digit_alpha
            }
        };

        if boundary && !current.is_empty() {
            parts.push(std::mem::take(&mut current));
        }
        current.push(ch);
    }

    if !current.is_empty() {
        parts.push(current);
    }

    parts.iter().map(|p| p.to_lowercase()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(text: &str) -> Vec<String> {
        tokenize(text)
    }

    #[test]
    fn splits_on_whitespace_and_lowercases() {
        assert_eq!(tokens("Fix The Bug"), vec!["fix", "the", "bug"]);
    }

    #[test]
    fn splits_a_path_into_components_and_keeps_them_findable() {
        let out = tokens("src/auth.rs");
        assert!(out.contains(&"src".to_string()));
        assert!(out.contains(&"auth".to_string()));
        assert!(out.contains(&"rs".to_string()));
    }

    #[test]
    fn keeps_the_whole_identifier_alongside_its_parts() {
        let out = tokens("encode_text");
        assert!(out.contains(&"encode".to_string()));
        assert!(out.contains(&"text".to_string()));
    }

    #[test]
    fn splits_camel_case() {
        let out = tokens("HnswIndex");
        assert!(
            out.contains(&"hnswindex".to_string()),
            "whole form: {out:?}"
        );
        assert!(out.contains(&"hnsw".to_string()), "parts: {out:?}");
        assert!(out.contains(&"index".to_string()), "parts: {out:?}");
    }

    #[test]
    fn keeps_acronyms_together() {
        let out = tokens("HTTPServer");
        assert!(out.contains(&"http".to_string()), "got {out:?}");
        assert!(out.contains(&"server".to_string()), "got {out:?}");
        assert!(
            !out.contains(&"h".to_string()),
            "must not shatter an acronym: {out:?}"
        );
    }

    #[test]
    fn splits_letter_digit_transitions() {
        let out = tokens("utf8");
        assert!(out.contains(&"utf8".to_string()));
        assert!(out.contains(&"utf".to_string()));
    }

    #[test]
    fn finds_a_cve_identifier_whole_and_in_parts() {
        let out = tokens("patch CVE-2024-1234");
        assert!(out.contains(&"cve".to_string()));
        assert!(out.contains(&"2024".to_string()));
        assert!(out.contains(&"1234".to_string()));
    }

    #[test]
    fn keeps_a_full_sha_but_drops_longer_blobs() {
        let sha = "a".repeat(40);
        assert!(tokens(&sha).contains(&sha));

        let blob = "b".repeat(41);
        assert!(
            tokens(&blob).is_empty(),
            "oversized tokens are noise, not signal"
        );
    }

    #[test]
    fn preserves_term_frequency() {
        let out = tokens("fix fix fix");
        assert_eq!(out.iter().filter(|t| *t == "fix").count(), 3);
    }

    #[test]
    fn does_not_duplicate_a_single_word() {
        assert_eq!(tokens("auth"), vec!["auth"]);
    }

    #[test]
    fn drops_single_character_tokens_whole_or_split() {
        let out = tokens("a_b_token");
        assert!(!out.contains(&"a".to_string()), "got {out:?}");
        assert!(!out.contains(&"b".to_string()), "got {out:?}");
        assert!(out.contains(&"token".to_string()), "got {out:?}");

        // Also as a split part: "utf8" -> "utf" survives, "8" does not.
        let digits = tokens("utf8");
        assert!(digits.contains(&"utf".to_string()), "got {digits:?}");
        assert!(!digits.contains(&"8".to_string()), "got {digits:?}");
    }

    #[test]
    fn handles_empty_and_punctuation_only_input() {
        assert!(tokens("").is_empty());
        assert!(tokens("--- +++ @@").is_empty());
    }

    #[test]
    fn tokenizes_a_realistic_diff_line() {
        let out = tokens("+    let engine = SearchEngine::new(model_manager)?;");
        for expected in ["let", "engine", "search", "new", "model", "manager"] {
            assert!(
                out.contains(&expected.to_string()),
                "missing {expected} in {out:?}"
            );
        }
    }
}
