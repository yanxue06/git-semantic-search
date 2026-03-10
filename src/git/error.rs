use thiserror::Error;

#[derive(Debug, Error)]
pub enum GitError {
    #[error("not a git repository or unable to open")]
    RepositoryNotFound(#[source] git2::Error),

    #[error("commit {0} not found in history — index may be stale after a rebase or force push")]
    CommitNotFound(String),

    #[error(transparent)]
    Git(#[from] git2::Error),
}
