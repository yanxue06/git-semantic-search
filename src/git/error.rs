use thiserror::Error;

#[derive(Debug, Error)]
pub enum GitError {
    #[error("not a git repository or unable to open")]
    RepositoryNotFound(#[source] git2::Error),

    #[error("commit {0} not found in history — index may be stale after a rebase or force push")]
    CommitNotFound(String),

    #[error("failed to walk commit history: no HEAD found")]
    NoHead,

    #[error("failed to read repository history")]
    RevwalkFailed(#[source] git2::Error),

    #[error("git operation failed")]
    Git(#[from] git2::Error),
}

impl GitError {
    /// Returns a user-facing hint for how to resolve this error.
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            Self::RepositoryNotFound(_) => Some(
                "Make sure you're inside a git repository, or use --path to specify one."
            ),
            Self::CommitNotFound(_) => Some(
                "The index references commits that no longer exist. Run: git-semantic index --force"
            ),
            Self::NoHead => Some(
                "The repository has no commits yet. Make at least one commit before indexing."
            ),
            Self::RevwalkFailed(_) => Some(
                "The git history may be corrupted. Try running: git fsck"
            ),
            Self::Git(_) => None,
        }
    }

    /// Returns an error code for programmatic identification.
    pub fn code(&self) -> &'static str {
        match self {
            Self::RepositoryNotFound(_) => "E2001",
            Self::CommitNotFound(_) => "E2002",
            Self::NoHead => "E2003",
            Self::RevwalkFailed(_) => "E2004",
            Self::Git(_) => "E2005",
        }
    }
}
