//! `git-semantic`, the name this tool had before `git-sem`.
//!
//! Prints a one-line note on stderr, then hands the whole command line to the
//! `git-sem` installed beside it, so existing scripts and `git semantic` keep
//! working.

use std::env;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

fn main() {
    eprintln!(
        "note: git-semantic is now git-sem, so `git sem` works too. The old name will go away in a future release."
    );

    let args: Vec<OsString> = env::args_os().skip(1).collect();
    run(&new_binary(), &args);
}

/// `git-sem` from the same directory as this binary, which is where
/// `cargo install` puts both. Anywhere else, whatever `git-sem` is on PATH.
fn new_binary() -> PathBuf {
    let name = format!("git-sem{}", env::consts::EXE_SUFFIX);
    if let Ok(this) = env::current_exe() {
        let sibling = this.with_file_name(&name);
        if sibling.exists() {
            return sibling;
        }
    }
    PathBuf::from(name)
}

/// Replace this process with `git-sem`, so its exit code, signals, and pipes
/// are exactly what the caller sees.
#[cfg(unix)]
fn run(binary: &Path, args: &[OsString]) {
    use std::os::unix::process::CommandExt;
    let err = Command::new(binary).args(args).exec();
    fail(binary, err);
}

#[cfg(not(unix))]
fn run(binary: &Path, args: &[OsString]) {
    match Command::new(binary).args(args).status() {
        Ok(status) => process::exit(status.code().unwrap_or(1)),
        Err(err) => fail(binary, err),
    }
}

fn fail(binary: &Path, err: io::Error) -> ! {
    eprintln!("\n  error: could not run {}: {err}", binary.display());
    eprintln!("\n  hint: Reinstall with: cargo install git-semantic\n");
    process::exit(1);
}
