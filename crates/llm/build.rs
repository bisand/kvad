//! Which commit this build is, for a worker and a coordinator to compare.
//!
//! Two builds with different kernels can both be right and not agree to the
//! bit, and a model split across them is then wrong in a way nobody can
//! find. So a coordinator refuses a worker of another build, and "another
//! build" has to mean more than another version number: every commit
//! between two releases carries the same one.
//!
//! No commit to ask — a tarball, a crate from a registry — and the revision
//! is empty, and the version number is all there is to compare.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let commit = git(&["rev-parse", "--short=10", "HEAD"]).unwrap_or_default();
    // Changes not yet committed make this build something no commit names.
    let dirty = !commit.is_empty() && git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    println!("cargo:rustc-env=KVAD_REVISION={commit}{}", if dirty { "-dirty" } else { "" });

    // Again when the commit moves or the sources do, and not otherwise.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    if let Some(dir) = git(&["rev-parse", "--git-dir"]) {
        for file in ["HEAD", "index", "refs/heads", "packed-refs"] {
            println!("cargo:rerun-if-changed={dir}/{file}");
        }
    }
    if let Some(dir) = git(&["rev-parse", "--git-common-dir"]) {
        println!("cargo:rerun-if-changed={dir}/refs/heads");
    }
}
