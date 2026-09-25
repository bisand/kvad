# Releasing

A release is made in the GitHub UI, and the release is what decides the
version. Nothing on a developer's machine bumps a version number, and no
workflow creates a release.

1. Make sure `master` is what you want to ship, and that the web UI builds:
   `cd web && npm ci && npm run build`. CI does this too, and fails the
   release if `web/dist/index.html` is missing — a `kvad-serve` built without
   it compiles happily and serves a stub instead of the UI.

   Nothing but the release builds for Linux: the pull-request check runs on
   macOS. So a macOS-only call first fails after the release is public, and a
   failed build attaches nothing — v0.3.0 went out with no binaries over
   `libc::F_NOCACHE`. Look through the diff since the last tag for `libc::`
   and other platform APIs in what Linux builds: `kvad`, and `kvad-serve`
   without its `gpu` feature, which leaves `kvad-gpu` out.

2. Draft a release at <https://github.com/bisand/kvad/releases/new> against a
   new tag `vMAJOR.MINOR.PATCH`, write the notes, and publish it. Or:

   ```bash
   gh release create v0.2.0 --title "kvad 0.2.0" --notes "…"
   ```

   A tag with a hyphen in it — `v0.2.0-rc1` — should be marked as a
   pre-release, which keeps `install.sh` from picking it up as the latest.

3. Publishing starts [`release.yml`](../.github/workflows/release.yml), which
   builds three tarballs — macOS on Apple Silicon, and Linux on x86_64 and
   arm64 — and attaches them to that same release, along with a `SHA256SUMS`.
   It takes a while; candle is not a small dependency. There has been no
   Intel Mac build since v0.6.0.

## What the version number touches

One line, in `[workspace.package]` in the root `Cargo.toml`. Every crate
inherits it with `version.workspace = true`, and the internal path
dependencies carry no version of their own, so there is exactly one number
and nothing to keep in sync.

The workflow rewrites that line from the release's tag before it builds, and
then — once the release has its files — commits it back to `master`, so the
branch says what the last release was. A build that disagrees with its
release fails the smoke test rather than shipping.

The builds themselves come from the tag, not from that commit. A release is a
statement about the code at a tag, and `master` may have moved on since it was
published; taking the tag's tree and stamping the version into it keeps the
artifacts faithful to the release they are attached to. The consequence worth
knowing is that the tagged tree does not contain its own version — check out
`v0.1.1` and `Cargo.toml` says whatever the release before it set. The commit
on `master` is what carries the number forward.

The commit is the last thing the workflow does, because a build that failed
should not leave `master` claiming a version with no binaries behind it. It is
also written to be safe to repeat: a re-run against a release whose version
`master` already carries says so and commits nothing.

`master` can move while a release builds — this workflow takes over ten
minutes — so each push attempt starts from whatever `master` is at that moment
and applies the version to it, rather than rebasing. Rebasing a commit that
edits `Cargo.toml` onto a `master` that also edited it conflicts, and a
conflict in an unattended job is a job that dies half-rebased.

Rewriting the version makes `Cargo.lock` stale, so the workflow runs
`cargo update --workspace` before building — that updates only the workspace
members' own entries and leaves every third-party dependency pinned where the
committed lock file put it. The build then uses `--locked`.

## Between publishing and the assets appearing

The release is the trigger, so for the length of the build it exists with no
files attached, and `install.sh` will tell anyone who runs it that there is no
build for their machine. If that matters for a particular release, publish it
when you can watch the workflow finish.

## If a build fails

Fix the cause, then re-run the workflow against the existing release with
`workflow_dispatch` — Actions → release → Run workflow → the tag. Assets are
uploaded with `--clobber`, so a re-run replaces what is there rather than
failing on it, and the install instructions appended to the notes are marked
so they are not appended twice.
