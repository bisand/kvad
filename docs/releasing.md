# Releasing

A release is made in the GitHub UI, and the release is what decides the
version. Nothing on a developer's machine bumps a version number, and no
workflow creates a release.

1. Make sure `master` is what you want to ship, and that the web UI builds:
   `cd web && npm ci && npm run build`. CI does this too, and fails the
   release if `web/dist/index.html` is missing — a `kvad-serve` built without
   it compiles happily and serves a stub instead of the UI.

2. Draft a release at <https://github.com/bisand/kvad/releases/new> against a
   new tag `vMAJOR.MINOR.PATCH`, write the notes, and publish it. Or:

   ```bash
   gh release create v0.2.0 --title "kvad 0.2.0" --notes "…"
   ```

   A tag with a hyphen in it — `v0.2.0-rc1` — should be marked as a
   pre-release, which keeps `install.sh` from picking it up as the latest.

3. Publishing starts [`release.yml`](../.github/workflows/release.yml), which
   builds four tarballs and attaches them to that same release, along with a
   `SHA256SUMS`. It takes a while; candle is not a small dependency.

## What the version number touches

One line, in `[workspace.package]` in the root `Cargo.toml`. Every crate
inherits it with `version.workspace = true`, and the internal path
dependencies carry no version of their own, so there is exactly one number
and nothing to keep in sync.

The workflow rewrites that line from the release's tag before it builds, and
does not commit the change. What is in git says what the last release was;
what a *binary* says when you run `kvad --version` comes from the release it
was built for. A build that disagrees with its release fails the smoke test
rather than shipping.

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
