//! The built web UI, baked into the binary.
//!
//! One file to deploy. `rust-embed` reads `web/dist` at compile time, so a
//! release build of `kvad-serve` carries the whole UI and needs nothing
//! beside it on the target machine.
//!
//! # `cargo build` never runs npm
//!
//! There is no `build.rs` calling out to Node, on purpose: `cargo test`
//! should work on a machine that has never installed npm, and a Rust build
//! that silently downloads a JavaScript dependency tree is not a Rust build.
//! So `web/dist` is built by hand — `cd web && npm ci && npm run build` — and
//! a binary built without it serves [`stub`], which says exactly that.
//!
//! `web/dist/.gitkeep` is committed for the same reason `rust-embed` needs:
//! the directory has to exist for the crate to compile, even when empty.
//!
//! # In development
//!
//! Vite serves the UI on its own port and proxies `/api` and `/v1` here, so
//! nothing in this file is involved and a page reload is instant.

use axum::http::{header, StatusCode, Uri};
use axum::response::{Html, IntoResponse, Response};

#[derive(rust_embed::Embed)]
#[folder = "../../web/dist"]
struct Dist;

/// Whether a UI was built into this binary.
pub fn is_embedded() -> bool {
    Dist::get("index.html").is_some()
}

/// Serve a file from the built UI, falling back to `index.html`.
///
/// The fallback is what makes a client-side router work: the browser asks for
/// `/models` after a reload, there is no such file, and the SPA has to be
/// handed back so its own router can read the path. Only requests that look
/// like a page get that treatment — a missing `.js` or `.css` is a 404, or a
/// broken build would answer every stale asset request with HTML and the
/// error in the console would be a syntax error in what looked like a script.
pub async fn serve(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    if let Some(response) = file(path) {
        return response;
    }
    match Dist::get("index.html") {
        _ if looks_like_a_file(path) => (StatusCode::NOT_FOUND, "not found").into_response(),
        Some(_) => file("index.html").expect("index.html was there a moment ago"),
        None => stub().into_response(),
    }
}

fn file(path: &str) -> Option<Response> {
    let asset = Dist::get(path)?;
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    // Vite fingerprints everything under `assets/`, so those can be cached
    // forever; `index.html` names them and must not be.
    let cache = match path.starts_with("assets/") {
        true => "public, max-age=31536000, immutable",
        false => "no-cache",
    };
    Some(
        (
            [(header::CONTENT_TYPE, mime.as_ref()), (header::CACHE_CONTROL, cache)],
            asset.data.into_owned(),
        )
            .into_response(),
    )
}

/// Whether a path is asking for a file rather than for a page.
fn looks_like_a_file(path: &str) -> bool {
    path.rsplit('/').next().is_some_and(|last| last.contains('.'))
}

/// What a binary built without the UI serves: the command that would fix it.
///
/// Plain HTML with no assets, because by definition there are none.
pub fn stub() -> Html<&'static str> {
    Html(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kvad-serve</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 16px/1.6 ui-sans-serif, system-ui, sans-serif; margin: 0;
         min-height: 100vh; display: grid; place-items: center; padding: 2rem; }
  main { max-width: 42rem; }
  h1 { font-size: 1.5rem; margin: 0 0 1rem; }
  pre { padding: 1rem; border-radius: .5rem; overflow-x: auto;
        background: color-mix(in srgb, currentColor 8%, transparent); }
  a { color: inherit; }
</style>
</head>
<body>
<main>
  <h1>The API is running. The web UI was not built into this binary.</h1>
  <p>Build it once, then start the server again:</p>
  <pre>cd web
npm ci
npm run build
cargo build --release -p kvad-serve</pre>
  <p>Or run the UI from Vite while you work on it, which proxies the API back
     here:</p>
  <pre>cd web
npm run dev</pre>
  <p><a href="/api/health">/api/health</a> answers either way.</p>
</main>
</body>
</html>
"#,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule that decides between a 404 and the SPA. Getting it backwards
    /// makes every missing script return HTML, and the browser then reports a
    /// syntax error rather than a missing file.
    #[test]
    fn a_path_with_a_dot_in_its_last_segment_is_a_file() {
        for path in ["assets/index-a1b2.js", "favicon.ico", "index.html", "a/b/c.css"] {
            assert!(looks_like_a_file(path), "{path}");
        }
        for path in ["", "models", "chat/new", "settings/auth", "v1.2/models"] {
            assert!(!looks_like_a_file(path), "{path}");
        }
    }

    /// Whatever else the stub says, it has to say how to fix itself.
    #[test]
    fn the_stub_names_the_command_that_builds_the_ui() {
        let Html(html) = stub();
        assert!(html.contains("npm run build"), "the stub does not say how to build the UI");
        assert!(html.contains("/api/health"));
    }
}
