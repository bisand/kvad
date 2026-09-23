//! Who a request is from.
//!
//! One question, asked the same way for every mode, in three steps:
//!
//! 1. **What did the request present?** A cookie, a bearer token, a Basic
//!    header, or nothing. [`Presented::of`] reads that out of the headers and
//!    touches nothing else — it is cheap and cannot fail in an interesting
//!    way.
//! 2. **Is this request allowed to use it?** A cookie is sent by the browser
//!    whether or not our page caused the request, so a cookie on a mutating
//!    request needs an `Origin` that is ours. See [`Presented::forgeable`].
//! 3. **Whose is it?** [`Provider::identify`], which reads the database and
//!    is therefore blocking; the extractor runs it on a blocking thread.
//!
//! Splitting it that way is what keeps each mode small: `local` knows about
//! cookies, `basic` knows about the `Authorization` header, and neither of
//! them knows anything about CSRF or about running off the async runtime.
//!
//! # How a handler says what it needs
//!
//! ```ignore
//! async fn list_users(_: Admin) -> Json<Vec<User>> { ... }   // admins only
//! async fn chat(who: Identity) -> ...                        // anyone signed in
//! ```
//!
//! Both are axum extractors, so the requirement is in the signature where it
//! can be read, and a handler that takes neither is one that deliberately
//! takes neither.

use crate::config::Mode;
use crate::db::Db;
use crate::users;
use axum::extract::FromRequestParts;
use axum::http::header::{AUTHORIZATION, COOKIE, ORIGIN, REFERER};
use axum::http::request::Parts;
use axum::http::{Method, StatusCode};
use base64::Engine;

/// The cookie a signed-in browser carries.
pub const COOKIE_NAME: &str = "kvad_session";

/// What somebody is allowed to do.
///
/// Two, and no more until there is a reason: `admin` can change the machine —
/// pull models, train, edit settings, manage accounts — and `user` can talk
/// to what is already there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Admin,
    User,
}

impl Role {
    fn parse(s: &str) -> Role {
        // An unrecognised role in the database is the *less* privileged one.
        // The column is constrained to two values, so this cannot happen; if
        // it somehow does, it must not fail open.
        match s {
            "admin" => Role::Admin,
            _ => Role::User,
        }
    }
}

/// Who a request is from, once the configured mode has decided.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Identity {
    /// The account's row, or `None` with `auth.mode = "none"`, where there
    /// are no accounts and the person at the keyboard is the only one there
    /// can be. Conversations made then have no owner; see the migration.
    pub id: Option<i64>,
    pub name: String,
    pub role: Role,
}

impl Identity {
    /// The identity every request has when there is no authentication: the
    /// person at the keyboard, who is the only one who can reach loopback.
    fn local_operator() -> Self {
        Identity { id: None, name: "local".into(), role: Role::Admin }
    }

    fn of(user: &users::User) -> Self {
        Identity { id: Some(user.id), name: user.name.clone(), role: Role::parse(&user.role) }
    }

    pub fn is_admin(&self) -> bool {
        self.role == Role::Admin
    }
}

/// What a request offers as proof of who it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Presented {
    Nothing,
    /// `Authorization: Bearer …` — an API key.
    Bearer(String),
    /// The session cookie.
    Cookie(String),
    /// `Authorization: Basic …`, already decoded.
    Basic { name: String, password: String },
}

impl Presented {
    /// Read the credential out of a request's headers.
    ///
    /// The `Authorization` header wins over the cookie. Something had to put
    /// it there on purpose, whereas the cookie is sent by the browser
    /// whatever the page; when both arrive, the deliberate one is the one
    /// that was meant.
    pub fn of(parts: &Parts) -> Presented {
        if let Some(value) = parts.headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok()) {
            let (scheme, rest) = value.split_once(' ').unwrap_or((value, ""));
            if scheme.eq_ignore_ascii_case("bearer") && !rest.trim().is_empty() {
                return Presented::Bearer(rest.trim().to_string());
            }
            if scheme.eq_ignore_ascii_case("basic") {
                if let Some(pair) = decode_basic(rest.trim()) {
                    return pair;
                }
            }
        }
        match cookie(parts, COOKIE_NAME) {
            Some(token) => Presented::Cookie(token),
            None => Presented::Nothing,
        }
    }

    /// Whether a browser could be tricked into sending this on our behalf.
    ///
    /// Only the cookie: it rides along with any request the browser makes to
    /// this origin, including one caused by somebody else's page. A bearer or
    /// Basic header had to be put there by whoever wrote the client, which is
    /// also why an ordinary API client — which sends no `Origin` at all — is
    /// not caught by the check this answer turns on.
    pub fn forgeable(&self) -> bool {
        matches!(self, Presented::Cookie(_))
    }
}

fn decode_basic(encoded: &str) -> Option<Presented> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(encoded).ok()?;
    let text = String::from_utf8(bytes).ok()?;
    // The password may contain colons; the name may not. Split on the first.
    let (name, password) = text.split_once(':')?;
    Some(Presented::Basic { name: name.to_string(), password: password.to_string() })
}

/// One cookie's value, from the `Cookie` header.
fn cookie(parts: &Parts, name: &str) -> Option<String> {
    let header = parts.headers.get(COOKIE)?.to_str().ok()?;
    header.split(';').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k.trim() == name).then(|| v.trim().to_string())
    })
}

/// Whether a request that carries a cookie may act on it.
///
/// A mutating request authenticated by a cookie has to prove it came from a
/// page of ours, or any site the signed-in person visits could make it. The
/// proof is the `Origin` header, which a browser sets and a page cannot; it
/// has to be present, because a missing one cannot be told from a stripped
/// one.
///
/// `SameSite=Lax` on the cookie already stops most of this. This is the
/// second lock, for the browsers and the request shapes where it does not.
pub fn origin_is_ours(parts: &Parts) -> bool {
    if matches!(parts.method, Method::GET | Method::HEAD | Method::OPTIONS) {
        return true;
    }
    let Some(host) = parts.headers.get(axum::http::header::HOST).and_then(|v| v.to_str().ok())
    else {
        // HTTP/2 puts the authority in the URI rather than in a header.
        let Some(authority) = parts.uri.authority().map(|a| a.as_str()) else { return false };
        return declared_origin(parts).is_some_and(|o| o == authority);
    };
    declared_origin(parts).is_some_and(|o| o == host)
}

/// The host:port the request says it came from, out of `Origin` or, failing
/// that, `Referer`.
fn declared_origin(parts: &Parts) -> Option<String> {
    let raw = parts
        .headers
        .get(ORIGIN)
        .or_else(|| parts.headers.get(REFERER))
        .and_then(|v| v.to_str().ok())?;
    // `https://host:port/whatever` -> `host:port`.
    let after_scheme = raw.split_once("://").map(|(_, rest)| rest).unwrap_or(raw);
    let host = after_scheme.split(['/', '?', '#']).next()?;
    (!host.is_empty()).then(|| host.to_string())
}

/// Decides who a request is from. One per mode.
///
/// A trait rather than a `match` so that each mode's rules are in one place
/// and can be tested without a server. `identify` reads the database and is
/// therefore blocking; the extractor runs it off the runtime.
pub trait Provider: Send + Sync + 'static {
    fn mode(&self) -> Mode;

    fn identify(&self, presented: &Presented, db: &Db) -> Option<Identity>;

    /// What to put in `WWW-Authenticate` on a 401. `None` is a plain refusal,
    /// which is what a mode with a login page wants — a browser shown a
    /// challenge pops up its own dialog instead of the page.
    fn challenge(&self) -> Option<String> {
        None
    }
}

/// The account behind an API key, in any mode that has accounts.
///
/// Bearer tokens work alongside every mode, because a script does not have a
/// browser to sign in with.
fn by_key(presented: &Presented, db: &Db) -> Option<Identity> {
    match presented {
        Presented::Bearer(token) => users::from_key(db, token).ok().flatten().map(|u| Identity::of(&u)),
        _ => None,
    }
}

/// Everyone is the operator. Safe only on loopback, which is enforced before
/// the socket is opened rather than here.
///
/// Credentials are ignored rather than checked: there are no accounts to
/// check them against, and an OpenAI client sends `Authorization: Bearer
/// sk-…` whether or not anybody asked it to. Refusing those would break the
/// compatible endpoint on exactly the setup it is most likely to be used on.
pub struct NoAuth;

impl Provider for NoAuth {
    fn mode(&self) -> Mode {
        Mode::None
    }
    fn identify(&self, _presented: &Presented, _db: &Db) -> Option<Identity> {
        Some(Identity::local_operator())
    }
}

/// Username and password, against this server's own accounts, with a
/// server-side session in a cookie.
pub struct Local;

impl Provider for Local {
    fn mode(&self) -> Mode {
        Mode::Local
    }

    fn identify(&self, presented: &Presented, db: &Db) -> Option<Identity> {
        match presented {
            Presented::Cookie(token) => {
                users::from_session(db, token).ok().flatten().map(|u| Identity::of(&u))
            }
            // Basic is not offered in this mode. Accepting it here would mean
            // every request could carry a password, which is the thing
            // sessions exist to avoid.
            other => by_key(other, db),
        }
    }
}

/// The same accounts over HTTP Basic, for scripts and for a reverse proxy
/// that would rather forward a header than a cookie.
///
/// Every request carries the password, and every request pays for an argon2
/// verification. That is the trade this mode is: simple to use from a shell,
/// slower and more exposed than a session.
pub struct Basic;

impl Provider for Basic {
    fn mode(&self) -> Mode {
        Mode::Basic
    }

    fn identify(&self, presented: &Presented, db: &Db) -> Option<Identity> {
        match presented {
            Presented::Basic { name, password } => {
                // `sign_in` writes a session row, which Basic has no use for.
                // Verify without one.
                let user = users::by_name(db, name).ok().flatten()?;
                let hash: String = db
                    .with(|c| {
                        c.query_row(
                            "SELECT coalesce(password_hash, '') FROM users WHERE id = ?1",
                            [user.id],
                            |r| r.get(0),
                        )
                    })
                    .ok()?;
                crate::secret::verify_password(password, &hash).then(|| Identity::of(&user))
            }
            // A browser sent here would keep its session cookie working, which
            // is how the Settings page stays usable in this mode.
            Presented::Cookie(token) => {
                users::from_session(db, token).ok().flatten().map(|u| Identity::of(&u))
            }
            other => by_key(other, db),
        }
    }

    fn challenge(&self) -> Option<String> {
        Some(r#"Basic realm="kvad", charset="UTF-8""#.into())
    }
}

/// An external identity provider decides who somebody is; this server decides
/// what they may do about it.
///
/// Signing in does not happen through `identify` at all — it happens at
/// `/api/auth/oidc/callback`, which ends in a session like any other. So this
/// looks exactly like [`Local`] once somebody is in, and the whole of the
/// difference is in [`crate::oidc`].
pub struct Oidc;

impl Provider for Oidc {
    fn mode(&self) -> Mode {
        Mode::Oidc
    }

    fn identify(&self, presented: &Presented, db: &Db) -> Option<Identity> {
        match presented {
            Presented::Cookie(token) => {
                users::from_session(db, token).ok().flatten().map(|u| Identity::of(&u))
            }
            other => by_key(other, db),
        }
    }
}

/// The provider for a mode, or an error naming the mode that has none yet.
pub fn provider(mode: Mode, _oidc: &crate::config::Oidc) -> Result<Box<dyn Provider>, String> {
    match mode {
        Mode::None => Ok(Box::new(NoAuth)),
        Mode::Local => Ok(Box::new(Local)),
        Mode::Basic => Ok(Box::new(Basic)),
        Mode::Oidc => Ok(Box::new(Oidc)),
    }
}

/// The one-time token that lets the first account be made.
///
/// Printed to the terminal at startup when a mode with accounts has none, and
/// gone the moment it is used or the server restarts. It exists so that the
/// answer to "how do I get in the first time" is not "edit the database" and
/// is not "there is a default password".
#[derive(Default)]
pub struct Setup(std::sync::Mutex<Option<String>>);

impl Setup {
    pub fn issue(&self) -> String {
        let token = crate::secret::token();
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(token.clone());
        token
    }

    pub fn wanted(&self) -> bool {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    /// Spend the token. True once, for the right token, and never again.
    pub fn spend(&self, offered: &str) -> bool {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        match held.as_deref() == Some(offered) {
            true => {
                *held = None;
                true
            }
            false => false,
        }
    }
}

/// Everything a handler needs that is not in the request.
#[derive(Clone)]
pub struct State {
    pub db: crate::db::Db,
    pub auth: std::sync::Arc<dyn Provider>,
    pub engine: std::sync::Arc<crate::scheduler::Scheduler>,
    pub jobs: std::sync::Arc<crate::jobs::Jobs>,
    pub metrics: std::sync::Arc<crate::metrics::Metrics>,
    pub setup: std::sync::Arc<Setup>,
    /// What an identity provider was told, and the sign-ins waiting on it.
    /// Empty and unused in every other mode.
    pub oidc: std::sync::Arc<(crate::config::Oidc, crate::oidc::Flows)>,
    pub started: std::time::Instant,
    /// Whether a completion naming a model that is not in memory loads it;
    /// see `config::Server::load_on_request`.
    pub load_on_request: bool,
}

/// An error a rejected request turns into: a status and a short reason.
///
/// Deliberately says nothing about *why* beyond the status. A 401 that
/// explained itself would be a way to ask questions about the accounts.
#[derive(Debug)]
pub struct Denied(pub StatusCode, pub Option<String>);

impl axum::response::IntoResponse for Denied {
    fn into_response(self) -> axum::response::Response {
        let body = axum::Json(serde_json::json!({
            "error": match self.0 {
                StatusCode::UNAUTHORIZED => "not signed in",
                StatusCode::FORBIDDEN => "not allowed",
                _ => "refused",
            }
        }));
        match self.1 {
            Some(challenge) => {
                (self.0, [(axum::http::header::WWW_AUTHENTICATE, challenge)], body).into_response()
            }
            None => (self.0, body).into_response(),
        }
    }
}

impl FromRequestParts<State> for Identity {
    type Rejection = Denied;

    async fn from_request_parts(parts: &mut Parts, state: &State) -> Result<Self, Self::Rejection> {
        let presented = Presented::of(parts);

        // Before anything is looked up: a cookie on a mutating request from
        // somewhere that is not us is not a credential, it is somebody else's
        // page using this person's browser.
        if presented.forgeable() && !origin_is_ours(parts) {
            return Err(Denied(StatusCode::FORBIDDEN, None));
        }

        let (auth, db) = (state.auth.clone(), state.db.clone());
        let who = tokio::task::spawn_blocking(move || auth.identify(&presented, &db))
            .await
            .map_err(|_| Denied(StatusCode::INTERNAL_SERVER_ERROR, None))?;

        who.ok_or_else(|| Denied(StatusCode::UNAUTHORIZED, state.auth.challenge()))
    }
}

/// An [`Identity`] that is also an administrator.
///
/// A separate extractor rather than a check inside the handler, so that a
/// handler which needs an admin cannot be written without asking for one.
pub struct Admin(pub Identity);

impl FromRequestParts<State> for Admin {
    type Rejection = Denied;

    async fn from_request_parts(parts: &mut Parts, state: &State) -> Result<Self, Self::Rejection> {
        let who = Identity::from_request_parts(parts, state).await?;
        match who.is_admin() {
            true => Ok(Admin(who)),
            false => Err(Denied(StatusCode::FORBIDDEN, None)),
        }
    }
}

#[cfg(test)]
mod tests;
