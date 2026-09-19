//! Signing in and out, and managing who may.
//!
//! # The cookie
//!
//! `HttpOnly`, so a script on the page cannot read it and a cross-site
//! scripting bug cannot become a stolen session. `SameSite=Lax`, so the
//! browser does not attach it to a request another site caused — which is the
//! first of the two locks against cross-site forgery; the second is the
//! `Origin` check in [`crate::auth`]. `Secure` only over HTTPS, because a
//! `Secure` cookie on `http://localhost` is a cookie the browser throws away.

use crate::api::{blocking, Fail};
use crate::auth::{Admin, Identity, State, COOKIE_NAME};
use crate::users::{self, ApiKey, Session, User, SESSION_DAYS};
use axum::extract::{Path, State as St};
use axum::http::header::{HeaderMap, SET_COOKIE, USER_AGENT};
use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;

/// What the sign-in page needs to know before anybody has signed in.
///
/// Answered without a credential, so it says as little as it can get away
/// with: which mode, and whether this server has any accounts at all.
#[derive(serde::Serialize)]
pub struct Situation {
    mode: String,
    /// True when the server has no accounts and is waiting for the setup
    /// token to be used. The token itself is only on the terminal.
    needs_setup: bool,
    /// Whether this request is already somebody.
    signed_in: Option<Identity>,
}

pub async fn situation(St(state): St<State>, headers: HeaderMap) -> Result<Json<Situation>, Fail> {
    let mode = state.auth.mode().to_string();
    let needs_setup = state.setup.wanted();

    // Deliberately not the `Identity` extractor: this route has to answer a
    // request that has no credential, which is the whole point of it.
    let presented = crate::auth::Presented::of(&parts_of(&headers));
    let (auth, db) = (state.auth.clone(), state.db.clone());
    let signed_in = tokio::task::spawn_blocking(move || auth.identify(&presented, &db))
        .await
        .map_err(|e| Fail::internal(format!("a background task failed: {e}")))?;

    Ok(Json(Situation { mode, needs_setup, signed_in }))
}

/// Enough of a `Parts` to read headers out of. `Presented::of` looks at
/// nothing else, and building a whole request here would be pretence.
fn parts_of(headers: &HeaderMap) -> axum::http::request::Parts {
    let mut parts = axum::http::Request::builder().body(()).expect("an empty request").into_parts().0;
    parts.headers = headers.clone();
    parts
}

#[derive(serde::Deserialize)]
pub struct Credentials {
    name: String,
    password: String,
}

pub async fn sign_in(
    St(state): St<State>,
    headers: HeaderMap,
    Json(body): Json<Credentials>,
) -> Result<impl IntoResponse, Fail> {
    let agent = headers.get(USER_AGENT).and_then(|v| v.to_str().ok()).map(|s| s.to_string());
    let secure = is_https(&headers);
    let db = state.db.clone();

    let (user, token) = blocking(move || {
        users::sign_in(&db, body.name.trim(), &body.password, agent.as_deref())
    })
    .await
    // A refused sign-in is 401 and not 500, and it says the one thing it is
    // allowed to say.
    .map_err(|e| Fail(axum::http::StatusCode::UNAUTHORIZED, e.1))?;

    Ok(([(SET_COOKIE, session_cookie(&token, secure))], Json(identity_of(&user))))
}

pub async fn sign_out(
    _: Identity,
    St(state): St<State>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, Fail> {
    if let crate::auth::Presented::Cookie(token) = crate::auth::Presented::of(&parts_of(&headers)) {
        let db = state.db.clone();
        blocking(move || users::sign_out(&db, &token)).await?;
    }
    Ok(([(SET_COOKIE, cleared_cookie(is_https(&headers)))], Json(json!({ "signed_out": true }))))
}

#[derive(serde::Deserialize)]
pub struct SetupRequest {
    token: String,
    name: String,
    password: String,
}

/// Make the first account, against the token printed at startup.
///
/// The token is spent before the account is made, so two people racing to set
/// the same server up cannot both succeed. If the account then fails to be
/// created — a name with a slash in it, a password too short — the token is
/// gone and the server has to be restarted, which is the safe way round.
pub async fn setup(
    St(state): St<State>,
    headers: HeaderMap,
    Json(body): Json<SetupRequest>,
) -> Result<impl IntoResponse, Fail> {
    if !state.setup.spend(body.token.trim()) {
        return Err(Fail(
            axum::http::StatusCode::UNAUTHORIZED,
            "that is not this server's setup token. It is printed in the terminal when the \
             server starts with no accounts."
                .into(),
        ));
    }
    let secure = is_https(&headers);
    let agent = headers.get(USER_AGENT).and_then(|v| v.to_str().ok()).map(|s| s.to_string());
    let db = state.db.clone();

    let (user, token) = blocking(move || {
        let user = users::bootstrap(&db, body.name.trim(), &body.password)?;
        let (_, token) = users::sign_in(&db, &user.name, &body.password, agent.as_deref())?;
        Ok((user, token))
    })
    .await
    .map_err(|e| Fail::bad(e.1))?;

    Ok(([(SET_COOKIE, session_cookie(&token, secure))], Json(identity_of(&user))))
}

fn identity_of(user: &User) -> serde_json::Value {
    json!({ "id": user.id, "name": user.name, "role": user.role })
}

/// Whether the browser reached us over HTTPS.
///
/// The connection here is plain TCP whatever is in front of it, so the only
/// evidence is what a proxy says. `X-Forwarded-Proto` is trusted because the
/// alternative — never setting `Secure` — is worse, and because a client that
/// can forge it is already on the inside.
fn is_https(headers: &HeaderMap) -> bool {
    headers
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(',').next().is_some_and(|p| p.trim().eq_ignore_ascii_case("https")))
}

fn session_cookie(token: &str, secure: bool) -> String {
    let mut cookie = format!(
        "{COOKIE_NAME}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={}",
        SESSION_DAYS * 24 * 60 * 60
    );
    if secure {
        cookie.push_str("; Secure");
    }
    cookie
}

fn cleared_cookie(secure: bool) -> String {
    let mut cookie = format!("{COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0");
    if secure {
        cookie.push_str("; Secure");
    }
    cookie
}

// ---------------------------------------------------------------------------
// Accounts
// ---------------------------------------------------------------------------

pub async fn list_users(_: Admin, St(state): St<State>) -> Result<Json<Vec<User>>, Fail> {
    let db = state.db.clone();
    blocking(move || users::list(&db)).await.map(Json)
}

#[derive(serde::Deserialize)]
pub struct NewUser {
    name: String,
    password: String,
    #[serde(default = "user_role")]
    role: String,
    #[serde(default)]
    email: Option<String>,
}

fn user_role() -> String {
    "user".into()
}

pub async fn create_user(
    _: Admin,
    St(state): St<State>,
    Json(body): Json<NewUser>,
) -> Result<Json<User>, Fail> {
    let db = state.db.clone();
    blocking(move || {
        users::create(&db, body.name.trim(), Some(&body.password), &body.role, body.email.as_deref())
    })
    .await
    .map(Json)
    .map_err(|e| Fail::bad(e.1))
}

#[derive(serde::Deserialize)]
pub struct UserPatch {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    password: Option<String>,
}

pub async fn update_user(
    who: Admin,
    St(state): St<State>,
    Path(id): Path<i64>,
    Json(body): Json<UserPatch>,
) -> Result<Json<User>, Fail> {
    // An administrator demoting themselves while holding the only admin role
    // is already refused by `set_role`; this is the friendlier version of the
    // same rule, caught before anything is written.
    if body.role.as_deref() == Some("user") && who.0.id == Some(id) {
        return Err(Fail::bad("you cannot take your own administrator role away"));
    }
    let db = state.db.clone();
    blocking(move || {
        if let Some(password) = &body.password {
            users::set_password(&db, id, password)?;
        }
        match &body.role {
            Some(role) => users::set_role(&db, id, role),
            None => users::get(&db, id)?.ok_or_else(|| format!("there is no account {id}").into()),
        }
    })
    .await
    .map(Json)
    .map_err(|e| Fail::bad(e.1))
}

pub async fn delete_user(
    who: Admin,
    St(state): St<State>,
    Path(id): Path<i64>,
) -> Result<Json<serde_json::Value>, Fail> {
    if who.0.id == Some(id) {
        return Err(Fail::bad("you cannot delete the account you are signed in with"));
    }
    let db = state.db.clone();
    match blocking(move || users::delete(&db, id)).await.map_err(|e| Fail::bad(e.1))? {
        true => Ok(Json(json!({ "deleted": id }))),
        false => Err(Fail::missing(format!("there is no account {id}"))),
    }
}

/// Change your own password. Not an admin route: it is about yourself.
#[derive(serde::Deserialize)]
pub struct PasswordChange {
    current: String,
    new: String,
}

pub async fn change_password(
    who: Identity,
    St(state): St<State>,
    Json(body): Json<PasswordChange>,
) -> Result<Json<serde_json::Value>, Fail> {
    let Some(id) = who.id else {
        return Err(Fail::bad("there are no accounts in this mode, so there is no password"));
    };
    let db = state.db.clone();
    blocking(move || {
        // Knowing the current one, so that a browser left open is not an
        // account somebody else now owns.
        users::sign_in(&db, &who.name, &body.current, None)
            .map_err(|_| "the current password is not right")?;
        users::set_password(&db, id, &body.new)
    })
    .await
    .map_err(|e| Fail::bad(e.1))?;
    // Every session went with the change, including this one.
    Ok(Json(json!({ "changed": true, "signed_out": true })))
}

// ---------------------------------------------------------------------------
// Sessions and keys — yours, not anybody's
// ---------------------------------------------------------------------------

pub async fn list_sessions(
    who: Identity,
    St(state): St<State>,
) -> Result<Json<Vec<Session>>, Fail> {
    let Some(id) = who.id else { return Ok(Json(Vec::new())) };
    let db = state.db.clone();
    blocking(move || users::sessions(&db, id)).await.map(Json)
}

pub async fn revoke_session(
    who: Identity,
    St(state): St<State>,
    Path(hash): Path<String>,
) -> Result<Json<serde_json::Value>, Fail> {
    let Some(id) = who.id else { return Err(Fail::bad("there are no sessions in this mode")) };
    let db = state.db.clone();
    match blocking(move || users::revoke_session(&db, id, &hash)).await? {
        true => Ok(Json(json!({ "revoked": true }))),
        false => Err(Fail::missing("no such session of yours")),
    }
}

pub async fn list_keys(who: Identity, St(state): St<State>) -> Result<Json<Vec<ApiKey>>, Fail> {
    let Some(id) = who.id else { return Ok(Json(Vec::new())) };
    let db = state.db.clone();
    blocking(move || users::keys(&db, id)).await.map(Json)
}

#[derive(serde::Deserialize)]
pub struct NewKey {
    name: String,
}

/// Make a key. The token is in the answer and nowhere else, ever again.
pub async fn create_key(
    who: Identity,
    St(state): St<State>,
    Json(body): Json<NewKey>,
) -> Result<Json<serde_json::Value>, Fail> {
    let Some(id) = who.id else {
        return Err(Fail::bad(
            "there are no accounts in this mode, so a key would belong to nobody",
        ));
    };
    let db = state.db.clone();
    let (key, token) =
        blocking(move || users::create_key(&db, id, &body.name)).await.map_err(|e| Fail::bad(e.1))?;
    Ok(Json(json!({ "key": key, "token": token })))
}

pub async fn revoke_key(
    who: Identity,
    St(state): St<State>,
    Path(id): Path<i64>,
) -> Result<Json<serde_json::Value>, Fail> {
    let Some(me) = who.id else { return Err(Fail::bad("there are no keys in this mode")) };
    let db = state.db.clone();
    match blocking(move || users::revoke_key(&db, me, id)).await? {
        true => Ok(Json(json!({ "revoked": id }))),
        false => Err(Fail::missing("no such key of yours")),
    }
}

// ---------------------------------------------------------------------------
// Through an identity provider
// ---------------------------------------------------------------------------

/// Send the browser to the provider.
///
/// A redirect rather than JSON with a URL in it, so that the sign-in button
/// can be an ordinary link and works with JavaScript turned off.
pub async fn oidc_start(St(state): St<State>) -> Result<impl IntoResponse, Fail> {
    if state.auth.mode() != crate::config::Mode::Oidc {
        return Err(Fail::bad("this server does not sign in through a provider"));
    }
    let (settings, flows) = (&state.oidc.0, &state.oidc.1);
    let url = crate::oidc::start(settings, flows).await.map_err(Fail::bad)?;
    Ok(axum::response::Redirect::to(&url))
}

#[derive(serde::Deserialize)]
pub struct Callback {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    state: Option<String>,
    /// What the provider says when it refused, e.g. `access_denied`.
    #[serde(default)]
    error: Option<String>,
}

/// Where the provider sends the browser back to.
///
/// Ends in a redirect either way, because what is at the other end of it is a
/// person looking at a browser and not a program reading JSON. A failure goes
/// to the sign-in page with a reason in the query string.
pub async fn oidc_callback(
    St(state): St<State>,
    headers: HeaderMap,
    axum::extract::Query(query): axum::extract::Query<Callback>,
) -> impl IntoResponse {
    let refuse = |why: String| {
        (
            [(SET_COOKIE, cleared_cookie(is_https(&headers)))],
            axum::response::Redirect::to(&format!("/?signin_error={}", urlencode(&why))),
        )
            .into_response()
    };

    if let Some(problem) = query.error {
        return refuse(format!("the provider refused: {problem}"));
    }
    let (Some(code), Some(flow_state)) = (query.code, query.state) else {
        return refuse("the provider came back without a code".into());
    };

    let arrived =
        match crate::oidc::finish(&state.oidc.0, &state.oidc.1, &code, &flow_state).await {
            Ok(arrived) => arrived,
            Err(why) => return refuse(why),
        };

    let agent = headers.get(USER_AGENT).and_then(|v| v.to_str().ok()).map(|s| s.to_string());
    let db = state.db.clone();
    let signed = tokio::task::spawn_blocking(move || adopt(&db, &arrived, agent.as_deref())).await;

    match signed {
        Ok(Ok(token)) => (
            [(SET_COOKIE, session_cookie(&token, is_https(&headers)))],
            axum::response::Redirect::to("/"),
        )
            .into_response(),
        Ok(Err(why)) => refuse(why),
        Err(e) => refuse(format!("the sign-in was interrupted: {e}")),
    }
}

/// Find or make the account behind an address, and start a session for it.
///
/// The role comes from `kvad.toml` every time: with a provider in charge of
/// identity, the file is in charge of authority, and a role set by hand in
/// Settings lasts until the next sign-in. The one exception is the rule that
/// keeps at least one administrator — if applying the config would leave
/// none, the old role stays and the refusal is logged rather than locking
/// everybody out of their own server.
fn adopt(
    db: &crate::db::Db,
    arrived: &crate::oidc::Arrived,
    agent: Option<&str>,
) -> Result<String, String> {
    let want = if arrived.admin { "admin" } else { "user" };

    let existing = users::by_email(db, &arrived.email).map_err(|e| e.to_string())?;
    let user = match existing {
        Some(user) => {
            if user.role != want {
                if let Err(e) = users::set_role(db, user.id, want) {
                    tracing::warn!("keeping {}'s role as {}: {e}", user.name, user.role);
                }
            }
            user
        }
        None => {
            // A name that is free: `ada`, then `ada-2`, and so on. Two people
            // at different providers can share a local part.
            let base = crate::oidc::name_for(&arrived.email, arrived.name.as_deref());
            let mut name = base.clone();
            for n in 2..100 {
                match users::by_name(db, &name).map_err(|e| e.to_string())? {
                    None => break,
                    Some(_) => name = format!("{base}-{n}"),
                }
            }
            users::create(db, &name, None, want, Some(&arrived.email)).map_err(|e| e.to_string())?
        }
    };

    users::open_session(db, user.id, agent).map_err(|e| e.to_string())
}

/// Percent-encode for a query string. Only what a message can contain.
fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            b' ' => "+".to_string(),
            other => format!("%{other:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three attributes that make the cookie safe, and the one that is
    /// conditional because setting it always would break `http://localhost`.
    #[test]
    fn the_session_cookie_is_httponly_lax_and_scoped_to_the_site() {
        let plain = session_cookie("abc", false);
        assert!(plain.starts_with("kvad_session=abc;"));
        assert!(plain.contains("HttpOnly"), "{plain}");
        assert!(plain.contains("SameSite=Lax"), "{plain}");
        assert!(plain.contains("Path=/"), "{plain}");
        assert!(!plain.contains("Secure"), "a Secure cookie over http is thrown away: {plain}");

        assert!(session_cookie("abc", true).contains("; Secure"));
        assert!(cleared_cookie(false).contains("Max-Age=0"));
    }

    #[test]
    fn https_is_believed_only_when_a_proxy_says_so() {
        let with = |value: &str| {
            let mut h = HeaderMap::new();
            h.insert("x-forwarded-proto", value.parse().unwrap());
            is_https(&h)
        };
        assert!(!is_https(&HeaderMap::new()));
        assert!(with("https"));
        assert!(with("HTTPS"));
        // A chain of proxies: the first entry is the client's own hop.
        assert!(with("https, http"));
        assert!(!with("http"));
        assert!(!with("http, https"));
    }
}
