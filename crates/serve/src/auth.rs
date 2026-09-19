//! Who a request is, and the seam the other modes will fit into.
//!
//! One mode is implemented: [`Mode::None`], where anyone who can reach the
//! socket is an administrator. That is the whole of it, and it is only safe
//! because [`crate::config::Config::check`] refuses to bind anything but
//! loopback in that mode.
//!
//! The reason there is a seam at all, this early, is that the alternative is
//! worse. Handlers written against a bare `Router` and retrofitted with auth
//! later are handlers where "did anyone check?" has to be answered one at a
//! time. Written against [`Identity`] from the start, a handler that forgot
//! does not compile, and the modes that arrive in Phase 3 change this file
//! and nothing else.
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
use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use axum::http::StatusCode;

/// What somebody is allowed to do.
///
/// Two, and no more until there is a reason: `admin` can change the machine —
/// pull models, train, edit settings — and `user` can talk to what is already
/// there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Admin,
    // Nothing hands one out until a mode that can tell people apart exists;
    // `NoAuth` has exactly one user and they own the machine.
    #[allow(dead_code)]
    User,
}

/// Who a request is from, once the configured mode has decided.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Identity {
    pub name: String,
    pub role: Role,
}

impl Identity {
    /// The identity every request has when there is no authentication: the
    /// person at the keyboard, who is the only one who can reach loopback.
    fn local_operator() -> Self {
        Identity { name: "local".into(), role: Role::Admin }
    }

    #[allow(dead_code)] // Used by the `Admin` extractor, which no route needs yet.
    pub fn is_admin(&self) -> bool {
        self.role == Role::Admin
    }
}

/// Decides who a request is from. One per mode.
///
/// A trait rather than a `match` in the extractor so that each mode's rules
/// are in one place and can be tested without a server: Phase 3 adds a type
/// per mode and this file's shape does not change.
pub trait Provider: Send + Sync + 'static {
    /// The identity behind these request parts, or `None` for a request that
    /// did not prove one.
    fn identify(&self, parts: &Parts) -> Option<Identity>;

    /// What the browser should be told to do about a request with no
    /// identity. `None` is a plain 401; a mode with a login page returns the
    /// value of a `WWW-Authenticate` header.
    fn challenge(&self) -> Option<&str> {
        None
    }
}

/// Everyone is the operator. Safe only on loopback, which is enforced before
/// the socket is opened rather than here.
pub struct NoAuth;

impl Provider for NoAuth {
    fn identify(&self, _parts: &Parts) -> Option<Identity> {
        Some(Identity::local_operator())
    }
}

/// The provider for a mode, or an error naming the mode that has none yet.
pub fn provider(mode: Mode) -> Result<Box<dyn Provider>, String> {
    match mode {
        Mode::None => Ok(Box::new(NoAuth)),
        other => Err(format!("auth mode `{other}` is not implemented yet")),
    }
}

/// Everything a handler needs that is not in the request.
///
/// Passed to every route as axum's state, and what the extractors below reach
/// into. It grows a scheduler in Phase 2.
#[derive(Clone)]
pub struct State {
    pub db: crate::db::Db,
    pub auth: std::sync::Arc<dyn Provider>,
    pub started: std::time::Instant,
}

/// An error a rejected request turns into: a status and a short reason.
///
/// Deliberately says nothing about *why* beyond the status. A 401 that
/// explained itself would be a way to ask questions about the user table.
#[derive(Debug)]
pub struct Denied(StatusCode, Option<String>);

impl axum::response::IntoResponse for Denied {
    fn into_response(self) -> axum::response::Response {
        let body = axum::Json(serde_json::json!({
            "error": match self.0 {
                StatusCode::UNAUTHORIZED => "not signed in",
                _ => "not allowed",
            }
        }));
        match self.1 {
            Some(challenge) => (
                self.0,
                [(axum::http::header::WWW_AUTHENTICATE, challenge)],
                body,
            )
                .into_response(),
            None => (self.0, body).into_response(),
        }
    }
}

impl FromRequestParts<State> for Identity {
    type Rejection = Denied;

    async fn from_request_parts(parts: &mut Parts, state: &State) -> Result<Self, Self::Rejection> {
        state.auth.identify(parts).ok_or_else(|| {
            Denied(StatusCode::UNAUTHORIZED, state.auth.challenge().map(str::to_string))
        })
    }
}

/// An [`Identity`] that is also an administrator.
///
/// A separate extractor rather than a check inside the handler, so that a
/// handler which needs an admin cannot be written without asking for one.
/// Nothing needs one yet — every route in this phase is readable by anyone
/// who got past the door — but the extractor is what the Models and Settings
/// handlers will be written against.
#[allow(dead_code)]
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
mod tests {
    use super::*;

    fn parts() -> Parts {
        axum::http::Request::builder().uri("/api/health").body(()).unwrap().into_parts().0
    }

    #[test]
    fn with_no_auth_every_request_is_the_local_operator() {
        let who = NoAuth.identify(&parts()).unwrap();
        assert_eq!(who.name, "local");
        assert!(who.is_admin());
        assert!(NoAuth.challenge().is_none());
    }

    /// The point of the extractor: a signed-in user who is not an
    /// administrator gets 403, and does so without the handler being written
    /// to check.
    ///
    /// Worth testing now rather than when the first admin-only route lands,
    /// because by then the test would be about that route instead of about
    /// this rule.
    #[tokio::test]
    async fn an_ordinary_user_is_not_an_admin() {
        struct AsUser;
        impl Provider for AsUser {
            fn identify(&self, _: &Parts) -> Option<Identity> {
                Some(Identity { name: "someone".into(), role: Role::User })
            }
        }
        let state = State {
            db: crate::db::Db::in_memory().unwrap(),
            auth: std::sync::Arc::new(AsUser),
            started: std::time::Instant::now(),
        };

        // Signed in, so `Identity` is happy...
        let who = Identity::from_request_parts(&mut parts(), &state).await.unwrap();
        assert_eq!(who.name, "someone");
        assert!(!who.is_admin());

        // ...and `Admin` is not, with a status that says "you, but no" rather
        // than "who are you".
        let Err(denied) = Admin::from_request_parts(&mut parts(), &state).await else {
            panic!("a plain user passed the admin extractor");
        };
        assert_eq!(denied.0, StatusCode::FORBIDDEN);
    }

    /// A request nobody can identify gets 401, not an empty identity.
    #[tokio::test]
    async fn an_unidentified_request_is_refused() {
        struct Nobody;
        impl Provider for Nobody {
            fn identify(&self, _: &Parts) -> Option<Identity> {
                None
            }
            fn challenge(&self) -> Option<&str> {
                Some("Basic realm=\"kvad\"")
            }
        }
        let state = State {
            db: crate::db::Db::in_memory().unwrap(),
            auth: std::sync::Arc::new(Nobody),
            started: std::time::Instant::now(),
        };
        let Err(denied) = Identity::from_request_parts(&mut parts(), &state).await else {
            panic!("an unidentified request was let through");
        };
        assert_eq!(denied.0, StatusCode::UNAUTHORIZED);
        assert_eq!(denied.1.as_deref(), Some("Basic realm=\"kvad\""));
    }

    /// Each unbuilt mode is refused by name, so a config that asks for one
    /// fails where somebody can read the reason.
    #[test]
    fn the_modes_that_do_not_exist_yet_say_so() {
        assert!(provider(Mode::None).is_ok());
        for mode in [Mode::Local, Mode::Basic, Mode::Oidc] {
            let Err(err) = provider(mode) else { panic!("{mode} claimed to be implemented") };
            assert!(err.contains(&mode.to_string()), "{err}");
        }
    }
}
