//! What each mode does and does not accept.
//!
//! Written against [`Provider::identify`] and the two checks around it rather
//! than against a running server, because the interesting cases are the ones
//! a server makes hard to produce: a cookie from somebody else's page, a
//! password for an account that does not exist, a key that was revoked a
//! moment ago.
//!
//! Every test here should fail if the check it is about is deleted. That was
//! confirmed by deleting each of them in turn; the comments name which test
//! catches which.

use super::*;
use axum::http::Request;

fn db_with_people() -> Db {
    let db = Db::in_memory().unwrap();
    users::create(&db, "ada", Some("lovelace-1843"), "admin", None).unwrap();
    users::create(&db, "bob", Some("builder-2024"), "user", None).unwrap();
    db
}

/// A request, with whatever headers a test needs.
fn request(method: &str, headers: &[(&str, &str)]) -> Parts {
    let mut b = Request::builder().method(method).uri("http://kvad.test/api/models");
    for (k, v) in headers {
        b = b.header(*k, *v);
    }
    b.header("host", "kvad.test").body(()).unwrap().into_parts().0
}

fn basic_header(name: &str, password: &str) -> String {
    format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode(format!("{name}:{password}"))
    )
}

// ---------------------------------------------------------------------------
// Reading the credential out of a request
// ---------------------------------------------------------------------------

#[test]
fn a_request_presents_one_thing_and_the_deliberate_one_wins() {
    assert_eq!(Presented::of(&request("GET", &[])), Presented::Nothing);

    assert_eq!(
        Presented::of(&request("GET", &[("cookie", "kvad_session=abc")])),
        Presented::Cookie("abc".into())
    );
    // Among several cookies, and with the spacing a browser actually sends.
    assert_eq!(
        Presented::of(&request("GET", &[("cookie", "theme=dim; kvad_session=abc; other=1")])),
        Presented::Cookie("abc".into())
    );
    // A cookie whose name merely ends in ours is not ours.
    assert_eq!(
        Presented::of(&request("GET", &[("cookie", "not_kvad_session=abc")])),
        Presented::Nothing
    );

    assert_eq!(
        Presented::of(&request("GET", &[("authorization", "Bearer key-123")])),
        Presented::Bearer("key-123".into())
    );
    // Schemes are case-insensitive in HTTP, and clients disagree about case.
    assert_eq!(
        Presented::of(&request("GET", &[("authorization", "bearer key-123")])),
        Presented::Bearer("key-123".into())
    );

    assert_eq!(
        Presented::of(&request("GET", &[("authorization", &basic_header("ada", "pw:with:colons"))])),
        Presented::Basic { name: "ada".into(), password: "pw:with:colons".into() }
    );

    // A header that was put there on purpose beats a cookie the browser
    // attached on its own.
    assert_eq!(
        Presented::of(&request(
            "GET",
            &[("authorization", "Bearer key-123"), ("cookie", "kvad_session=abc")]
        )),
        Presented::Bearer("key-123".into())
    );

    // Nonsense in the header is nothing, not a credential to be checked.
    assert_eq!(Presented::of(&request("GET", &[("authorization", "Bearer ")])), Presented::Nothing);
    assert_eq!(Presented::of(&request("GET", &[("authorization", "Basic !!!!")])), Presented::Nothing);
    assert_eq!(Presented::of(&request("GET", &[("authorization", "Weird abc")])), Presented::Nothing);
}

/// Only the cookie rides along on its own, so only the cookie needs the
/// origin check.
#[test]
fn only_a_cookie_is_forgeable() {
    assert!(Presented::Cookie("x".into()).forgeable());
    assert!(!Presented::Bearer("x".into()).forgeable());
    assert!(!Presented::Basic { name: "a".into(), password: "b".into() }.forgeable());
    assert!(!Presented::Nothing.forgeable());
}

// ---------------------------------------------------------------------------
// Cross-site requests
// ---------------------------------------------------------------------------

/// Delete the comparison in `origin_is_ours` and this fails: a POST from
/// evil.example carrying the victim's cookie would be accepted.
#[test]
fn a_mutating_request_from_somewhere_else_is_not_ours() {
    // Reading is safe whatever the origin; that is what makes a link work.
    assert!(origin_is_ours(&request("GET", &[("origin", "https://evil.example")])));
    assert!(origin_is_ours(&request("HEAD", &[("origin", "https://evil.example")])));

    // Writing is not.
    assert!(!origin_is_ours(&request("POST", &[("origin", "https://evil.example")])));
    assert!(!origin_is_ours(&request("DELETE", &[("origin", "https://evil.example")])));
    assert!(!origin_is_ours(&request("POST", &[("origin", "http://kvad.test.evil.example")])));

    // Our own page, over either scheme, with or without a path.
    assert!(origin_is_ours(&request("POST", &[("origin", "http://kvad.test")])));
    assert!(origin_is_ours(&request("POST", &[("origin", "https://kvad.test")])));
    assert!(origin_is_ours(&request("POST", &[("referer", "http://kvad.test/chat")])));

    // A missing Origin cannot be told from a stripped one, so it is refused.
    assert!(!origin_is_ours(&request("POST", &[])));

    // A port is part of the origin: 8080 is not 8081.
    let mut parts = request("POST", &[("origin", "http://kvad.test:8081")]);
    parts.headers.insert("host", "kvad.test:8080".parse().unwrap());
    assert!(!origin_is_ours(&parts));
    parts.headers.insert("origin", "http://kvad.test:8080".parse().unwrap());
    assert!(origin_is_ours(&parts));
}

// ---------------------------------------------------------------------------
// mode = "none"
// ---------------------------------------------------------------------------

#[test]
fn with_no_auth_everyone_is_the_operator_whatever_they_send() {
    let db = Db::in_memory().unwrap();
    for presented in [
        Presented::Nothing,
        // An OpenAI client sends a bearer token whether or not anybody asked
        // it to. Refusing those would break /v1 on exactly the setup it is
        // most likely to be used on.
        Presented::Bearer("sk-whatever".into()),
        Presented::Cookie("stale".into()),
        Presented::Basic { name: "ada".into(), password: "wrong".into() },
    ] {
        let who = NoAuth.identify(&presented, &db).expect("refused in mode none");
        assert!(who.is_admin());
        assert_eq!(who.id, None);
    }
    assert!(NoAuth.challenge().is_none());
}

// ---------------------------------------------------------------------------
// mode = "local"
// ---------------------------------------------------------------------------

/// Delete the `from_session` lookup and this fails: any cookie value would be
/// a sign-in.
#[test]
fn local_accepts_a_live_session_and_nothing_that_looks_like_one() {
    let db = db_with_people();
    let (ada, token) = users::sign_in(&db, "ada", "lovelace-1843", None).unwrap();

    let who =
        Local.identify(&Presented::Cookie(token.clone()), &db).expect("the session was refused");
    assert_eq!(who.id, Some(ada.id));
    assert_eq!(who.role, Role::Admin);

    // A token nobody issued, and one that has been signed out.
    assert!(Local.identify(&Presented::Cookie(crate::secret::token()), &db).is_none());
    users::sign_out(&db, &token).unwrap();
    assert!(Local.identify(&Presented::Cookie(token), &db).is_none());

    // No credential at all is nobody, and there is no challenge — a browser
    // shown one pops up its own dialog instead of our login page.
    assert!(Local.identify(&Presented::Nothing, &db).is_none());
    assert!(Local.challenge().is_none());
}

/// Delete the arm that ignores `Presented::Basic` in `Local` and this fails:
/// a password would be accepted on every request in a mode built so that it
/// is not.
#[test]
fn local_does_not_take_a_password_on_every_request() {
    let db = db_with_people();
    let right = Presented::Basic { name: "ada".into(), password: "lovelace-1843".into() };
    assert!(Local.identify(&right, &db).is_none(), "local accepted Basic");
}

// ---------------------------------------------------------------------------
// mode = "basic"
// ---------------------------------------------------------------------------

/// Delete the `verify_password` call and this fails: any password would do.
#[test]
fn basic_needs_the_right_password_for_an_account_that_exists() {
    let db = db_with_people();

    let who = Basic
        .identify(&Presented::Basic { name: "bob".into(), password: "builder-2024".into() }, &db)
        .expect("the right password was refused");
    assert_eq!(who.name, "bob");
    assert_eq!(who.role, Role::User, "bob is not an administrator");

    for (name, password) in
        [("bob", "wrong-password"), ("nobody", "builder-2024"), ("bob", ""), ("", "")]
    {
        let presented = Presented::Basic { name: name.into(), password: password.into() };
        assert!(Basic.identify(&presented, &db).is_none(), "{name}/{password} was accepted");
    }

    // A browser that signed in through the login page keeps working, so the
    // Settings page is usable in this mode too.
    let (_, token) = users::sign_in(&db, "ada", "lovelace-1843", None).unwrap();
    assert!(Basic.identify(&Presented::Cookie(token), &db).is_some());

    assert!(Basic.challenge().unwrap().starts_with("Basic realm="));
}

/// An account with no password — one that exists to sign in some other way —
/// must not be signed in by sending an empty one.
#[test]
fn an_account_without_a_password_cannot_be_signed_in_to() {
    let db = Db::in_memory().unwrap();
    users::create(&db, "claimed", None, "admin", Some("a@example.com")).unwrap();
    for password in ["", "anything"] {
        let presented = Presented::Basic { name: "claimed".into(), password: password.into() };
        assert!(Basic.identify(&presented, &db).is_none(), "signed in with `{password}`");
    }
}

// ---------------------------------------------------------------------------
// API keys, in every mode that has accounts
// ---------------------------------------------------------------------------

/// Delete the `from_key` lookup and this fails: a bearer token would be
/// nobody and every script would stop working.
#[test]
fn a_key_signs_in_as_whoever_made_it_until_it_is_revoked() {
    let db = db_with_people();
    let bob = users::by_name(&db, "bob").unwrap().unwrap();
    let (key, token) = users::create_key(&db, bob.id, "script").unwrap();

    for provider in [&Local as &dyn Provider, &Basic as &dyn Provider] {
        let who =
            provider.identify(&Presented::Bearer(token.clone()), &db).expect("a live key was refused");
        assert_eq!(who.id, Some(bob.id));
        // A key carries its owner's role and not more than it.
        assert_eq!(who.role, Role::User);
    }

    assert!(Local.identify(&Presented::Bearer(crate::secret::token()), &db).is_none());

    users::revoke_key(&db, bob.id, key.id).unwrap();
    assert!(Local.identify(&Presented::Bearer(token.clone()), &db).is_none());
    assert!(Basic.identify(&Presented::Bearer(token), &db).is_none());
}

/// Deleting an account has to take everything that could sign in as it.
#[test]
fn a_deleted_account_takes_its_sessions_and_keys_with_it() {
    let db = db_with_people();
    let bob = users::by_name(&db, "bob").unwrap().unwrap();
    let (_, session) = users::sign_in(&db, "bob", "builder-2024", None).unwrap();
    let (_, key) = users::create_key(&db, bob.id, "script").unwrap();

    assert!(users::delete(&db, bob.id).unwrap());
    assert!(Local.identify(&Presented::Cookie(session), &db).is_none());
    assert!(Local.identify(&Presented::Bearer(key), &db).is_none());
}

// ---------------------------------------------------------------------------
// Roles
// ---------------------------------------------------------------------------

#[test]
fn an_unreadable_role_is_the_smaller_one() {
    assert_eq!(Role::parse("admin"), Role::Admin);
    assert_eq!(Role::parse("user"), Role::User);
    // The column is constrained to two values, so this cannot happen. If it
    // somehow does, it must not be the one that can change the machine.
    assert_eq!(Role::parse("wizard"), Role::User);
    assert_eq!(Role::parse(""), Role::User);
}

/// Delete the `is_admin` check in the `Admin` extractor and this fails: every
/// admin-only route would take anybody signed in.
///
/// Through the extractor rather than through `Identity::is_admin`, because
/// the extractor is what the routes are written against and is the thing that
/// can be got wrong.
#[tokio::test]
async fn the_admin_extractor_takes_administrators_and_nobody_else() {
    let db = db_with_people();

    struct As(&'static str);
    impl Provider for As {
        fn mode(&self) -> Mode {
            Mode::Local
        }
        fn identify(&self, _: &Presented, db: &Db) -> Option<Identity> {
            users::by_name(db, self.0).ok().flatten().map(|u| Identity::of(&u))
        }
    }
    let state = |name: &'static str| State {
        db: db.clone(),
        auth: std::sync::Arc::new(As(name)),
        engine: std::sync::Arc::new(crate::scheduler::Scheduler::spawn(Box::new(|_, _, _, _| {
            Err("no backend in tests".into())
        }))),
        setup: std::sync::Arc::new(Setup::default()),
        oidc: std::sync::Arc::new(Default::default()),
        started: std::time::Instant::now(),
    };

    // Ada is an administrator...
    let ada = state("ada");
    assert!(Admin::from_request_parts(&mut request("GET", &[]), &ada).await.is_ok());

    // ...and bob is signed in, which is not the same thing.
    let bob = state("bob");
    let who = Identity::from_request_parts(&mut request("GET", &[]), &bob).await.unwrap();
    assert_eq!(who.name, "bob");
    let Err(denied) = Admin::from_request_parts(&mut request("GET", &[]), &bob).await else {
        panic!("a plain user passed the admin extractor");
    };
    // 403 and not 401: "you, but no", rather than "who are you".
    assert_eq!(denied.0, StatusCode::FORBIDDEN);
}

/// Delete the `forgeable` check in the extractor and this fails: a cookie
/// from somebody else's page would act on the signed-in person's behalf.
#[tokio::test]
async fn a_cross_site_post_is_refused_before_the_cookie_is_even_looked_up() {
    let db = db_with_people();
    let (_, token) = users::sign_in(&db, "ada", "lovelace-1843", None).unwrap();

    let state = State {
        db: db.clone(),
        auth: std::sync::Arc::new(Local),
        engine: std::sync::Arc::new(crate::scheduler::Scheduler::spawn(Box::new(|_, _, _, _| {
            Err("no backend in tests".into())
        }))),
        setup: std::sync::Arc::new(Setup::default()),
        oidc: std::sync::Arc::new(Default::default()),
        started: std::time::Instant::now(),
    };
    let cookie = format!("kvad_session={token}");

    // The same live session, from our page and from somewhere else.
    let ours = &[("cookie", cookie.as_str()), ("origin", "http://kvad.test")];
    let theirs = &[("cookie", cookie.as_str()), ("origin", "https://evil.example")];

    assert!(Identity::from_request_parts(&mut request("POST", ours), &state).await.is_ok());
    let Err(denied) = Identity::from_request_parts(&mut request("POST", theirs), &state).await
    else {
        panic!("a cross-site POST was accepted");
    };
    assert_eq!(denied.0, StatusCode::FORBIDDEN);

    // Reading is fine from anywhere: that is what makes a link work.
    assert!(Identity::from_request_parts(&mut request("GET", theirs), &state).await.is_ok());

    // And a bearer token is not forgeable, so it is not caught by the check.
    let ada = users::by_name(&db, "ada").unwrap().unwrap();
    let (_, key) = users::create_key(&db, ada.id, "script").unwrap();
    let bearer = format!("Bearer {key}");
    let with_key = &[("authorization", bearer.as_str())];
    assert!(Identity::from_request_parts(&mut request("POST", with_key), &state).await.is_ok());
}

// ---------------------------------------------------------------------------
// The setup token
// ---------------------------------------------------------------------------

#[test]
fn the_setup_token_works_once_and_only_for_itself() {
    let setup = Setup::default();
    assert!(!setup.wanted());
    assert!(!setup.spend("anything"), "an unissued token was spent");

    let token = setup.issue();
    assert_eq!(token.len(), 64);
    assert!(setup.wanted());
    assert!(!setup.spend(&crate::secret::token()), "the wrong token was accepted");
    assert!(setup.spend(&token));
    assert!(!setup.spend(&token), "the token was spent twice");
    assert!(!setup.wanted());
}

// ---------------------------------------------------------------------------
// The modes that are not built yet
// ---------------------------------------------------------------------------

/// Every mode the config file can name is a mode this server can build, so a
/// `kvad.toml` that asks for one cannot be refused at startup for a reason
/// nobody can act on.
#[test]
fn every_named_mode_has_a_provider() {
    let settings = crate::config::Oidc::default();
    for mode in [Mode::None, Mode::Local, Mode::Basic, Mode::Oidc] {
        let built = provider(mode, &settings).unwrap_or_else(|e| panic!("{mode}: {e}"));
        assert_eq!(built.mode(), mode);
    }
}

/// An identity provider decides who somebody is at the callback, not on every
/// request; afterwards it is an ordinary session, and nothing else gets in.
#[test]
fn oidc_accepts_a_session_or_a_key_and_nothing_else() {
    let db = db_with_people();
    let (ada, token) = users::sign_in(&db, "ada", "lovelace-1843", None).unwrap();
    assert_eq!(Oidc.identify(&Presented::Cookie(token), &db).unwrap().id, Some(ada.id));

    let (_, key) = users::create_key(&db, ada.id, "script").unwrap();
    assert!(Oidc.identify(&Presented::Bearer(key), &db).is_some());

    // A password is not a way in when the provider is somewhere else.
    let password = Presented::Basic { name: "ada".into(), password: "lovelace-1843".into() };
    assert!(Oidc.identify(&password, &db).is_none());
    assert!(Oidc.identify(&Presented::Nothing, &db).is_none());
}
