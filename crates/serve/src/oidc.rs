//! Signing in through somebody else's identity provider.
//!
//! Authorization code with PKCE, which is the flow to use in 2026 whatever
//! the provider's documentation still shows. The browser goes to the
//! provider, comes back with a code, and this server swaps that code for an
//! ID token over a connection the browser is not part of.
//!
//! # What is checked, and in what order
//!
//! 1. **`state`** — a value we made and remembered. It ties the callback to a
//!    sign-in this server started, so a callback nobody asked for is refused
//!    before anything is spent on it. Each one is good once.
//! 2. **PKCE** — the verifier for that state. A code intercepted on its way
//!    back cannot be redeemed without it.
//! 3. **The ID token's signature, audience, expiry and `nonce`** — all by
//!    `openidconnect` against the provider's published keys. Nothing here
//!    reads a claim until that has happened.
//! 4. **The allow-list** — the provider says who somebody is, not whether
//!    they may use this machine. `accounts.google.com` will happily
//!    authenticate every Google account in the world.
//!
//! # Roles
//!
//! With an identity provider, `kvad.toml` is the source of truth about who is
//! an administrator: the role is recomputed from `admin_emails` and
//! `admin_roles` at every sign-in and written back. Changing somebody's role
//! in Settings will therefore last until they sign in again. That is the
//! point of putting it in the provider's hands.
//!
//! # What is not tested
//!
//! The round trip. Everything below that can be tested without a provider is
//! — the flow store, the allow-list, the claim reading, the role mapping —
//! but no test here has ever spoken to a real identity provider, and that
//! should be believed.

use crate::config::Oidc as Settings;
use base64::Engine;
use openidconnect::core::{CoreAuthenticationFlow, CoreClient, CoreProviderMetadata};
use openidconnect::{
    AuthorizationCode, ClientId, ClientSecret, CsrfToken, IssuerUrl, Nonce, OAuth2TokenResponse,
    PkceCodeChallenge, PkceCodeVerifier, RedirectUrl, Scope, TokenResponse,
};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How long a half-finished sign-in is kept. Long enough to type a password
/// and answer a second factor, short enough that an abandoned one is gone.
const FLOW_TTL: Duration = Duration::from_secs(10 * 60);

/// One sign-in in progress, remembered between the redirect out and the
/// callback back.
struct Flow {
    verifier: String,
    nonce: Nonce,
    started: Instant,
}

/// The sign-ins this server has started and not yet finished.
///
/// In memory, not in the database: they are worthless after ten minutes and
/// a restart is allowed to lose them — somebody whose sign-in was interrupted
/// by a restart presses the button again.
#[derive(Default)]
pub struct Flows(Mutex<HashMap<String, Flow>>);

impl Flows {
    fn remember(&self, state: String, verifier: String, nonce: Nonce) {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        // Swept here rather than on a timer: the map only grows when somebody
        // starts a sign-in, so that is when it is worth tidying.
        held.retain(|_, f| f.started.elapsed() < FLOW_TTL);
        held.insert(state, Flow { verifier, nonce, started: Instant::now() });
    }

    /// Take the flow for `state`, if there is a live one. Good once: a
    /// callback replayed with the same state finds nothing.
    fn take(&self, state: &str) -> Option<(PkceCodeVerifier, Nonce)> {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let flow = held.remove(state)?;
        (flow.started.elapsed() < FLOW_TTL)
            .then(|| (PkceCodeVerifier::new(flow.verifier), flow.nonce))
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

/// An HTTP client for talking to the provider.
///
/// Redirects are off: following one from a URL the provider chose is how an
/// OIDC client becomes a way to make this server fetch arbitrary addresses.
fn http() -> Result<openidconnect::reqwest::Client, String> {
    openidconnect::reqwest::ClientBuilder::new()
        .redirect(openidconnect::reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| format!("could not build an HTTP client: {e}"))
}

/// Ask the provider what its endpoints are, and build a client.
///
/// Done on demand rather than at startup, so that a provider which is down
/// makes signing in fail with a message rather than making the server refuse
/// to boot.
async fn client(settings: &Settings) -> Result<CoreClient<
    openidconnect::EndpointSet,
    openidconnect::EndpointNotSet,
    openidconnect::EndpointNotSet,
    openidconnect::EndpointNotSet,
    openidconnect::EndpointMaybeSet,
    openidconnect::EndpointMaybeSet,
>, String> {
    let http = http()?;
    let issuer = IssuerUrl::new(settings.issuer.trim().to_string())
        .map_err(|e| format!("auth.oidc.issuer is not a URL: {e}"))?;
    let metadata = CoreProviderMetadata::discover_async(issuer, &http)
        .await
        .map_err(|e| format!("could not read the provider's configuration: {e}"))?;
    let redirect = RedirectUrl::new(settings.redirect_url.trim().to_string())
        .map_err(|e| format!("auth.oidc.redirect_url is not a URL: {e}"))?;

    let secret = (!settings.client_secret.trim().is_empty())
        .then(|| ClientSecret::new(settings.client_secret.trim().to_string()));
    Ok(
        CoreClient::from_provider_metadata(
            metadata,
            ClientId::new(settings.client_id.trim().to_string()),
            secret,
        )
        .set_redirect_uri(redirect),
    )
}

/// Where to send the browser to start signing in.
pub async fn start(settings: &Settings, flows: &Flows) -> Result<String, String> {
    let client = client(settings).await?;
    let (challenge, verifier) = PkceCodeChallenge::new_random_sha256();

    let mut request = client
        .authorize_url(CoreAuthenticationFlow::AuthorizationCode, CsrfToken::new_random, Nonce::new_random)
        // `openid` is implied; `email` is what the allow-list is checked
        // against, so without it nobody could be let in at all.
        .add_scope(Scope::new("email".into()))
        .add_scope(Scope::new("profile".into()))
        .set_pkce_challenge(challenge);
    for scope in &settings.scopes {
        request = request.add_scope(Scope::new(scope.clone()));
    }

    let (url, state, nonce) = request.url();
    flows.remember(state.secret().clone(), verifier.secret().clone(), nonce);
    Ok(url.to_string())
}

/// Who came back, once everything about them has been checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arrived {
    pub email: String,
    pub name: Option<String>,
    pub admin: bool,
}

/// Finish a sign-in: swap the code for a token, check it, and say who it is.
pub async fn finish(
    settings: &Settings,
    flows: &Flows,
    code: &str,
    state: &str,
) -> Result<Arrived, String> {
    let Some((verifier, nonce)) = flows.take(state) else {
        return Err("this sign-in was not started here, or it took too long. Try again.".into());
    };

    let client = client(settings).await?;
    let http = http()?;
    let tokens = client
        .exchange_code(AuthorizationCode::new(code.to_string()))
        .map_err(|e| format!("could not ask for a token: {e}"))?
        .set_pkce_verifier(verifier)
        .request_async(&http)
        .await
        .map_err(|e| format!("the provider refused the code: {e}"))?;

    let id_token = tokens.id_token().ok_or("the provider returned no ID token")?;
    let verifier = client.id_token_verifier();
    // Signature, issuer, audience, expiry and nonce, all at once. Nothing
    // below reads a claim until this has returned.
    let claims = id_token
        .claims(&verifier, &nonce)
        .map_err(|e| format!("the ID token did not check out: {e}"))?;

    // The access token could have been swapped for somebody else's; the ID
    // token carries a hash of the one it was issued with.
    if let Some(expected) = claims.access_token_hash() {
        let actual = openidconnect::AccessTokenHash::from_token(
            tokens.access_token(),
            id_token.signing_alg().map_err(|e| e.to_string())?,
            id_token.signing_key(&verifier).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if actual != *expected {
            return Err("the access token does not belong to this ID token".into());
        }
    }

    let email = claims
        .email()
        .map(|e| e.as_str().trim().to_ascii_lowercase())
        .filter(|e| !e.is_empty())
        .ok_or("the provider did not say what this account's address is, so it cannot be \
                checked against the allow-list")?;

    if !settings.allows(&email) {
        // Named, because the alternative is somebody staring at "refused"
        // while the answer is one line of kvad.toml.
        return Err(format!("{email} is not on this server's allow-list"));
    }

    let roles = match &settings.role_claim {
        None => Vec::new(),
        Some(claim) => roles_from(&id_token.to_string(), claim),
    };

    Ok(Arrived {
        admin: settings.is_admin(&email, &roles),
        name: claims
            .preferred_username()
            .map(|n| n.as_str().to_string())
            .or_else(|| claims.name().and_then(|n| n.get(None)).map(|n| n.as_str().to_string())),
        email,
    })
}

/// The values of one claim in an ID token, as a list of strings.
///
/// Read out of the token's payload directly, because `openidconnect`'s typed
/// claims cover the standard set and a `groups` claim is not in it. Safe
/// *only because* the caller has already had the signature, audience, expiry
/// and nonce checked: this decodes the same bytes that verification passed,
/// and must never be called on a token that has not been through it.
///
/// A claim holding one string is treated as a list of one, because providers
/// disagree about which they send.
fn roles_from(jwt: &str, claim: &str) -> Vec<String> {
    let Some(payload) = jwt.split('.').nth(1) else { return Vec::new() };
    let Ok(bytes) = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(payload) else {
        return Vec::new();
    };
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Vec::new();
    };
    match json.get(claim) {
        Some(serde_json::Value::String(one)) => vec![one.clone()],
        Some(serde_json::Value::Array(many)) => {
            many.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()
        }
        _ => Vec::new(),
    }
}

/// The account name an address becomes, for somebody who has never signed in
/// here before.
///
/// The local part, with anything a name may not contain replaced, and the
/// whole address if that leaves nothing usable.
pub fn name_for(email: &str, offered: Option<&str>) -> String {
    let candidate = offered
        .map(str::to_string)
        .unwrap_or_else(|| email.split('@').next().unwrap_or(email).to_string());
    let cleaned: String = candidate
        .chars()
        .map(|c| match c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '@') {
            true => c,
            false => '-',
        })
        .take(64)
        .collect();
    match cleaned.trim_matches('-').is_empty() {
        true => email.chars().take(64).collect(),
        false => cleaned,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(claims: serde_json::Value) -> String {
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(claims.to_string());
        format!("header.{payload}.signature")
    }

    /// A state is good once, and only while it is fresh.
    #[test]
    fn a_sign_in_can_only_be_finished_once() {
        let flows = Flows::default();
        flows.remember("state-1".into(), "verifier-1".into(), Nonce::new("n".into()));
        assert_eq!(flows.len(), 1);

        assert!(flows.take("state-2").is_none(), "an unknown state was accepted");
        let (verifier, _) = flows.take("state-1").expect("a live state was refused");
        assert_eq!(verifier.secret(), "verifier-1");
        assert!(flows.take("state-1").is_none(), "a state was spent twice");
        assert_eq!(flows.len(), 0);
    }

    /// Half-finished sign-ins do not pile up forever, and an old one is not
    /// accepted even before it is swept.
    #[test]
    fn an_abandoned_sign_in_expires() {
        let flows = Flows::default();
        {
            let mut held = flows.0.lock().unwrap();
            held.insert(
                "stale".into(),
                Flow {
                    verifier: "v".into(),
                    nonce: Nonce::new("n".into()),
                    started: Instant::now() - FLOW_TTL - Duration::from_secs(1),
                },
            );
        }
        assert!(flows.take("stale").is_none(), "an expired flow was accepted");

        // And starting another one clears the leftovers.
        {
            let mut held = flows.0.lock().unwrap();
            held.insert(
                "stale".into(),
                Flow {
                    verifier: "v".into(),
                    nonce: Nonce::new("n".into()),
                    started: Instant::now() - FLOW_TTL - Duration::from_secs(1),
                },
            );
        }
        flows.remember("fresh".into(), "v".into(), Nonce::new("n".into()));
        assert_eq!(flows.len(), 1, "the expired flow was kept");
    }

    /// Providers disagree about whether a groups claim is a string or a list.
    #[test]
    fn a_role_claim_is_read_whether_it_is_one_value_or_many() {
        let many = encode(serde_json::json!({ "groups": ["kvad-admins", "everyone"] }));
        assert_eq!(roles_from(&many, "groups"), ["kvad-admins", "everyone"]);

        let one = encode(serde_json::json!({ "groups": "kvad-admins" }));
        assert_eq!(roles_from(&one, "groups"), ["kvad-admins"]);

        // Absent, wrong shape, or not a token at all: no roles, not a panic.
        assert!(roles_from(&many, "roles").is_empty());
        assert!(roles_from(&encode(serde_json::json!({ "groups": 7 })), "groups").is_empty());
        assert!(roles_from("not.a.jwt", "groups").is_empty());
        assert!(roles_from("nodots", "groups").is_empty());
        assert!(roles_from("", "groups").is_empty());
    }

    #[test]
    fn a_new_account_gets_a_name_that_is_allowed_to_be_one() {
        assert_eq!(name_for("ada@example.com", None), "ada");
        assert_eq!(name_for("ada@example.com", Some("ada.lovelace")), "ada.lovelace");
        // Spaces and anything else `users::check_name` refuses.
        assert_eq!(name_for("ada@example.com", Some("Ada Lovelace")), "Ada-Lovelace");
        // A name that cleans down to nothing falls back to the address, which
        // is always allowed because `@` and `.` are.
        assert_eq!(name_for("ada@example.com", Some("!!!")), "ada@example.com");
        for made in [
            name_for("ada@example.com", None),
            name_for("ada@example.com", Some("Ada Lovelace")),
            name_for("ada@example.com", Some("!!!")),
        ] {
            assert!(crate::users::check_name(&made).is_ok(), "`{made}` is not a usable name");
        }
    }
}
