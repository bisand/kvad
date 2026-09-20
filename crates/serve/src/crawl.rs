//! Turning a documentation site into a corpus.
//!
//! Somebody types `https://doc.rust-lang.org/book/`, and a job walks it,
//! reads the prose out of every page, and writes one text file that
//! [`crate::datasets`] then owns like any other. The file on disk is still
//! the truth; this is a different way of getting one there.
//!
//! # Where it stops
//!
//! "The same domain" is the obvious boundary and it is not the right one.
//! The Rust book links into `doc.rust-lang.org/std/` on nearly every page, so
//! a crawl of the book that stayed on the domain would come back with the
//! whole of the standard library's rustdoc — tens of thousands of pages of
//! generated signatures, which is not what anybody meant by "the book".
//!
//! So the default scope is the *directory* the starting URL is in: `/book/`
//! for that address, and `/book/` still for `/book/ch01-02-hello-world.html`.
//! Widening to the whole host is a checkbox, for the sites that are one book
//! at their root.
//!
//! On top of that: `robots.txt` is obeyed, there is a delay between requests,
//! and the crawl stops at a number of pages and a number of bytes. A crawler
//! that a stranger can point at a stranger's server should be boring.
//!
//! # What comes back
//!
//! Two files. `datasets/<name>` is the text, which is what training reads,
//! and `datasets/<name>.crawl.json` is the manifest: the address of every
//! page in the order they were fetched, what was skipped and why, and what
//! the cleaning pass changed. In six months that manifest is the difference
//! between "a corpus" and "this corpus, from here, on that day".

mod clean;
mod html;

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::net::{IpAddr, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use url::Url;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Files that are the rest of the site again.
///
/// mdBook renders the whole book a second time at `print.html`, and rustdoc
/// lists every item at `all.html`. Both are real pages and both would double
/// the corpus, so they are passed over by name. Everything else that repeats
/// itself is caught by the content check in [`run`], which is about pages
/// that are the same page under two addresses — `/book/` and
/// `/book/title-page.html` are the first two a crawl of the book meets.
const REPEATS: &[&str] = &["print.html", "all.html"];

/// How the crawler names itself. Honest, and with the project in it, so that
/// somebody reading their access log can find out what this was.
const AGENT: &str = "kvad-crawler/0.1 (+https://github.com/bisand/kvad)";

/// What a crawl was asked for. Stored as the job's `params`, so a corpus can
/// be rebuilt from the row that made it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Request {
    pub url: String,
    pub name: String,
    /// Follow links anywhere on the host, not just under the starting
    /// directory. Off by default, for the reason at the top of this file.
    #[serde(default)]
    pub same_host: bool,
    #[serde(default = "default_pages")]
    pub max_pages: usize,
    #[serde(default = "default_bytes")]
    pub max_bytes: usize,
    #[serde(default = "default_delay")]
    pub delay_ms: u64,
    /// Drop characters that occur fewer than this many times. 0 or 1 keeps
    /// everything; see [`clean`] for why the default is not 0.
    ///
    /// An absolute count rather than a share of the corpus, because that is
    /// what the number means: a row of an embedding table seen eight times is
    /// untrained whether the corpus around it is 5 kB or 5 MB. Measured on
    /// the Rust book, a threshold of 3 still kept fifty rows for the Japanese,
    /// Hindi, Hebrew and Cyrillic in one chapter's "hello world" examples.
    #[serde(default = "default_rare")]
    pub drop_rare: usize,
}

fn default_pages() -> usize {
    400
}
fn default_bytes() -> usize {
    16 * 1024 * 1024
}
fn default_delay() -> u64 {
    250
}
fn default_rare() -> usize {
    10
}

impl Request {
    /// Clamp what a form can ask for to what this is willing to do.
    ///
    /// Not validation — nothing here is a refusal — but a crawl with no delay
    /// and no page limit is a crawl that gets an address blocked, and the
    /// person filling in the form is not the person it would happen to.
    pub fn sane(mut self) -> Self {
        self.max_pages = self.max_pages.clamp(1, 5_000);
        self.max_bytes = self.max_bytes.clamp(1024, crate::datasets::MAX_BYTES);
        self.delay_ms = self.delay_ms.clamp(50, 10_000);
        self.drop_rare = self.drop_rare.min(1_000);
        self
    }
}

/// One page, as the manifest remembers it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Fetched {
    pub url: String,
    pub title: String,
    pub characters: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Skipped {
    pub url: String,
    pub why: String,
}

/// Everything about a crawl except the text it produced.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Manifest {
    pub start: String,
    /// The prefix a URL had to match to be followed.
    pub scope: String,
    pub same_host: bool,
    pub fetched_at: String,
    pub agent: String,
    pub pages: Vec<Fetched>,
    pub skipped: Vec<Skipped>,
    pub stopped: Option<String>,
    pub characters: usize,
    pub distinct: usize,
    /// What [`clean::normalise`] replaced, commonest first.
    pub mapped: Vec<clean::Mapped>,
    /// What the rare-character threshold removed.
    pub dropped: Vec<clean::Count>,
    /// The alphabet the corpus ended up with. This is the vocabulary a model
    /// trained on it would have, so it is worth being able to read.
    pub alphabet: Vec<clean::Count>,
    pub request: Request,
}

pub struct Crawled {
    pub text: String,
    pub manifest: Manifest,
}

/// Something worth telling a watcher about.
pub enum Note {
    /// A page went into the corpus. `total` is a guess — the queue is still
    /// growing — and never more than the page limit.
    Page { url: String, title: String, done: usize, total: usize },
    Say(String),
}

// ---------------------------------------------------------------------------
// Scope
// ---------------------------------------------------------------------------

/// Where a crawl is allowed to go.
struct Scope {
    origin: (String, String, u16),
    /// The directory of the starting URL, with its trailing slash.
    prefix: String,
    same_host: bool,
}

impl Scope {
    fn of(start: &Url, same_host: bool) -> Res<Scope> {
        let host = start.host_str().ok_or("that address has no host")?.to_ascii_lowercase();
        let port = start.port_or_known_default().ok_or("that address has no port")?;
        let path = start.path();
        // `/book/ch01.html` is in `/book/`, and so is `/book/`.
        let prefix = match path.ends_with('/') {
            true => path.to_string(),
            false => path[..path.rfind('/').map_or(0, |i| i + 1)].to_string(),
        };
        Ok(Scope { origin: (start.scheme().to_string(), host, port), prefix, same_host })
    }

    fn allows(&self, url: &Url) -> bool {
        let same = url.scheme() == self.origin.0
            && url.host_str().map(str::to_ascii_lowercase).as_deref() == Some(&self.origin.1)
            && url.port_or_known_default() == Some(self.origin.2);
        same && (self.same_host || url.path().starts_with(&self.prefix))
    }

    fn describe(&self) -> String {
        let (scheme, host, _) = &self.origin;
        match self.same_host {
            true => format!("{scheme}://{host}/"),
            false => format!("{scheme}://{host}{}", self.prefix),
        }
    }
}

/// A URL as the `seen` set knows it: no fragment, and no empty query.
///
/// `#section` is a place on a page, not another page, and following it would
/// fetch the same page once per heading.
fn key(url: &Url) -> String {
    let mut url = url.clone();
    url.set_fragment(None);
    if url.query() == Some("") {
        url.set_query(None);
    }
    url.to_string()
}

// ---------------------------------------------------------------------------
// Not fetching the machine it is running on
// ---------------------------------------------------------------------------

/// Refuse an address that resolves to this network.
///
/// The form is behind an admin session, so this is not the last line of
/// anything — but "fetch a URL for me" is the shape of request that reads a
/// cloud metadata endpoint or an internal admin page and puts it in a file,
/// and the cost of saying no is one DNS lookup.
///
/// It is checked again on every redirect. It is still a check of the name
/// rather than of the socket: a host that resolves twice, differently, would
/// get past it. Closing that needs a resolver of our own, which is a bigger
/// change than this feature deserves.
fn reachable(url: &Url) -> Res<()> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!("`{}` is not an address this can fetch", url.scheme()).into());
    }
    let host = url.host_str().ok_or("that address has no host")?;
    let port = url.port_or_known_default().unwrap_or(443);
    let addrs: Vec<IpAddr> = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("{host} does not resolve: {e}"))?
        .map(|a| a.ip())
        .collect();
    if addrs.is_empty() {
        return Err(format!("{host} does not resolve").into());
    }
    match addrs.iter().find(|ip| !public(**ip)) {
        Some(ip) => Err(format!("{host} is {ip}, which is on this network").into()),
        None => Ok(()),
    }
}

/// Whether an address is out on the internet rather than in here.
///
/// `IpAddr::is_global` is still unstable, so the ranges are written out. The
/// list is the one that matters for a fetch: loopback, the private blocks,
/// link-local — which is where a cloud's metadata service lives — and the
/// carrier-grade NAT block.
fn public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, ..] = v4.octets();
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_multicast()
                || v4.is_documentation()
                // 100.64.0.0/10, shared address space.
                || (a == 100 && (64..128).contains(&b))
                // 198.18.0.0/15, benchmarking.
                || (a == 198 && (18..20).contains(&b))
                // 0.0.0.0/8 and 240.0.0.0/4.
                || a == 0
                || a >= 240)
        }
        IpAddr::V6(v6) => {
            // A v4 address wearing a v6 hat is still a v4 address.
            if let Some(v4) = v6.to_ipv4_mapped() {
                return public(IpAddr::V4(v4));
            }
            let first = v6.segments()[0];
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                // fc00::/7, unique local.
                || (first & 0xfe00) == 0xfc00
                // fe80::/10, link local.
                || (first & 0xffc0) == 0xfe80)
        }
    }
}

// ---------------------------------------------------------------------------
// Fetching
// ---------------------------------------------------------------------------

fn agent() -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .user_agent(AGENT)
            .timeout_connect(Some(Duration::from_secs(10)))
            .timeout_global(Some(Duration::from_secs(60)))
            // Redirects are followed by hand, a hop at a time, so that each
            // new address is checked against the scope and the network guard
            // before anything is read. A redirect off the site is a redirect
            // out of the crawl.
            .max_redirects(0)
            .max_redirects_will_error(false)
            // A 404 is an answer, not an error: it belongs in the manifest
            // beside the URL that gave it.
            .http_status_as_error(false)
            .build(),
    )
}

struct Body {
    url: Url,
    kind: String,
    text: String,
}

/// Fetch one address, following redirects within the scope.
fn fetch(agent: &ureq::Agent, scope: &Scope, url: &Url, limit: usize) -> Res<Body> {
    let mut url = url.clone();
    for _ in 0..5 {
        reachable(&url)?;
        let mut response = agent.get(url.as_str()).call()?;
        let status = response.status().as_u16();

        if (300..400).contains(&status) {
            let location = response
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or("a redirect with nowhere to go")?;
            let next = url.join(location).map_err(|e| format!("{location}: {e}"))?;
            if !scope.allows(&next) {
                return Err(format!("redirects to {next}, which is outside the crawl").into());
            }
            url = next;
            continue;
        }
        if !(200..300).contains(&status) {
            return Err(format!("the server said {status}").into());
        }

        let kind = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        // The limit is on the body, so a 400 MB file cannot be read into
        // memory by a crawl that was told to collect 16 MB of text.
        let text = response
            .body_mut()
            .with_config()
            .limit(limit as u64)
            .read_to_string()
            .map_err(|e| format!("could not read it as text: {e}"))?;
        return Ok(Body { url, kind, text });
    }
    Err("too many redirects".into())
}

// ---------------------------------------------------------------------------
// robots.txt
// ---------------------------------------------------------------------------

/// The paths a site has asked crawlers to leave alone.
///
/// Prefix rules only: `*` and `$` are treated as the end of the pattern,
/// which makes this stricter than the site meant rather than looser, and that
/// is the right direction to be wrong in.
struct Robots {
    disallow: Vec<String>,
    allow: Vec<String>,
}

impl Robots {
    fn none() -> Robots {
        Robots { disallow: Vec::new(), allow: Vec::new() }
    }

    fn parse(body: &str) -> Robots {
        let mut robots = Robots::none();
        // Only the group for `*` — naming ourselves in a robots.txt would be
        // a thing to do once anybody has heard of this crawler.
        let mut listening = false;
        for line in body.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            let Some((field, value)) = line.split_once(':') else { continue };
            let (field, value) = (field.trim().to_ascii_lowercase(), value.trim());
            match field.as_str() {
                "user-agent" => listening = value == "*",
                "disallow" if listening && !value.is_empty() => {
                    robots.disallow.push(pattern(value))
                }
                "allow" if listening && !value.is_empty() => robots.allow.push(pattern(value)),
                _ => {}
            }
        }
        robots
    }

    /// The longest matching rule wins, and `Allow` wins a tie — the rule
    /// every other crawler follows.
    fn allows(&self, path: &str) -> bool {
        let longest = |rules: &[String]| {
            rules.iter().filter(|r| path.starts_with(r.as_str())).map(|r| r.len()).max()
        };
        match (longest(&self.disallow), longest(&self.allow)) {
            (None, _) => true,
            (Some(deny), Some(allow)) => allow >= deny,
            (Some(_), None) => false,
        }
    }
}

fn pattern(value: &str) -> String {
    value.split(['*', '$']).next().unwrap_or("").to_string()
}

fn robots_for(agent: &ureq::Agent, start: &Url) -> Robots {
    let Ok(url) = start.join("/robots.txt") else { return Robots::none() };
    if reachable(&url).is_err() {
        return Robots::none();
    }
    let Ok(mut response) = agent.get(url.as_str()).call() else { return Robots::none() };
    if response.status().as_u16() != 200 {
        // No robots.txt, or a redirect to a login page: nothing was asked of
        // us, so nothing is denied.
        return Robots::none();
    }
    match response.body_mut().with_config().limit(512 * 1024).read_to_string() {
        Ok(body) => Robots::parse(&body),
        Err(_) => Robots::none(),
    }
}

// ---------------------------------------------------------------------------
// The crawl
// ---------------------------------------------------------------------------

/// Everything about a request that can be known before any fetching.
///
/// The handler's to call, so that an address which is not an address is a
/// refusal of the form rather than a job row that fails a second later. It is
/// not the whole check — [`run`] does all of this again, because a redirect
/// is a new address and the first one being fine says nothing about the
/// tenth.
pub fn check(request: &Request) -> Res<()> {
    let start = Url::parse(request.url.trim())
        .map_err(|e| format!("`{}` is not an address: {e}", request.url.trim()))?;
    Scope::of(&start, request.same_host)?;
    reachable(&start)
}

/// Walk a site and come back with a corpus.
///
/// `say` is called as pages arrive, and `cancel` is read between them: a
/// stopped crawl keeps what it has rather than throwing it away, the same
/// bargain a stopped training run makes with its best checkpoint.
pub fn run(
    request: &Request,
    say: &mut dyn FnMut(Note),
    cancel: &AtomicBool,
) -> Res<Crawled> {
    let start = Url::parse(request.url.trim())
        .map_err(|e| format!("`{}` is not an address: {e}", request.url.trim()))?;
    let scope = Scope::of(&start, request.same_host)?;
    reachable(&start)?;

    let agent = agent();
    say(Note::Say(format!("reading {}", scope.describe())));
    let robots = robots_for(&agent, &start);
    if !robots.allows(start.path()) {
        return Err(format!("robots.txt on {} asks crawlers not to read {}", scope.origin.1, start.path()).into());
    }

    let mut queue: Vec<Url> = vec![start.clone()];
    let mut seen: HashSet<String> = HashSet::from([key(&start)]);
    let mut pages: Vec<Fetched> = Vec::new();
    let mut skipped: Vec<Skipped> = Vec::new();
    let mut stopped: Option<String> = None;
    // The text of every page kept, by hash, so that the same page under two
    // addresses is in the corpus once.
    let mut already: HashMap<u64, String> = HashMap::new();
    let mut body = String::new();
    let mut at = 0;

    while at < queue.len() {
        if cancel.load(Ordering::Relaxed) {
            stopped = Some("stopped".into());
            break;
        }
        if pages.len() >= request.max_pages {
            stopped = Some(format!("the limit of {} pages", request.max_pages));
            break;
        }
        if body.len() >= request.max_bytes {
            stopped = Some(format!(
                "the limit of {}",
                kvad::hub::human_bytes(request.max_bytes as u64)
            ));
            break;
        }

        let url = queue[at].clone();
        at += 1;

        // Politeness. Before the request rather than after, so that a crawl
        // that stops early has not just finished hammering something.
        if at > 1 {
            std::thread::sleep(Duration::from_millis(request.delay_ms));
        }

        let room = request.max_bytes.saturating_sub(body.len()).max(4096);
        let fetched = match fetch(&agent, &scope, &url, room) {
            Ok(fetched) => fetched,
            Err(e) => {
                skipped.push(Skipped { url: url.to_string(), why: e.to_string() });
                continue;
            }
        };

        let markup = matches!(fetched.kind.as_str(), "text/html" | "application/xhtml+xml");
        let plain = matches!(fetched.kind.as_str(), "text/plain" | "text/markdown" | "");
        if !markup && !plain {
            skipped.push(Skipped { url: url.to_string(), why: format!("it is {}", fetched.kind) });
            continue;
        }

        let page = match markup {
            true => html::extract(&fetched.text),
            false => html::Page {
                title: String::new(),
                text: fetched.text.clone(),
                links: Vec::new(),
                base: None,
            },
        };

        // Follow first, keep second: a page whose prose is empty — an index,
        // a redirect page — is still worth having read for its links.
        if markup {
            let base = page
                .base
                .as_deref()
                .and_then(|b| fetched.url.join(b).ok())
                .unwrap_or_else(|| fetched.url.clone());
            for href in &page.links {
                let Ok(next) = base.join(href) else { continue };
                if !scope.allows(&next) || !seen.insert(key(&next)) {
                    continue;
                }
                if REPEATS.contains(&next.path().rsplit('/').next().unwrap_or("")) {
                    skipped.push(Skipped {
                        url: next.to_string(),
                        why: "it is the rest of the site again".into(),
                    });
                    continue;
                }
                if !robots.allows(next.path()) {
                    skipped.push(Skipped {
                        url: next.to_string(),
                        why: "robots.txt asks crawlers not to read it".into(),
                    });
                    continue;
                }
                queue.push(next);
            }
        }

        let text = page.text.trim();
        if text.is_empty() {
            skipped.push(Skipped { url: url.to_string(), why: "no text on it".into() });
            continue;
        }

        let mut hash = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hash);
        if let Some(first) = already.insert(hash.finish(), fetched.url.to_string()) {
            skipped.push(Skipped {
                url: url.to_string(),
                why: format!("the same text as {first}"),
            });
            continue;
        }

        let title = match page.title.is_empty() {
            true => fetched.url.path().to_string(),
            false => page.title.clone(),
        };
        // One heading per page, so that the corpus reads as a document rather
        // than as a pile — but only where the page does not already start
        // with one of its own. mdBook titles its pages `Ownership - The Rust
        // Programming Language` and then opens with `# Ownership`, and both
        // of those in a row is one too many. The address is in neither: it is
        // in the manifest, where it is provenance rather than something to
        // learn to write.
        match text.starts_with('#') {
            true => body.push_str(&format!("\n\n{text}\n")),
            false => body.push_str(&format!("\n\n# {title}\n\n{text}\n")),
        }
        pages.push(Fetched { url: fetched.url.to_string(), title: title.clone(), characters: text.chars().count() });

        let total = (pages.len() + queue.len() - at).min(request.max_pages);
        say(Note::Page { url: fetched.url.to_string(), title, done: pages.len(), total });
    }

    if pages.is_empty() {
        let why = skipped
            .first()
            .map(|s| format!(" — {}", s.why))
            .unwrap_or_else(|| " and nothing was found to read".into());
        return Err(format!("nothing came back from {}{why}", scope.describe()).into());
    }

    say(Note::Say(format!("{} pages read; cleaning up the text", pages.len())));
    let (text, mapped) = clean::normalise(&body);
    let (text, dropped) = clean::drop_rare(&text, request.drop_rare);
    let alphabet = clean::histogram(&text);

    let manifest = Manifest {
        start: start.to_string(),
        scope: scope.describe(),
        same_host: request.same_host,
        fetched_at: now(),
        agent: AGENT.to_string(),
        pages,
        skipped,
        stopped,
        characters: text.chars().count(),
        distinct: alphabet.len(),
        mapped,
        dropped,
        alphabet,
        request: request.clone(),
    };
    Ok(Crawled { text, manifest })
}

/// Today, as `2026-09-20T11:03:12Z`.
///
/// Hand-rolled from the epoch rather than a date crate: this is the only
/// place in the server that needs a formatted timestamp that SQLite is not
/// already writing for it.
fn now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rest) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (hour, minute, second) = (rest / 3600, (rest % 3600) / 60, rest % 60);

    // Days since the epoch to a civil date, shifting the year to start in
    // March so that the leap day is the last day of it and the month lengths
    // fall into a 153-day pattern. Howard Hinnant's `civil_from_days`.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(start: &str, same_host: bool) -> Scope {
        Scope::of(&Url::parse(start).unwrap(), same_host).unwrap()
    }

    /// The reason this is not "the same domain": the book links into the
    /// standard library's rustdoc on nearly every page, and following that
    /// would fetch tens of thousands of pages nobody asked for.
    #[test]
    fn the_default_scope_is_the_directory_not_the_domain() {
        let book = scope("https://doc.rust-lang.org/book/ch01-02-hello-world.html", false);
        assert_eq!(book.prefix, "/book/");
        let allows = |u: &str| book.allows(&Url::parse(u).unwrap());
        assert!(allows("https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html"));
        assert!(allows("https://doc.rust-lang.org/book/"));
        assert!(!allows("https://doc.rust-lang.org/std/vec/struct.Vec.html"));
        assert!(!allows("https://crates.io/"));
        // A different scheme or port is a different place.
        assert!(!allows("http://doc.rust-lang.org/book/x.html"));
        assert!(!allows("https://doc.rust-lang.org:8443/book/x.html"));

        // And the checkbox that widens it.
        let host = scope("https://doc.rust-lang.org/book/", true);
        assert!(host.allows(&Url::parse("https://doc.rust-lang.org/std/index.html").unwrap()));
        assert!(!host.allows(&Url::parse("https://www.rust-lang.org/").unwrap()));
    }

    #[test]
    fn a_fragment_is_a_place_on_a_page_not_another_page() {
        let one = key(&Url::parse("https://x.test/a.html#install").unwrap());
        assert_eq!(one, "https://x.test/a.html");
        assert_eq!(one, key(&Url::parse("https://x.test/a.html?").unwrap()));
        assert_ne!(one, key(&Url::parse("https://x.test/a.html?v=2").unwrap()));
    }

    #[test]
    fn robots_rules_are_longest_match_and_allow_wins_a_tie() {
        let robots = Robots::parse(
            "User-agent: bingbot\nDisallow: /\n\n\
             User-agent: *\n# a comment\nDisallow: /private/\nAllow: /private/ok/\nDisallow: /tmp*\n",
        );
        assert!(robots.allows("/book/ch01.html"));
        assert!(!robots.allows("/private/secret.html"));
        assert!(robots.allows("/private/ok/fine.html"));
        // `*` is cut to its prefix, which denies more than the site asked.
        assert!(!robots.allows("/tmpfile"));
        // The group for another crawler is not ours to obey.
        assert!(Robots::parse("User-agent: bingbot\nDisallow: /\n").allows("/anything"));
        assert!(Robots::none().allows("/anything"));
    }

    /// The addresses a "fetch this URL for me" feature must not fetch.
    #[test]
    fn this_network_is_not_the_internet() {
        for private in ["127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1", "169.254.169.254", "100.64.0.1", "0.0.0.0", "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1"] {
            assert!(!public(private.parse().unwrap()), "{private} was allowed");
        }
        for out_there in ["1.1.1.1", "140.82.121.4", "2606:4700::1111"] {
            assert!(public(out_there.parse().unwrap()), "{out_there} was refused");
        }
        assert!(reachable(&Url::parse("http://localhost/").unwrap()).is_err());
        assert!(reachable(&Url::parse("file:///etc/passwd").unwrap()).is_err());
    }

    /// What the form is told before a job exists.
    #[test]
    fn an_address_that_cannot_be_fetched_is_refused_up_front() {
        let asking = |url: &str| {
            check(&Request {
                url: url.into(),
                name: "x".into(),
                same_host: false,
                max_pages: 1,
                max_bytes: 4096,
                delay_ms: 50,
                drop_rare: 0,
            })
        };
        assert!(asking("not a url").is_err());
        assert!(asking("file:///etc/passwd").is_err());
        assert!(asking("http://127.0.0.1:8080/").is_err());
        // A literal address, so that this test does not need a resolver: a
        // name here would make the suite need the internet to pass.
        assert!(asking("https://1.1.1.1/docs/").is_ok());
    }

    #[test]
    fn a_form_cannot_ask_for_an_unlimited_crawl() {
        let wild = Request {
            url: "https://x.test/".into(),
            name: "x".into(),
            same_host: true,
            max_pages: 10_000_000,
            max_bytes: usize::MAX,
            delay_ms: 0,
            drop_rare: 9_999_999,
        }
        .sane();
        assert_eq!(wild.max_pages, 5_000);
        assert_eq!(wild.max_bytes, crate::datasets::MAX_BYTES);
        assert_eq!(wild.delay_ms, 50);
        assert_eq!(wild.drop_rare, 1_000);
    }

    /// The real thing, because every other test here is about a string this
    /// file also wrote. Ignored by default: it fetches three pages from
    /// doc.rust-lang.org, and a test suite should not need the internet.
    ///
    ///     cargo test -p kvad-serve -- --ignored the_rust_book
    #[test]
    #[ignore = "fetches from the network"]
    fn the_rust_book_comes_back_as_prose() {
        let request = Request {
            url: "https://doc.rust-lang.org/book/".into(),
            name: "rust-book-test".into(),
            same_host: false,
            max_pages: 6,
            max_bytes: 4 * 1024 * 1024,
            delay_ms: 500,
            drop_rare: 10,
        };
        let crawled =
            run(&request, &mut |_| {}, &AtomicBool::new(false)).expect("the crawl failed");
        let text = &crawled.text;

        println!("{}", &text[..text.len().min(1500)]);
        println!(
            "\n{} pages, {} characters, {} distinct, {} skipped",
            crawled.manifest.pages.len(),
            crawled.manifest.characters,
            crawled.manifest.distinct,
            crawled.manifest.skipped.len()
        );
        for page in &crawled.manifest.pages {
            println!("{:>7} {}", page.characters, page.url);
        }
        for skipped in &crawled.manifest.skipped {
            println!("skipped {} — {}", skipped.url, skipped.why);
        }
        println!("alphabet: {:?}", crawled.manifest.alphabet.iter().map(|c| c.character.as_str()).collect::<Vec<_>>());

        assert_eq!(crawled.manifest.pages.len(), 6);
        // Prose, not a table of contents: the sidebar lists every chapter on
        // every page, so a crawler that kept it would repeat itself.
        let toc = text.matches("Programming a Guessing Game").count();
        assert!(toc <= 6, "the table of contents is in the text {toc} times");
        // Six pages of a book, not six copies of a book: `print.html` is the
        // whole of it again, and `/book/` and `/book/title-page.html` are one
        // page under two addresses.
        assert!(crawled.manifest.characters < 200_000, "{} characters", crawled.manifest.characters);
        // A character tokeniser's vocabulary. English and Rust with the rare
        // tail cut off is around a hundred characters; several hundred would
        // mean the cleaning pass is not working.
        assert!(crawled.manifest.distinct < 150, "{} distinct characters", crawled.manifest.distinct);
        // It found its way off the first page, which needed the sidebar's
        // links even though the sidebar is not in the text.
        assert!(crawled.manifest.pages.iter().any(|p| p.url.contains("ch01")));
    }

    #[test]
    fn the_epoch_is_a_date() {
        assert!(now().ends_with('Z') && now().len() == 20);
        assert!(now() > "2026-01-01T00:00:00Z".to_string());
    }
}
