//! The command line's side of talking to a server: `kvad service`, and every
//! command that sends its work to a running `kvad-serve`.
//!
//! These modules belong to the `kvad` binary, not to the library beside it.
//! The library's half — which server, how to reach it, how to read what it
//! streams, and how to drive the service manager — is `kvad::client` and
//! `kvad::daemon`, where a terminal UI could use it too. What is here is the
//! part that prints.
//!
//! * [`service`] — `kvad service`: the server as something that starts at
//!   login.
//! * [`models`] — the commands that also run in this process, answered by a
//!   server instead: `ls`, `pull`, `run`, `chat`, `train` and the rest. Each
//!   keeps the shape of its local output, so the answer looks the same
//!   whichever process gave it.
//! * [`api`] — the commands only a server can answer: conversations, jobs,
//!   datasets, evals, benchmarks, metrics, accounts, and `kvad api` for any
//!   route at all.
//! * [`out`] — tables, questions and progress lines.

pub mod api;
pub mod models;
pub mod out;
pub mod service;

use crate::Args;
use kvad::client::{self, Remote, Target};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Resolve where a command runs, and say so before anything else is printed.
///
/// Not said for a command only a server can answer when there is none: "in
/// this process" followed by "there is no server" is two lines that argue.
pub fn target(args: &Args) -> Res<Target> {
    let target = client::resolve(&args.choice)?;
    if matches!(target, Target::Remote(_)) || !remote_only(&args.command) {
        eprintln!("{}", target.line());
    }
    Ok(target)
}

/// The commands that have no answer in this process.
pub fn remote_only(command: &str) -> bool {
    matches!(
        command,
        "ps" | "load"
            | "unload"
            | "cancel"
            | "tokenize"
            | "conversations"
            | "images"
            | "videos"
            | "jobs"
            | "datasets"
            | "evals"
            | "bench"
            | "metrics"
            | "auth"
            | "users"
            | "sessions"
            | "keys"
            | "api"
    )
}

/// Whether a command exists at all, asked before anything is sent anywhere:
/// a typo should be told so, not probe for a server first.
pub fn known(command: &str) -> bool {
    remote_only(command) || models::LOCAL_TOO.contains(&command)
}

/// A command only a server can answer, and no server to answer it.
pub fn no_server(command: &str) -> ! {
    let hint = match command {
        "ps" | "load" | "unload" => {
            "\nIn this process a model is loaded by the command that uses it, and let go\n\
             when that command ends, so there is nothing to keep in memory between them."
        }
        _ => "",
    };
    eprintln!(
        "`kvad {command}` asks a running server, and there is none to ask.{hint}\n\n\
         Start this machine's service:   kvad service start\n\
         or install it, to run at login: kvad service install\n\
         or name one somewhere else:     kvad {command} --remote http://HOST:{port}",
        port = client::DEFAULT_PORT,
    );
    std::process::exit(1);
}

/// Send a command to a server.
pub fn remote(remote: &Remote, args: &Args) -> Res<()> {
    match args.command.as_str() {
        "ls" => models::ls(remote, args),
        "ps" => models::ps(remote, args),
        "search" => models::search(remote, args),
        "info" => models::info(remote, args),
        "pull" => models::pull(remote, args),
        "use" => models::use_model(remote, args),
        "rm" => models::remove(remote, args),
        "cache" => models::cache(remote, args),
        "load" => models::load(remote, args).map(|_| ()),
        "unload" => models::unload(remote, args),
        "cancel" => models::cancel(remote, args),
        "tokenize" => models::tokenize(remote, args),
        "run" => models::run(remote, args),
        "chat" => models::chat(remote, args),
        "train" => models::train(remote, args),
        "conversations" => api::conversations(remote, args),
        "images" => api::images(remote, args),
        "videos" => api::videos(remote, args),
        "jobs" => api::jobs(remote, args),
        "datasets" => api::datasets(remote, args),
        "evals" => api::evals(remote, args),
        "bench" => api::bench(remote, args),
        "metrics" => api::metrics(remote, args),
        "auth" => api::auth(remote, args),
        "users" => api::users(remote, args),
        "sessions" => api::sessions(remote, args),
        "keys" => api::keys(remote, args),
        "api" => api::raw(remote, args),
        other => {
            eprintln!("unknown command `{other}`");
            crate::usage();
        }
    }
}

/// `--help` after a command: that command's own usage, where it has one.
pub fn help(command: &str) -> ! {
    let text = match command {
        "service" => service::USAGE,
        "conversations" => api::CONVERSATIONS,
        "images" => api::IMAGES,
        "videos" => api::VIDEOS,
        "jobs" => api::JOBS,
        "datasets" => api::DATASETS,
        "evals" => api::EVALS,
        "bench" => api::BENCH,
        "metrics" => api::METRICS,
        "auth" => api::AUTH,
        "users" => api::USERS,
        "sessions" => api::SESSIONS,
        "keys" => api::KEYS,
        "api" => api::API,
        _ => crate::usage(),
    };
    println!("{text}");
    std::process::exit(0);
}

/// The subcommand, and the words after it.
///
/// `default` is what a bare `kvad jobs` means, which for every group is the
/// listing.
pub fn split<'a>(args: &'a Args, default: &'a str) -> (&'a str, &'a [String]) {
    match args.words.split_first() {
        Some((first, rest)) => (first.as_str(), rest),
        None => (default, &[]),
    }
}

/// An unknown subcommand: say so, show the group's usage, and stop.
pub fn unknown(group: &str, sub: &str, usage: &str) -> ! {
    eprintln!("`kvad {group} {sub}` is not a command.\n\n{usage}");
    std::process::exit(2);
}

/// The one word a subcommand needs, or its usage.
pub fn needs<'a>(words: &'a [String], what: &str, usage: &str) -> &'a str {
    match words.first() {
        Some(w) => w.as_str(),
        None => {
            eprintln!("this needs {what}.\n\n{usage}");
            std::process::exit(2);
        }
    }
}

/// A word that has to be a number: an id.
pub fn id(word: &str) -> Res<i64> {
    word.parse().map_err(|_| format!("`{word}` is not an id; ids are numbers, as the listing shows them").into())
}

/// A value for a query string.
pub fn enc(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The source of every module that talks to a server.
    const SOURCES: &[(&str, &str)] = &[
        ("models.rs", include_str!("cli/models.rs")),
        ("api.rs", include_str!("cli/api.rs")),
        ("service.rs", include_str!("cli/service.rs")),
    ];

    const METHODS: &[&str] = &["get", "post", "patch", "delete"];

    /// The client's calls that take the method as their first argument.
    const TAKES_A_METHOD: &[&str] = &["call", "call_for_cookie", "stream"];

    /// The client's calls that are a GET under another name.
    const GETS: &[&str] = &["download"];

    /// Every `(method, path)` the CLI's source requests.
    ///
    /// Read out of the source the way `kvad-serve` reads its router: every
    /// string literal that starts `/api/` or `/v1/`, with the method taken
    /// from the call it is an argument to — `remote.get("/api/…")`, or
    /// `remote.stream("post", "/api/…", …)` where the method is a word of its
    /// own. A path built with `format!` keeps its placeholders, which are
    /// named after what they hold — `{id}`, `{hash}` — so they read the same
    /// as the server's.
    ///
    /// `kvad api` sends whatever it is given and is not scanned: its paths
    /// are the user's, not the source's.
    fn requested() -> BTreeSet<(String, String)> {
        let mut found = BTreeSet::new();
        for (file, source) in SOURCES {
            for (at, _) in source.match_indices('"') {
                let rest = &source[at + 1..];
                if !(rest.starts_with("/api/") || rest.starts_with("/v1/")) {
                    continue;
                }
                let end = rest.find(['"', '?']).expect("an unterminated string");
                let path = rest[..end].to_string();
                let method = method_before(&source[..at])
                    .unwrap_or_else(|| panic!("{file}: no method found for {path}"));
                found.insert((method, path));
            }
        }
        found
    }

    /// The method of the call a path literal is an argument to.
    ///
    /// Walks back over `&format!(`, a `"post", ` argument and whitespace to
    /// the `.name(` that opens the call. For the calls in [`TAKES_A_METHOD`],
    /// the method is the string argument before the path.
    fn method_before(before: &str) -> Option<String> {
        let trimmed = before.trim_end().trim_end_matches("&format!(").trim_end_matches("format!(");
        // `remote.stream("post", "/api/...")`: the method is its own literal.
        if let Some(prefix) = trimmed.trim_end().strip_suffix(',') {
            let prefix = prefix.trim_end();
            let lit = prefix.strip_suffix('"')?;
            let open = lit.rfind('"')?;
            let method = &lit[open + 1..];
            let call = lit[..open].trim_end().strip_suffix('(')?;
            let name = call.rsplit(['.', ' ', '\n', '(']).next()?;
            return (TAKES_A_METHOD.contains(&name) && METHODS.contains(&method))
                .then(|| method.to_string());
        }
        let call = trimmed.trim_end().strip_suffix('(')?;
        let name = call.rsplit(['.', ' ', '\n', '(']).next()?;
        match GETS.contains(&name) {
            true => Some("get".into()),
            false => METHODS.contains(&name).then(|| name.to_string()),
        }
    }

    /// What the table promises and what the source does are the same set.
    ///
    /// With `kvad-serve`'s test that every route is in the table, this is
    /// what makes "the CLI reaches everything the API has" something a test
    /// says rather than something somebody remembers.
    #[test]
    fn every_route_in_the_table_is_requested_and_nothing_else_is() {
        let requested = requested();
        let promised: BTreeSet<(String, String)> = client::COMMANDS
            .iter()
            .map(|(method, path, _)| (method.to_string(), path.to_string()))
            .collect();

        let unused: Vec<_> = promised.difference(&requested).collect();
        assert!(unused.is_empty(), "in kvad::client::COMMANDS and never requested: {unused:#?}");
        let unlisted: Vec<_> = requested.difference(&promised).collect();
        assert!(unlisted.is_empty(), "requested and not in kvad::client::COMMANDS: {unlisted:#?}");
        assert!(requested.len() > 50, "only found {} requests; the scanner is broken", requested.len());
    }

    #[test]
    fn the_scanner_finds_the_method_of_each_shape_of_call() {
        assert_eq!(method_before("remote.get(").as_deref(), Some("get"));
        assert_eq!(method_before("r.delete(&format!(").as_deref(), Some("delete"));
        assert_eq!(method_before("remote.stream(\"post\", ").as_deref(), Some("post"));
        assert_eq!(method_before("remote.download(&format!(").as_deref(), Some("get"));
        assert_eq!(method_before("remote\n        .call(\"patch\", &format!(").as_deref(), Some("patch"));
        assert_eq!(method_before("r.call_for_cookie(\n    \"post\",\n    ").as_deref(), Some("post"));
        assert_eq!(method_before("let path = "), None);
    }

    /// Every command the table names is one the CLI dispatches.
    #[test]
    fn every_command_the_table_names_exists() {
        let known: BTreeSet<&str> = [
            "ls", "ps", "search", "info", "pull", "use", "rm", "cache", "load", "unload", "cancel",
            "tokenize", "run", "chat", "train", "service", "conversations", "images", "videos", "jobs",
            "datasets", "evals", "bench", "metrics", "auth", "users", "sessions", "keys", "api",
        ]
        .into();
        for (method, path, commands) in client::COMMANDS {
            for command in commands.split(", ") {
                let word = command.split_whitespace().nth(1).unwrap_or("");
                assert!(known.contains(word), "{method} {path} names `{command}`, which is not a command");
                assert!(
                    word == "service" || word == "api" || remote_only(word) || crate::cli::models::LOCAL_TOO.contains(&word),
                    "`{word}` is dispatched to neither a server nor this process"
                );
            }
        }
    }

    #[test]
    fn words_go_anywhere_and_flags_are_still_flags() {
        let a = crate::parse_from(["keys", "rm", "--json", "4"].map(String::from).to_vec());
        assert_eq!(a.words, ["rm", "4"]);
        assert!(a.json);

        let a = crate::parse_from(["service", "logs", "-f", "-n", "20"].map(String::from).to_vec());
        assert_eq!(a.words, ["logs"]);
        assert!(a.follow);
        assert_eq!(a.limit, Some(20));

        let a = crate::parse_from(["search", "qwen", "coder"].map(String::from).to_vec());
        assert_eq!(a.target.as_deref(), Some("qwen coder"));

        let a = crate::parse_from(["chat", "--remote", "box:5823"].map(String::from).to_vec());
        assert_eq!(a.choice, client::Choice::Remote("box:5823".into()));
        let a = crate::parse_from(["run", "--local", "--quant", "q8"].map(String::from).to_vec());
        assert_eq!(a.choice, client::Choice::Local);
        assert!(a.quant_given);
    }
}
