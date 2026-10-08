//! Tab completion: `kvad completions SHELL`, and the `kvad __complete` the
//! shells' scripts call.
//!
//! # Why the scripts are so short
//!
//! The usual way to complete a command is to generate, per shell, a script
//! that knows every subcommand and flag. That is three programs in three
//! languages saying what [`super::spec`] already says, and none of them can
//! know what is worth completing most: which models are on this disk, which
//! are in the server's memory, what job 16 was.
//!
//! So the scripts here know nothing. Each hands the words typed so far to
//! `kvad __complete`, and prints what comes back. The answer is worked out
//! in this file, from the same table the help is printed from, and from the
//! machine: a model's name is read from the cache directory or asked of the
//! server, at the moment of the Tab.
//!
//! # What `__complete` says
//!
//! One candidate to a line, a tab, and what it is — zsh and fish show the
//! second half beside the first, and bash drops it. Then one last line that
//! is not a candidate: `:files`, `:dirs` or `:none`, for whether the shell
//! should offer its own file names as well. Only the shell knows how to do
//! that properly, with `~` and quoting and the rest.
//!
//! # Being quick, and being quiet
//!
//! This runs between a keypress and the next. A server is asked through
//! [`Remote::hurried`], which gives up after two seconds, and is asked at
//! all only for a word that needs it: `kvad ru<Tab>` touches neither disk
//! nor network. Nothing is ever written to stderr, and a failure of any kind
//! is an empty answer. A Tab that prints an error into the middle of a
//! command line is worse than one that does nothing.

use super::out;
use super::spec::{self, Command, Flag, Kind, Listing, Sub, Want};
use kvad::client::{self, Choice, Remote, Target};
use serde_json::Value;
use std::cell::OnceCell;

/// Whether the shell should add file names of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Files {
    None,
    Files,
    Dirs,
}

pub struct Answer {
    /// A candidate, and what it is.
    pub items: Vec<(String, String)>,
    pub files: Files,
}

/// A model, as much of it as choosing one needs.
#[derive(Clone, Debug, Default)]
pub struct Model {
    pub id: String,
    /// `chat`, `image` or `video`, when a server said; nothing otherwise.
    pub kind: String,
    pub arch: Option<String>,
    pub bytes: u64,
    pub lora: bool,
    pub complete: bool,
    pub trained: bool,
    pub resident: bool,
    pub active: bool,
}

/// Where the things only the machine knows come from. A trait so that the
/// tests can answer for the machine.
pub trait Source {
    fn models(&self) -> Vec<Model>;
    /// The models in a server's memory, by id.
    fn residents(&self) -> Vec<(String, String)>;
    fn backends(&self) -> Vec<(String, String)>;
    fn rows(&self, listing: Listing) -> Vec<(String, String)>;
}

// ---------------------------------------------------------------------------
// What to offer
// ---------------------------------------------------------------------------

/// The candidates for the last of `words`, which are everything typed after
/// `kvad`, the word being completed included — empty, when the cursor
/// follows a space.
pub fn answer(words: &[String], source: &dyn Source) -> Answer {
    let nothing = Answer { items: Vec::new(), files: Files::None };
    let (cur, done) = match words.split_last() {
        Some((cur, done)) => (cur.as_str(), done),
        None => ("", &[][..]),
    };

    // The command itself. Flags before any command are `run`'s.
    let (command, rest) = match done.first() {
        None if cur.starts_with('-') => ("run", done),
        None => {
            let commands = spec::COMMANDS.iter().map(|c| (c.name.to_string(), c.about.to_string()));
            return Answer { items: matching(commands, cur), files: Files::None };
        }
        Some(first) if first.starts_with('-') => ("run", done),
        Some(first) => (first.as_str(), &done[1..]),
    };
    let Some(c) = spec::command(command) else {
        return nothing;
    };

    // Walk what is already there: which words are words, and whether the
    // last of them is a flag still waiting for its value.
    let takes_words = !c.words.is_empty() || !c.subs.is_empty();
    let mut positional: Vec<&str> = Vec::new();
    let mut used: Vec<&str> = Vec::new();
    let mut waiting: Option<&str> = None;
    for word in rest {
        if waiting.take().is_some() {
            continue;
        }
        if word.len() > 1 && word.starts_with('-') {
            used.push(word);
            if spec::takes_value(word) == Some(true) {
                waiting = Some(word);
            }
        } else if takes_words {
            positional.push(word);
        }
    }
    let sub = positional.first().and_then(|w| c.sub(w));

    // A flag's value.
    if let Some(flag) = waiting {
        return match find_flag(c, sub, flag) {
            Some(f) => of_kind(f.kind, cur, &positional, source),
            None => nothing,
        };
    }

    let flags = || {
        let offered = applicable(c, sub).into_iter().filter(|f| repeats(f) || !used.iter().any(|u| *u == f.name || *u == f.short));
        let mut items: Vec<(String, String)> = offered.map(|f| (f.name.to_string(), f.about.to_string())).collect();
        items.push(("--help".into(), format!("what `kvad {}` takes, with examples", c.name)));
        Answer { items: matching(items.into_iter(), cur), files: Files::None }
    };
    if cur.starts_with('-') {
        return flags();
    }

    // A word. The first of a command that has subcommands is one of them;
    // after it, the words are that subcommand's.
    let mut found = Answer { items: Vec::new(), files: Files::None };
    let (kinds, at, repeats) = match (sub, positional.len()) {
        (Some(s), n) => (s.words, n - 1, s.args.contains("...")),
        (None, n) => (c.words, n, c.args.contains("...")),
    };
    if positional.is_empty() && !c.subs.is_empty() {
        let subs = c.subs.iter().map(|s| (s.name.to_string(), s.about.to_string()));
        found.items = matching(subs, cur);
        // `train` and `tune` are mostly options, with one word they also
        // take. Offered alone, that word would be typed in by the Tab.
        if cur.is_empty() && !c.flags.is_empty() {
            found.items.extend(flags().items);
        }
    }
    let kind = kinds.get(at).or(if repeats { kinds.last() } else { None });
    match kind {
        Some(kind) => {
            let more = of_kind(*kind, cur, &positional, source);
            found.items.extend(more.items);
            found.files = more.files;
            found
        }
        // Nothing more to say in words: what is left is options.
        None if found.items.is_empty() && cur.is_empty() => flags(),
        None => found,
    }
}

/// The flags a command takes where the cursor is: its own, its
/// subcommand's — or every subcommand's, before one is chosen — and where
/// it runs.
fn applicable(c: &'static Command, sub: Option<&'static Sub>) -> Vec<&'static Flag> {
    let mut flags: Vec<&'static Flag> = match sub {
        Some(s) => c.flags.iter().chain(s.flags).flat_map(|g| g.iter()).collect(),
        None => c.all_flags(),
    };
    if c.routed {
        flags.extend(spec::WHERE);
    }
    let mut once: Vec<&'static Flag> = Vec::new();
    for flag in flags {
        if !once.iter().any(|f| f.name == flag.name) {
            once.push(flag);
        }
    }
    once
}

fn find_flag(c: &'static Command, sub: Option<&'static Sub>, name: &str) -> Option<&'static Flag> {
    let is = |f: &&'static Flag| f.name == name || (!f.short.is_empty() && f.short == name);
    applicable(c, sub).into_iter().find(is).or_else(|| c.all_flags().into_iter().find(is))
}

/// Flags worth offering a second time.
fn repeats(f: &Flag) -> bool {
    matches!(f.name, "--lora") || (f.name == "--sample" && f.value == "TEXT")
}

/// The candidates of one kind of word.
fn of_kind(kind: Kind, cur: &str, positional: &[&str], source: &dyn Source) -> Answer {
    let (items, files): (Vec<(String, String)>, Files) = match kind {
        Kind::Text => (Vec::new(), Files::None),
        Kind::File => (Vec::new(), Files::Files),
        Kind::Dir => (Vec::new(), Files::Dirs),
        // A model can be a directory too, so the shell's own are offered
        // once what is typed looks like a path.
        Kind::Model(want) => (
            source.models().iter().filter(|m| wanted(m, want)).map(described).collect(),
            if kvad::weights::looks_like_path(cur) { Files::Dirs } else { Files::None },
        ),
        Kind::Lora => (source.models().iter().filter(|m| m.lora).map(described).collect(), Files::None),
        Kind::Resident => (source.residents(), Files::None),
        Kind::Backend => (source.backends(), Files::None),
        // `MODEL@` is asking for a backend; anything before the `@` is
        // asking for a model.
        Kind::Variant => match cur.rsplit_once('@') {
            Some((model, _)) => {
                (source.backends().into_iter().map(|(id, label)| (format!("{model}@{id}"), label)).collect(), Files::None)
            }
            None => (source.models().iter().filter(|m| wanted(m, Want::Text)).map(described).collect(), Files::None),
        },
        Kind::Id(listing) => (source.rows(listing), Files::None),
        Kind::Choice(choices) => (choices.iter().map(|(v, about)| (v.to_string(), about.to_string())).collect(), Files::None),
        Kind::TrainSize => (kvad::train::SIZES.iter().map(|s| (s.name.to_string(), s.shape())).collect(), Files::None),
        Kind::Command => (spec::COMMANDS.iter().map(|c| (c.name.to_string(), c.about.to_string())).collect(), Files::None),
        Kind::Route => (routes(positional), Files::None),
    };
    Answer { items: matching(items.into_iter(), cur), files }
}

/// `kvad api`'s words: a method, then a path that method has; or a path
/// straight away, which is a GET.
fn routes(positional: &[&str]) -> Vec<(String, String)> {
    const METHODS: &[&str] = &["get", "post", "put", "patch", "delete"];
    let method = match positional {
        [] => None,
        [m] if METHODS.contains(&m.to_ascii_lowercase().as_str()) => Some(m.to_ascii_lowercase()),
        // A path has been given: what follows is a body.
        _ => return Vec::new(),
    };
    let mut found: Vec<(String, String)> = Vec::new();
    if method.is_none() {
        found.extend(METHODS.iter().skip(1).map(|m| (m.to_string(), "then a path".to_string())));
    }
    let wanted = method.as_deref().unwrap_or("get");
    let commands = client::COMMANDS.iter().map(|(m, path, what)| (*m, *path, what.to_string()));
    let others = client::NOT_COMMANDS.iter().map(|(m, path, _)| (*m, *path, String::new()));
    for (m, path, what) in commands.chain(others) {
        if m == wanted && !found.iter().any(|(p, _)| p == path) {
            found.push((path.to_string(), what));
        }
    }
    found
}

fn wanted(m: &Model, want: Want) -> bool {
    match want {
        Want::Any => true,
        // The server calls anything that is not a picture or a clip `chat`,
        // a text encoder included. An architecture is what tells them apart.
        Want::Text => !m.lora && m.arch.is_some() && m.complete && matches!(m.kind.as_str(), "" | "chat"),
        Want::Image => !m.lora && matches!(m.kind.as_str(), "image") || m.kind.is_empty() && m.arch.is_none() && !m.lora,
        Want::Video => !m.lora && matches!(m.kind.as_str(), "video") || m.kind.is_empty() && m.arch.is_none() && !m.lora,
        Want::Trained => m.trained && !m.lora,
    }
}

/// A model and the line beside it: what it is, how big, and whether it is
/// already in memory.
fn described(m: &Model) -> (String, String) {
    let mut parts: Vec<String> = Vec::new();
    match (m.lora, &m.arch, m.kind.as_str()) {
        (true, ..) => parts.push("LoRA".into()),
        (_, Some(arch), _) => parts.push(arch.clone()),
        (_, None, "") => {}
        (_, None, kind) => parts.push(kind.to_string()),
    }
    if m.trained && !m.lora {
        parts.push("trained here".into());
    }
    if m.bytes > 0 {
        parts.push(kvad::hub::human_bytes(m.bytes));
    }
    if !m.complete {
        parts.push("config only".into());
    }
    if m.resident {
        parts.push("in memory".into());
    }
    if m.active {
        parts.push("the default".into());
    }
    (m.id.clone(), parts.join(" · "))
}

/// The candidates that start with what is typed, whatever its case, with
/// their descriptions made fit to print on one line.
fn matching(items: impl Iterator<Item = (String, String)>, cur: &str) -> Vec<(String, String)> {
    let typed = cur.to_lowercase();
    items
        .filter(|(value, _)| !value.is_empty() && value.to_lowercase().starts_with(&typed))
        .map(|(value, about)| {
            let about: String = about.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
            (value, out::cut(about.trim(), 72))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The machine
// ---------------------------------------------------------------------------

/// The real source: a server when the command would go to one, and this
/// machine's disk when it would not.
pub struct Live {
    choice: Choice,
    remote: OnceCell<Option<Remote>>,
    listing: OnceCell<Option<Value>>,
}

impl Live {
    /// `words` are read for `--remote` and `--local`, so that a Tab asks the
    /// server the command is about to be sent to.
    pub fn new(words: &[String]) -> Live {
        let mut choice = Choice::Unset;
        for (i, word) in words.iter().enumerate() {
            match (word.as_str(), words.get(i + 1)) {
                ("--local", _) => choice = Choice::Local,
                // Not the word being typed, which is the last one.
                ("--remote", Some(url)) if i + 2 < words.len() => choice = Choice::Remote(url.clone()),
                _ => {}
            }
        }
        Live { choice, remote: OnceCell::new(), listing: OnceCell::new() }
    }

    fn remote(&self) -> Option<&Remote> {
        self.remote
            .get_or_init(|| match client::resolve(&self.choice) {
                Ok(Target::Remote(remote)) => Some(remote.hurried()),
                _ => None,
            })
            .as_ref()
    }

    /// `/api/models`: what is on the server's disk, in its memory, and what
    /// it can load on. Asked once, whatever is wanted from it.
    fn listing(&self) -> Option<&Value> {
        self.listing.get_or_init(|| self.remote()?.get("/api/models").ok()).as_ref()
    }

    /// This machine's own models, read from its directories.
    fn on_this_disk() -> Vec<Model> {
        let active = kvad::hub::State::active();
        let row = |m: kvad::hub::LocalModel, trained: bool| Model {
            active: active.as_deref() == Some(m.id.as_str()),
            kind: String::new(),
            arch: m.arch.map(|a| a.to_string()),
            bytes: m.bytes,
            lora: m.lora.is_some(),
            complete: m.complete,
            trained,
            resident: false,
            id: m.id,
        };
        let trained = kvad::hub::trained_models().into_iter().map(|m| row(m, true));
        trained.chain(kvad::hub::local_models().into_iter().map(|m| row(m, false))).collect()
    }
}

impl Source for Live {
    fn models(&self) -> Vec<Model> {
        let Some(listing) = self.listing() else {
            // A server that was named and did not answer has models this
            // disk knows nothing of. Offering these would be a guess.
            return match self.choice {
                Choice::Remote(_) => Vec::new(),
                _ if self.remote().is_some() => Vec::new(),
                _ => Live::on_this_disk(),
            };
        };
        let resident: Vec<&str> = out::items(&listing["residents"]).iter().filter_map(|r| r["repo"].as_str()).collect();
        let active = listing["active"].as_str();
        out::items(&listing["trained"])
            .iter()
            .chain(out::items(&listing["downloaded"]))
            .filter_map(|m| {
                let id = m["id"].as_str()?;
                Some(Model {
                    id: id.to_string(),
                    kind: m["kind"].as_str().unwrap_or("").to_string(),
                    arch: m["arch"].as_str().map(str::to_string),
                    bytes: m["bytes"].as_u64().unwrap_or(0),
                    lora: m["lora"] == true,
                    complete: m["complete"] != false,
                    trained: m["trained"] == true,
                    resident: resident.iter().any(|r| r.eq_ignore_ascii_case(id)),
                    active: active == Some(id),
                })
            })
            .collect()
    }

    fn residents(&self) -> Vec<(String, String)> {
        let Some(listing) = self.listing() else {
            return Vec::new();
        };
        out::items(&listing["residents"])
            .iter()
            .filter_map(|r| Some((r["id"].as_str()?.to_string(), format!("{} · {}", out::s(&r["backend"]), out::bytes(&r["weight_bytes"])))))
            .collect()
    }

    fn backends(&self) -> Vec<(String, String)> {
        let Some(listing) = self.listing() else {
            return Vec::new();
        };
        let default = listing["backend"].as_str();
        out::items(&listing["backends"])
            .iter()
            .filter_map(|b| {
                let id = b["id"].as_str()?;
                let label = out::s(&b["label"]);
                Some((id.to_string(), if default == Some(id) { format!("{label} · the server's default") } else { label }))
            })
            .collect()
    }

    fn rows(&self, listing: Listing) -> Vec<(String, String)> {
        let Some(remote) = self.remote() else {
            return Vec::new();
        };
        // Where the listing is, which field names a row, and which fields
        // say what the row is.
        let (path, id, about): (&str, &str, &[&str]) = match listing {
            Listing::Jobs => ("/api/jobs?limit=50", "id", &["kind", "state", "label"]),
            Listing::Conversations => ("/api/conversations", "id", &["title"]),
            Listing::Images => ("/api/images", "id", &["prompt"]),
            Listing::Videos => ("/v1/videos?limit=100", "id", &["status", "prompt"]),
            Listing::Datasets => ("/api/datasets", "id", &["name", "kind"]),
            Listing::Suites => ("/api/evals/suites", "id", &["name"]),
            Listing::EvalRuns => ("/api/evals/runs", "id", &["state", "label"]),
            Listing::BenchRuns => ("/api/bench/runs", "id", &["state", "label"]),
            Listing::Users => ("/api/users", "id", &["name", "role"]),
            Listing::Keys => ("/api/keys", "id", &["name"]),
            Listing::Sessions => ("/api/sessions", "token_hash", &["user_agent"]),
        };
        let Ok(list) = remote.get(path) else {
            return Vec::new();
        };
        // The OpenAI-shaped listings keep their rows under `data`.
        let rows = match list["data"].is_array() {
            true => out::items(&list["data"]),
            false => out::items(&list),
        };
        rows.iter()
            .filter(|row| !row[id].is_null())
            .map(|row| {
                let name = out::s(&row[id]);
                let name = match listing {
                    // `video_63` to the API, `63` to `kvad videos`.
                    Listing::Videos => name.trim_start_matches("video_").to_string(),
                    // The start of a hash is enough, and is what the table shows.
                    Listing::Sessions => name.chars().take(12).collect(),
                    _ => name,
                };
                let said: Vec<String> = about.iter().map(|k| out::s(&row[*k])).filter(|v| v != "-").collect();
                (name, said.join(" · "))
            })
            .collect()
    }
}

/// `kvad __complete WORDS...`.
pub fn run(words: &[String]) {
    let live = Live::new(words);
    let answer = answer(words, &live);
    let mut text = String::new();
    for (value, about) in &answer.items {
        match about.is_empty() {
            true => text.push_str(&format!("{value}\n")),
            false => text.push_str(&format!("{value}\t{about}\n")),
        }
    }
    text.push_str(match answer.files {
        Files::None => ":none\n",
        Files::Files => ":files\n",
        Files::Dirs => ":dirs\n",
    });
    print!("{text}");
}

// ---------------------------------------------------------------------------
// The shells
// ---------------------------------------------------------------------------

/// `kvad completions [SHELL]`.
pub fn completions(shell: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    match shell {
        Some("zsh") => print!("{ZSH}"),
        Some("bash") => print!("{BASH}"),
        Some("fish") => print!("{FISH}"),
        Some(other) => return Err(format!("no completions for `{other}`: zsh, bash and fish are the shells there are scripts for").into()),
        None => print!("{}", setup()),
    }
    Ok(())
}

/// How to install them, with the reader's own shell first.
fn setup() -> String {
    let shell = std::env::var("SHELL").unwrap_or_default();
    let mine = ["zsh", "bash", "fish"].into_iter().find(|s| shell.ends_with(s));
    let how = |shell: &str| match shell {
        "zsh" => {
            "zsh — add to ~/.zshrc, after the line that runs compinit if there is one:\n\n    \
             eval \"$(kvad completions zsh)\"\n"
        }
        "bash" => "bash — add to ~/.bashrc (on macOS, ~/.bash_profile):\n\n    eval \"$(kvad completions bash)\"\n",
        _ => "fish — run once:\n\n    kvad completions fish > ~/.config/fish/completions/kvad.fish\n",
    };
    let mut out = String::from("Tab completion for kvad: commands, their options, and the models on this machine.\n\n");
    let order: Vec<&str> = mine.into_iter().chain(["zsh", "bash", "fish"].into_iter().filter(|s| Some(*s) != mine)).collect();
    for (i, shell) in order.iter().enumerate() {
        if i == 1 && mine.is_some() {
            out.push_str("Other shells:\n\n");
        }
        out.push_str(how(shell));
        out.push('\n');
    }
    out.push_str(
        "Then open a new terminal, and try:\n\n    \
         kvad <Tab>                  every command, and what it does\n    \
         kvad run --model <Tab>      the models on disk\n    \
         kvad videos make --<Tab>    what a video can be asked for\n\n\
         Nothing needs doing again after an upgrade: the script only asks the kvad that\n\
         is installed, so it knows whatever that one knows.\n",
    );
    out
}

/// zsh. Works both ways it can be installed: evaluated from `.zshrc`, where
/// the last line registers it, or as a file named `_kvad` on `$fpath`, where
/// the `#compdef` line does and the last line runs it.
const ZSH: &str = r#"#compdef kvad
# Tab completion for kvad, in zsh. From `kvad completions zsh`.
# It knows nothing itself: it asks `kvad __complete`, so it never goes stale.

_kvad() {
    local -a lines described
    local line value about directive kvad

    # The kvad that was typed, which may be a path, and may start with `~`.
    kvad=${(Q)words[1]}
    kvad=${~kvad}
    lines=("${(@f)$($kvad __complete "${(@)words[2,CURRENT]}" 2>/dev/null)}")
    directive=${lines[-1]}
    lines[-1]=()

    for line in $lines; do
        value=${line%%$'\t'*}
        about=
        [[ $line == *$'\t'* ]] && about=${line#*$'\t'}
        # A colon parts a candidate from its description, so one inside the
        # candidate (`repo:Q4_K_S`) has to be escaped.
        value=${value//:/\\:}
        described+=("${value}${about:+:$about}")
    done

    (( $#described )) && _describe -V -t candidates 'kvad' described -M 'm:{a-zA-Z}={A-Za-z}'
    case $directive in
        :files) _files ;;
        :dirs) _files -/ ;;
    esac
}

if [[ ${funcstack[1]} == _kvad ]]; then
    _kvad "$@"
else
    # Evaluated from .zshrc. The completion system may not be up yet.
    (( $+functions[compdef] )) || { autoload -Uz compinit && compinit }
    compdef _kvad kvad
fi
"#;

/// bash, 3.2 included: macOS still ships it, so no `mapfile`, and `compopt`
/// only where it exists.
const BASH: &str = r#"# Tab completion for kvad, in bash. From `kvad completions bash`.
# It knows nothing itself: it asks `kvad __complete`, so it never goes stale.

_kvad() {
    local cur cword line directive=
    local -a words

    # bash splits a word at each colon, which a model's name may have
    # (`repo:Q4_K_S`). bash-completion can put it back together.
    if declare -F _get_comp_words_by_ref >/dev/null; then
        _get_comp_words_by_ref -n : cur words cword
    else
        cur=${COMP_WORDS[COMP_CWORD]}
        words=("${COMP_WORDS[@]}")
        cword=$COMP_CWORD
    fi

    COMPREPLY=()
    while IFS= read -r line; do
        case $line in
            :files | :dirs | :none) directive=$line ;;
            *) COMPREPLY+=("${line%%$'\t'*}") ;;
        esac
    done < <("${words[0]}" __complete "${words[@]:1:$cword}" 2>/dev/null)

    local IFS=$'\n'
    case $directive in
        :files)
            COMPREPLY+=($(compgen -f -- "$cur"))
            compopt -o filenames 2>/dev/null
            ;;
        :dirs)
            COMPREPLY+=($(compgen -d -- "$cur"))
            compopt -o filenames 2>/dev/null
            ;;
    esac
    if declare -F __ltrim_colon_completions >/dev/null; then
        __ltrim_colon_completions "$cur"
    fi
}

complete -F _kvad kvad
"#;

/// fish, which shows descriptions and filters the candidates itself.
const FISH: &str = r#"# Tab completion for kvad, in fish. From `kvad completions fish`.
# It knows nothing itself: it asks `kvad __complete`, so it never goes stale.

function __kvad_complete
    set -l cur (commandline -ct)
    set -l words (commandline -opc) "$cur"
    set -l lines (kvad __complete $words[2..-1] 2>/dev/null)

    switch "$lines[-1]"
        case ':files'
            __fish_complete_path "$cur"
        case ':dirs'
            __fish_complete_directories "$cur"
    end
    string match -v -r '^:(files|dirs|none)$' -- $lines
end

complete -c kvad -f -a '(__kvad_complete)'
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// A machine with three models, one of them in memory.
    struct Fake;

    impl Source for Fake {
        fn models(&self) -> Vec<Model> {
            let model = |id: &str, kind: &str, arch: Option<&str>| Model {
                id: id.into(),
                kind: kind.into(),
                arch: arch.map(str::to_string),
                bytes: 1 << 30,
                complete: true,
                ..Model::default()
            };
            vec![
                Model { trained: true, ..model("book", "chat", Some("gpt2")) },
                Model { resident: true, active: true, ..model("Qwen/Qwen3-14B", "chat", Some("llama")) },
                model("openai/clip-vit-large-patch14", "chat", None),
                model("stabilityai/sdxl-turbo", "image", None),
                model("city96/FLUX.1-schnell-gguf:Q4_K_S", "image", None),
                model("Lightricks/LTX-2.5", "video", None),
                Model { lora: true, ..model("lightx2v/Lightning:8steps.safetensors", "chat", None) },
            ]
        }

        fn residents(&self) -> Vec<(String, String)> {
            vec![("Qwen/Qwen3-14B@gpu-q8".into(), "metal q8".into())]
        }

        fn backends(&self) -> Vec<(String, String)> {
            vec![("cpu-q8".into(), "cpu q8".into()), ("gpu-q8".into(), "gpu q8".into())]
        }

        fn rows(&self, listing: Listing) -> Vec<(String, String)> {
            match listing {
                Listing::Jobs => vec![("16".into(), "pull · done".into()), ("17".into(), "train · running".into())],
                Listing::Datasets => vec![("1".into(), "book · text".into())],
                _ => Vec::new(),
            }
        }
    }

    /// What a Tab offers after `line`, a trailing space meaning a new word.
    fn tab(line: &str) -> Vec<String> {
        let mut words: Vec<String> = line.split(' ').map(String::from).collect();
        assert_eq!(words.remove(0), "kvad");
        answer(&words, &Fake).items.into_iter().map(|(value, _)| value).collect()
    }

    #[test]
    fn a_bare_tab_is_every_command() {
        let all = tab("kvad ");
        assert_eq!(all.len(), spec::COMMANDS.len());
        assert_eq!(tab("kvad ch"), ["chat"]);
        assert_eq!(tab("kvad help vi"), ["videos"]);
        assert!(tab("kvad frobnicate ").is_empty());
    }

    #[test]
    fn a_model_is_offered_where_one_is_wanted_and_of_the_kind_wanted() {
        assert_eq!(tab("kvad run --model "), ["book", "Qwen/Qwen3-14B"]);
        assert_eq!(tab("kvad use q"), ["Qwen/Qwen3-14B"]);
        assert_eq!(tab("kvad images make --model "), ["stabilityai/sdxl-turbo", "city96/FLUX.1-schnell-gguf:Q4_K_S"]);
        assert_eq!(tab("kvad videos make a boat --model "), ["Lightricks/LTX-2.5"]);
        assert_eq!(tab("kvad images make --lora "), ["lightx2v/Lightning:8steps.safetensors"]);
        assert_eq!(tab("kvad train --from "), ["book"]);
        assert_eq!(tab("kvad rm ").len(), 7);
        assert_eq!(tab("kvad unload "), ["Qwen/Qwen3-14B@gpu-q8"]);
        assert_eq!(tab("kvad load x --backend g"), ["gpu-q8"]);
    }

    #[test]
    fn a_model_says_what_it_is() {
        let words = ["use".to_string(), "Q".to_string()];
        let items = answer(&words, &Fake).items;
        assert_eq!(items[0].1, "llama · 1.0 GB · in memory · the default");
    }

    #[test]
    fn subcommands_then_their_words_then_their_flags() {
        assert_eq!(tab("kvad jobs "), ["ls", "show", "watch", "cancel", "pictures"]);
        assert_eq!(tab("kvad jobs show "), ["16", "17"]);
        // One id is all `show` takes: what is left is options.
        assert!(tab("kvad jobs show 16 ").contains(&"--json".to_string()));
        assert_eq!(tab("kvad jobs pictures 16 --o"), ["--out"]);
        // `cache` takes a model or the word `clear`.
        assert_eq!(tab("kvad cache "), ["clear", "book", "Qwen/Qwen3-14B"]);
        assert_eq!(tab("kvad service l"), ["logs"]);
        assert_eq!(tab("kvad service logs -"), ["--follow", "--lines", "--help"]);
    }

    #[test]
    fn flags_are_the_commands_own_and_each_once() {
        let run = tab("kvad run --");
        assert!(run.contains(&"--prompt".to_string()) && run.contains(&"--remote".to_string()));
        assert!(!run.contains(&"--epilogue".to_string()));
        assert!(!tab("kvad run --prompt hello --").contains(&"--prompt".to_string()));
        // A prompt is not a place to offer anything, and a flag's value is
        // not a place to offer flags.
        assert!(tab("kvad run --prompt ").is_empty());
        assert!(tab("kvad run --prompt --").is_empty());
        // No command at all is `run`.
        assert_eq!(tab("kvad --prom"), ["--prompt"]);
        assert_eq!(tab("kvad --model b"), ["book"]);
        // A LoRA may be given twice.
        assert!(tab("kvad images make x --lora a --").contains(&"--lora".to_string()));
        // `edit` takes what `make` takes, and its own.
        let edit = tab("kvad images edit x --");
        assert!(edit.contains(&"--mask".to_string()) && edit.contains(&"--guidance".to_string()));
        assert!(!tab("kvad images make x --").contains(&"--mask".to_string()));
    }

    #[test]
    fn choices_variants_and_routes() {
        assert_eq!(tab("kvad run --quant q"), ["q8", "q4"]);
        assert_eq!(tab("kvad videos make --pipeline "), ["fast", "dfr"]);
        assert_eq!(tab("kvad train --size "), ["small", "medium", "large"]);
        assert_eq!(tab("kvad bench run Qwen/Qwen3-14B@"), ["Qwen/Qwen3-14B@cpu-q8", "Qwen/Qwen3-14B@gpu-q8"]);
        assert_eq!(tab("kvad bench run a@cpu-q8 b"), ["book"]);
        assert_eq!(tab("kvad evals perplexity "), ["1"]);
        assert_eq!(tab("kvad api /api/he"), ["/api/health"]);
        assert_eq!(tab("kvad api delete /api/k"), ["/api/keys/{id}"]);
        assert_eq!(tab("kvad api po"), ["post"]);
        assert_eq!(tab("kvad completions "), ["zsh", "bash", "fish"]);
    }

    #[test]
    fn files_are_the_shells_to_complete() {
        let files = |line: &str| {
            let words: Vec<String> = line.split(' ').skip(1).map(String::from).collect();
            answer(&words, &Fake).files
        };
        assert_eq!(files("kvad train --data "), Files::Files);
        assert_eq!(files("kvad tune --data "), Files::Dirs);
        assert_eq!(files("kvad images edit x --image "), Files::Files);
        assert_eq!(files("kvad run --model "), Files::None);
        assert_eq!(files("kvad run --model ./"), Files::Dirs);
        assert_eq!(files("kvad run --prompt "), Files::None);
    }

    /// Every word of every command has something written for it: a walk of
    /// the whole table that must not panic, whatever the position.
    #[test]
    fn every_position_of_every_command_answers() {
        for c in spec::COMMANDS {
            for sub in c.subs.iter().map(|s| Some(s.name)).chain([None]) {
                for extra in 0..4 {
                    let mut words = vec![c.name.to_string()];
                    words.extend(sub.map(String::from));
                    words.extend(std::iter::repeat_n("x".to_string(), extra));
                    for cur in ["", "-", "--"] {
                        let mut words = words.clone();
                        words.push(cur.to_string());
                        answer(&words, &Fake);
                    }
                }
            }
            for flag in c.all_flags() {
                answer(&[c.name.to_string(), flag.name.to_string(), String::new()], &Fake);
            }
        }
    }
}
