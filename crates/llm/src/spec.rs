//! What a command line is made of, and the three small ones.
//!
//! [`Command`], [`Flag`] and [`Kind`] are how a command, an option and the
//! thing an option wants are written down: once, in a table, so that the
//! help that prints it and the Tab completion that offers it cannot say
//! different things. [`crate::help`] is the printing.
//!
//! This is in the library because four binaries have a command line. `kvad`'s
//! own table is beside its parser, in that binary. The other three —
//! `kvad-serve`, `kvad-gpu` and `kvad-tui` — have theirs here, where each
//! can print its own help from it and `kvad` can complete all of them, which
//! it does because it is the one binary every install has.
//!
//! No parser is driven by these tables. Each binary's stays a `match`, which
//! is the plainest way to say what a flag does, and the tests at the bottom
//! read those parsers' source to hold them to what is said here.

/// What a word or a flag's value *is*, which decides what a Tab offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Free text, a number, an address: nothing to offer.
    Text,
    File,
    Dir,
    /// A model on disk — the server's when one answers, this machine's
    /// otherwise.
    Model(Want),
    /// A model in a server's memory, by the id `kvad ps` shows.
    Resident,
    /// A backend a server loads on: `gpu-q8`.
    Backend,
    /// `MODEL[@BACKEND]`, as `kvad bench run` takes them.
    Variant,
    /// A LoRA on disk, `NAME` or `NAME:SCALE`.
    Lora,
    /// A row of one of the server's listings, by its id.
    Id(Listing),
    /// One of a few words, each with what it means.
    Choice(&'static [(&'static str, &'static str)]),
    /// A model shape `kvad train --size` takes.
    TrainSize,
    /// A command's name, for `kvad help`.
    Command,
    /// `kvad api`'s first word: a method or a path.
    Route,
}

/// Which models a word wants offered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    /// Anything on disk.
    Any,
    /// A language model with its weights here.
    Text,
    Image,
    Video,
    /// One trained here, which `--from` can train further.
    Trained,
}

/// The server's listings a Tab can read ids from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Listing {
    Jobs,
    Conversations,
    Images,
    Videos,
    Datasets,
    Suites,
    EvalRuns,
    BenchRuns,
    Users,
    Keys,
    Sessions,
}

pub struct Flag {
    pub name: &'static str,
    /// The one-letter spelling, or nothing.
    pub short: &'static str,
    /// What the value is called in the usage, or nothing for a switch.
    pub value: &'static str,
    pub kind: Kind,
    pub about: &'static str,
}

/// A flag that takes a value.
pub const fn f(name: &'static str, value: &'static str, kind: Kind, about: &'static str) -> Flag {
    Flag { name, short: "", value, kind, about }
}

/// A flag that stands alone.
pub const fn sw(name: &'static str, about: &'static str) -> Flag {
    Flag { name, short: "", value: "", kind: Kind::Text, about }
}

impl Flag {
    pub fn takes_value(&self) -> bool {
        !self.value.is_empty()
    }
}

pub struct Sub {
    pub name: &'static str,
    /// The words after it, as the usage writes them. One that ends `...`
    /// repeats.
    pub args: &'static str,
    pub about: &'static str,
    /// What each of those words is, in order.
    pub words: &'static [Kind],
    /// In groups, as a command's are.
    pub flags: &'static [&'static [Flag]],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Models,
    Use,
    Make,
    Train,
    Server,
    Measure,
    Accounts,
    Cli,
}

impl Group {
    pub const ALL: [Group; 8] = [
        Group::Models,
        Group::Use,
        Group::Make,
        Group::Train,
        Group::Server,
        Group::Measure,
        Group::Accounts,
        Group::Cli,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Group::Models => "get a model",
            Group::Use => "talk to one",
            Group::Make => "make pictures and clips",
            Group::Train => "train your own",
            Group::Server => "the server",
            Group::Measure => "measure",
            Group::Accounts => "accounts, on a server that has them",
            Group::Cli => "this program",
        }
    }
}

pub struct Command {
    pub name: &'static str,
    pub group: Group,
    /// Its words, as the overview and the usage write them.
    pub args: &'static str,
    /// One line, for the overview and for a shell's menu.
    pub about: &'static str,
    /// Only a running server can answer it.
    pub server: bool,
    /// Takes `--remote`, `--local` and `--json`: everything but the few
    /// commands that are about this machine and nothing else.
    pub routed: bool,
    pub words: &'static [Kind],
    pub subs: &'static [Sub],
    /// Groups of flags, so that one written down once — the sampling
    /// flags — can belong to several commands.
    pub flags: &'static [&'static [Flag]],
    /// Usage and prose written by hand, where a command has them. Printed
    /// as they are; the options and examples follow.
    pub usage: Option<&'static str>,
    /// What is worth knowing that an option's one line cannot hold.
    pub notes: &'static str,
    pub examples: &'static [&'static str],
}

impl Command {
    pub fn sub(&self, name: &str) -> Option<&'static Sub> {
        self.subs.iter().find(|s| s.name == name)
    }

    /// Every flag it reads, its subcommands' among them, each name once.
    pub fn all_flags(&self) -> Vec<&'static Flag> {
        let mut seen: Vec<&'static Flag> = Vec::new();
        let own = self.flags.iter().flat_map(|g| g.iter());
        for flag in own.chain(self.subs.iter().flat_map(|s| s.flags.iter().flat_map(|g| g.iter()))) {
            if !seen.iter().any(|s| s.name == flag.name) {
                seen.push(flag);
            }
        }
        seen
    }
}

/// One of the binaries kvad is installed as.
pub struct Program {
    pub name: &'static str,
    /// One line: what it is for.
    pub about: &'static str,
    pub commands: &'static [Command],
    /// What flags and words belong to when no command is named: `run`, for
    /// the two that have commands, and the program itself for the two that
    /// have none.
    pub bare: &'static Command,
    /// Runs its models in its own process, so the models to offer at a Tab
    /// are this disk's and never a server's.
    pub local: bool,
}

impl Program {
    pub fn command(&self, name: &str) -> Option<&'static Command> {
        self.commands.iter().find(|c| c.name == name)
    }

    /// Whether a flag takes a value, asked of every command at once: a
    /// parser has one answer for each spelling, whichever command it
    /// follows.
    pub fn takes_value(&self, flag: &str) -> Option<bool> {
        let routed = self.commands.iter().any(|c| c.routed);
        WHERE
            .iter()
            .filter(|_| routed)
            .chain(self.commands.iter().chain([self.bare]).flat_map(|c| c.all_flags()))
            .find(|f| f.name == flag || (!f.short.is_empty() && f.short == flag))
            .map(Flag::takes_value)
    }
}

// ---------------------------------------------------------------------------
// Flags more than one program has
// ---------------------------------------------------------------------------

/// Where a command runs. Every routed command takes these.
pub const WHERE: &[Flag] = &[
    f("--remote", "URL", Kind::Text, "send it to the kvad-serve at URL"),
    sw("--local", "run it in this process, even with a server running"),
    sw("--json", "print what the server sent, as JSON"),
];

pub const SAMPLING: &[Flag] = &[
    f("--max-tokens", "N", Kind::Text, "most tokens to write (default 4096 for a reply; 256 when continuing text, as a base model or --raw does)"),
    f("--temperature", "F", Kind::Text, "how freely to choose; 0 always takes the likeliest (default 0.7)"),
    f("--top-k", "N", Kind::Text, "choose among the N likeliest tokens (default 40)"),
    f("--top-p", "F", Kind::Text, "or among the likeliest that add up to F (default 0.95)"),
    f("--seed", "N", Kind::Text, "the same seed writes the same text (default 7)"),
    sw("--greedy", "shorthand for --temperature 0"),
];

/// What `kvad-serve` takes, and so what `kvad serve` hands on to it.
pub const SERVE_FLAGS: &[Flag] = &[
    f("--bind", "HOST:PORT", Kind::Text, "where to listen (default: server.bind in kvad.toml, else 127.0.0.1:5823)"),
    f("--config", "FILE", Kind::File, "a kvad.toml other than the usual one"),
    f("--db", "FILE", Kind::File, "the database to keep conversations, jobs and accounts in"),
    sw("--insecure", "listen beyond this machine with no accounts; anyone who can reach it can use it"),
    Flag { name: "--version", short: "-V", value: "", kind: Kind::Text, about: "print the version and exit" },
];

// ---------------------------------------------------------------------------
// The three small programs
// ---------------------------------------------------------------------------

/// A command with no subcommands, on no server.
const fn plain(
    name: &'static str,
    args: &'static str,
    about: &'static str,
    words: &'static [Kind],
    flags: &'static [&'static [Flag]],
    notes: &'static str,
    examples: &'static [&'static str],
) -> Command {
    Command { name, group: Group::Cli, args, about, server: false, routed: false, words, subs: &[], flags, usage: None, notes, examples }
}

/// Serving, which is what `kvad-serve` does with no command.
const SERVING: Command = plain(
    "serve",
    "",
    "serve the API and the web UI until Ctrl-C; the default",
    &[],
    &[SERVE_FLAGS],
    "Serves the OpenAI-compatible API and the web UI in this terminal, until Ctrl-C.\n\
     `kvad serve` is the same thing. To keep it running after the terminal closes,\n\
     and start it at login, `kvad service install` installs it as a service.\n\n\
     An address beyond this machine is refused while the configuration has no\n\
     accounts, unless --insecure says that is meant.",
    &["kvad-serve", "kvad-serve --bind 127.0.0.1:8080", "kvad-serve --config ./kvad.toml --db ./kvad.db"],
);

pub const SERVE: Program = Program {
    name: "kvad-serve",
    about: "the HTTP server and web UI",
    commands: &[
        SERVING,
        plain(
            "password",
            "NAME",
            "set an account's password, for when it is lost",
            &[Kind::Text],
            &[&[
                f("--config", "FILE", Kind::File, "the kvad.toml of the server, if not the usual one"),
                f("--db", "FILE", Kind::File, "the server's database, if not the one kvad.toml names"),
            ]],
            "Opens the database the server with the same --config and --db would, asks for\n\
             a new password twice, and sets it. Every session of the account ends, as when\n\
             it changes its own; its API keys go on working. The server can be running.\n\n\
             It gives nothing away: whoever can open the database file already owns the\n\
             server, every account and conversation in it, and could edit the rows by hand.",
            &["kvad-serve password ada", "kvad-serve password ada --config ./kvad.toml"],
        ),
    ],
    bare: &SERVING,
    local: false,
};

const GPU_FLAGS: &[Flag] = &[
    f("--model", "REPO", Kind::Model(Want::Text), "a name trained here, a directory, or a Hub repo id (default: the one `kvad use` set)"),
    f(
        "--device",
        "D",
        Kind::Choice(&[("metal", "the Mac's GPU"), ("cuda", "an NVIDIA GPU, in a build that has it"), ("cpu", "candle's CPU backend")]),
        "where to run: metal, cuda or cpu (default: the best there is)",
    ),
    f(
        "--dtype",
        "T",
        Kind::Choice(&[("bf16", "what these checkpoints ship as, and the default"), ("f16", "half precision"), ("f32", "single precision: twice the memory")]),
        "the precision the weights are held at: bf16, f16 or f32 (default bf16)",
    ),
    f(
        "--quant",
        "Q",
        Kind::Choice(&[
            ("none", "not quantised, and the default"),
            ("q8", "8 bits a weight"),
            ("q4", "4 bits a weight"),
            ("q4k", "4 bits, k-quantised: a little better than q4"),
            ("q6k", "6 bits, k-quantised"),
        ]),
        "quantise the weights on load: none, q8, q4, q4k or q6k (default none)",
    ),
    f("--prompt", "TEXT", Kind::Text, "what `run` answers (default: a prompt that shows the model works)"),
    f("--system", "TEXT", Kind::Text, "a system prompt"),
];

pub const GPU: Program = Program {
    name: "kvad-gpu",
    about: "the same models on the GPU, in this terminal, with no server",
    commands: &[
        plain(
            "run",
            "",
            "answer one prompt and exit",
            &[],
            &[GPU_FLAGS, SAMPLING],
            "`kvad-gpu --prompt TEXT`, with no command, is `kvad-gpu run --prompt TEXT`.\n\n\
             Everything but the forward pass is the CPU engine's: the same Hub client,\n\
             tokenizer, chat template and sampler. With `kvad-serve` running, `kvad run`\n\
             reaches the same GPU through it, and keeps the model loaded between runs.",
            &["kvad-gpu run --prompt \"Why is the sky blue?\"", "kvad-gpu run --model Qwen/Qwen3-14B --quant q4k --prompt \"...\""],
        ),
        plain(
            "chat",
            "",
            "a conversation in the terminal",
            &[],
            &[GPU_FLAGS, SAMPLING],
            "Inside it: /reset forgets the conversation so far, /quit leaves.",
            &["kvad-gpu chat", "kvad-gpu chat --model Qwen/Qwen3-14B --system \"Answer in one sentence.\""],
        ),
        plain(
            "tune",
            "[MODEL] --data DIR --name NAME",
            "train a LoRA for SDXL on a folder of pictures",
            &[Kind::Model(Want::Image)],
            &[&[
                f("--data", "DIR", Kind::Dir, "the folder of pictures and captions"),
                f("--name", "NAME", Kind::Text, "what to call the LoRA; it is written to the data directory's loras/NAME.safetensors"),
                f("--out", "FILE", Kind::File, "or the file to write it to"),
                f("--from", "FILE", Kind::File, "go on from a LoRA this wrote"),
                f("--caption", "TEXT", Kind::Text, "the caption of every picture that has none"),
                f("--size", "N", Kind::Text, "pixels a side, a multiple of 64 (default 1024)"),
                f("--rank", "N", Kind::Text, "the LoRA's rank (default 16)"),
                f("--alpha", "F", Kind::Text, "the LoRA is scaled by alpha / rank (default: the rank)"),
                f("--steps", "N", Kind::Text, "training steps (default 1000)"),
                f("--lr", "F", Kind::Text, "learning rate (default 1e-4)"),
                f("--eval-every", "N", Kind::Text, "steps between validation measurements (default 100)"),
                f("--holdout", "N", Kind::Text, "pictures kept out of training to measure on (default: a tenth, from one to four; none of under 5)"),
                f("--sample", "TEXT", Kind::Text, "a prompt to draw before the first step and at every measurement, from the same seed each time; may be given more than once"),
                f("--sample-size", "N", Kind::Text, "pixels a side of a sample (default: --size)"),
                f("--sample-steps", "N", Kind::Text, "a sample's denoising steps (default 20)"),
                f("--samples", "DIR", Kind::Dir, "where the samples are written (default: the LoRA's file's name, ending .samples)"),
                f("--seed", "N", Kind::Text, "(default 1337)"),
                f("--ffmpeg", "FILE", Kind::File, "the ffmpeg that decodes the pictures (default: found)"),
                f("--cap", "GB", Kind::Text, "end the run if its memory passes this (default: three quarters of the machine's)"),
                f(
                    "--progress",
                    "json",
                    Kind::Choice(&[("json", "one JSON object a line, for a program that follows the run")]),
                    "also say what the run does on standard output, one JSON object a line, for a program that follows it; the run then ends when the program that started it does",
                ),
            ]],
            "Each picture's caption is a .txt of the same name beside it. MODEL is SDXL's\n\
             repo unless given. Ctrl-C ends the run after the step it is in, with its last\n\
             step written beside its best.\n\n\
             `kvad tune` trains the same LoRA as a job on a server, where it can be\n\
             watched from the web UI; this has --from, --alpha and --holdout, which that\n\
             does not.",
            &["kvad-gpu tune --data ./my-dog --name my-dog --caption \"a photo of sks dog\"", "kvad-gpu tune --data ./my-dog --from my-dog.safetensors --out my-dog-2.safetensors --steps 500"],
        ),
    ],
    bare: &plain("run", "", "", &[], &[GPU_FLAGS, SAMPLING], "", &[]),
    local: true,
};

pub const TUI: Program = Program {
    name: "kvad-tui",
    about: "browse, download and chat with models in the terminal",
    commands: &[],
    bare: &plain(
        "kvad-tui",
        "[MODEL]",
        "browse, download and chat with models in the terminal",
        &[Kind::Model(Want::Text)],
        &[&[Flag { name: "--version", short: "-V", value: "", kind: Kind::Text, about: "print the version and exit" }]],
        "MODEL is loaded as it opens: a name trained here, a directory, or a Hub repo\n\
         id. With none it opens on the list of models, where one can be searched for,\n\
         downloaded and loaded.\n\n\
         It runs models in its own process, on the GPU where there is one, and does\n\
         not use a running kvad-serve. The line at the bottom of the screen says which\n\
         keys do what, where you are.",
        &["kvad-tui", "kvad-tui Qwen/Qwen2.5-1.5B-Instruct"],
    ),
    local: true,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The flags a source matches on between two markers: `"--flag" =>`
    /// arms and `flag == "--flag"` tests alike.
    fn dashed(source: &str, from: &str, to: &str) -> Vec<String> {
        let start = source.find(from).unwrap_or_else(|| panic!("no `{from}`"));
        let end = start + source[start..].find(to).unwrap_or_else(|| panic!("no `{to}` after `{from}`"));
        let mut found: Vec<String> = Vec::new();
        for piece in source[start..end].split('"').skip(1).step_by(2) {
            let flag = piece.len() > 2 && piece.starts_with("--") && piece[2..].chars().all(|c| c == '-' || c.is_ascii_alphanumeric());
            if flag && !found.iter().any(|f| f == piece) {
                found.push(piece.to_string());
            }
        }
        found
    }

    fn names(flags: Vec<&'static Flag>) -> Vec<String> {
        flags.into_iter().map(|f| f.name.to_string()).collect()
    }

    fn same_flags(what: &str, mut table: Vec<String>, mut parser: Vec<String>) {
        table.sort();
        parser.sort();
        assert_eq!(table, parser, "{what}: the table on the left, its parser on the right");
    }

    /// The other binaries' flags are the ones their own parsers take.
    ///
    /// Read from their source, which is in other crates: nothing but a test
    /// reaches across, so the crates still build apart.
    #[test]
    fn the_other_binaries_flags_are_their_parsers() {
        let gpu = include_str!("../../gpu/src/main.rs");
        let serve = include_str!("../../serve/src/main.rs");
        let of = |program: &str, command: &str| {
            let p = program_named(program);
            names(p.command(command).unwrap_or(p.bare).all_flags())
        };

        // `--help` and `--version` are asked for before either parser runs.
        let asked = |f: &String| !matches!(f.as_str(), "--help" | "--version");
        let parsed: Vec<String> = dashed(gpu, "fn parse_args()", "fn load(").into_iter().filter(asked).collect();
        same_flags("kvad-gpu run", of("kvad-gpu", "run"), parsed.clone());
        same_flags("kvad-gpu chat", of("kvad-gpu", "chat"), parsed);

        // `tune` takes whatever follows two dashes and looks each one up by
        // name afterwards: `take("data")`, a row of `("size", &mut opts.size)`,
        // and `"sample"`, the one it keeps a list of.
        let tune = &gpu[gpu.find("fn tune(").expect("kvad-gpu's tune")..];
        let mut taken: Vec<String> = vec!["--sample".into()];
        for (opener, closer) in [("take(\"", "\")"), ("(\"", "\", &mut opts.")] {
            for (at, _) in tune.match_indices(opener) {
                let rest = &tune[at + opener.len()..];
                let name = &rest[..rest.find('"').unwrap_or(0)];
                let flag = format!("--{name}");
                if rest[name.len()..].starts_with(closer) && !name.is_empty() && !taken.contains(&flag) {
                    taken.push(flag);
                }
            }
        }
        same_flags("kvad-gpu tune", of("kvad-gpu", "tune"), taken);

        same_flags("kvad-serve", of("kvad-serve", ""), dashed(serve, "fn parse_args()", "#[tokio::main]").into_iter().filter(|f| f != "--help").collect());

        // And a flag that stands alone there stands alone here.
        for (program, alone) in [("kvad-gpu", vec!["--greedy"]), ("kvad-serve", vec!["--insecure", "--version"]), ("kvad-tui", vec!["--version"])] {
            let p = program_named(program);
            for c in p.commands.iter().chain([p.bare]) {
                for flag in c.all_flags() {
                    assert_eq!(!flag.takes_value(), alone.contains(&flag.name), "{program} {}: {}", c.name, flag.name);
                }
            }
        }

        // Every one of them answers `--help` from this table, and none
        // keeps a usage text of its own to fall behind it.
        let tui = include_str!("../../tui/src/main.rs");
        for (name, source) in [("kvad-gpu", gpu), ("kvad-serve", serve), ("kvad-tui", tui)] {
            assert!(source.contains("kvad::help::"), "{name} does not print its help from the table");
            assert!(!source.contains("fn usage()"), "{name} has a usage of its own again");
        }
    }

    fn program_named(name: &str) -> &'static Program {
        [&SERVE, &GPU, &TUI].into_iter().find(|p| p.name == name).unwrap_or_else(|| panic!("no program `{name}`"))
    }
}
