//! Every command, its words and its flags, written down once.
//!
//! Two things read this table and nothing else: [`super::help`], which
//! prints it, and [`super::complete`], which offers it to a shell at a Tab.
//! That is the whole reason it is a table and not prose in a `println!` —
//! the help said what the commands were, the completion would have had to
//! say it again, and the second copy of anything is the one that is wrong.
//!
//! The parser in `main.rs` is not driven by it. It stays a `match`, which is
//! the plainest way to say what a flag does, and the tests at the bottom
//! hold the two together: every flag here is one the parser takes, taking a
//! value where this says it does, and every flag the parser takes is here.

pub use kvad::spec::*;

pub fn command(name: &str) -> Option<&'static Command> {
    COMMANDS.iter().find(|c| c.name == name)
}

/// Whether a flag takes a value, asked of every command at once: the parser
/// has one answer for each spelling, whichever command it follows.
pub fn takes_value(flag: &str) -> Option<bool> {
    PROGRAMS[0].takes_value(flag)
}

pub fn program(name: &str) -> Option<&'static Program> {
    PROGRAMS.iter().find(|p| p.name == name)
}

/// Every binary kvad is installed as. `kvad` is the first, and the only one
/// this binary *is*; the other three are the library's, and are here
/// because a shell completing any of them asks this one.
pub const PROGRAMS: &[Program] = &[
    Program { name: "kvad", about: "run, train and serve language, image and video models", commands: COMMANDS, bare: &RUN, local: false },
    kvad::spec::SERVE,
    kvad::spec::GPU,
    kvad::spec::TUI,
];

// ---------------------------------------------------------------------------
// Flags several commands share
// ---------------------------------------------------------------------------

const YES: Flag = Flag { name: "--yes", short: "-y", value: "", kind: Kind::Text, about: "do not ask before deleting" };

const QUANT: Flag = f(
    "--quant",
    "f32|q8|q4",
    Kind::Choice(&[
        ("f32", "weights as they were published"),
        ("q8", "8 bits a weight: a quarter of the memory"),
        ("q4", "4 bits a weight: an eighth, and a little worse"),
    ]),
    "quantise the weights on load (default f32); on a server, its CPU backend at that precision",
);

const BACKEND: Flag = f("--backend", "ID", Kind::Backend, "on a server: the backend to load on, such as gpu-q8");

const TEXT_MODEL: Flag = f(
    "--model",
    "MODEL",
    Kind::Model(Want::Text),
    "a name trained here, a directory, or a Hub repo id (default: the one `kvad use` set)",
);

const CRAWL: &[Flag] = &[
    f("--pages", "N", Kind::Text, "most pages to read (default 400)"),
    f("--mb", "F", Kind::Text, "most megabytes of text to collect (default 16)"),
    f("--pause", "MS", Kind::Text, "milliseconds to wait between requests (default 250)"),
    f("--drop-rare", "N", Kind::Text, "drop characters seen fewer than N times (default 10)"),
    sw("--same-host", "follow links anywhere on the host, not only under the address's own directory"),
];

/// `kvad cluster`'s usage. Here and not beside the command, which a build
/// without the `cluster` feature does not have.
pub const CLUSTER: &str = "usage: kvad cluster plan MODEL --workers NAME[,NAME]   which machine would hold what
       kvad cluster status              what a running server's part in a cluster is
       kvad cluster find [--all]        who is on the other end of each cable
       kvad cluster listen [--port N]   be the far end of a measurement
       kvad cluster ping [HOST]         measure the link to a machine that is listening
       kvad cluster link IFACE          say what one interface's link is

With no HOST, `ping` measures the one machine `find` finds, if it finds one.";

/// `run` and `chat`, for a model spread over several machines.
const ACROSS: &[Flag] = &[
    f("--workers", "NAME[,NAME]", Kind::Text, "spread the model over these machines, each running `kvad worker`: by name, id or address"),
    f("--split", "A..B,C..D", Kind::Text, "the layers each machine holds, this one first, instead of the ones worked out"),
    f("--context", "N", Kind::Text, "the longest conversation to leave memory for, in tokens (default: the model's, at most 8192)"),
    f(
        "--place",
        "reply|prompt|N:M",
        Kind::Text,
        "what to place the layers for: the reply written quickest (default), the prompt read quickest, or N tokens read to M written",
    ),
];

const CLUSTER_PORT: Flag = f("--port", "N", Kind::Text, "the port to listen on, or to reach (default 7420)");

const TRANSCRIBE: &[Flag] = &[
    f("--model", "MODEL", Kind::Model(Want::Any), "the Whisper to use: a Hub repo id or a directory"),
    f("--language", "CODE", Kind::Text, "what is spoken, as no or en (default: the model's own, or detected); auto detects"),
    sw("--translate", "write it down in English, whatever is spoken"),
    sw("--timestamps", "time each stretch of text (on for srt, vtt and verbose_json)"),
    sw("--words", "time every word as well, in verbose_json (its default with this)"),
    f(
        "--format",
        "F",
        Kind::Choice(&[
            ("text", "the words (default)"),
            ("json", "{\"text\": …}"),
            ("verbose_json", "the words, the language, and every segment with its times"),
            ("srt", "subtitles"),
            ("vtt", "subtitles for the web"),
        ]),
        "what to write",
    ),
    f("--prompt", "T", Kind::Text, "text to read the recording as following: names, spelling, style"),
    f("--temperature", "F", Kind::Text, "above 0, decode at this temperature only (default, and 0: greedy, warmer for a window that fails)"),
    f("--out", "FILE", Kind::File, "where to write it (default: the terminal)"),
    sw("--live", "with FILE `-`: raw 16 kHz mono f32 from stdin, written down as it comes, in this process"),
    QUANT,
    BACKEND,
];

const LORA: Flag = f("--lora", "NAME[:SCALE]", Kind::Lora, "apply a LoRA, at SCALE if given (default 1); as often as you like");

// ---------------------------------------------------------------------------
// The commands
// ---------------------------------------------------------------------------

const ROLES: Kind = Kind::Choice(&[("admin", "may manage accounts and settings"), ("user", "may use the models")]);

pub const COMMANDS: &[Command] = &[
    // -- get a model -------------------------------------------------------
    Command {
        name: "search",
        group: Group::Models,
        args: "QUERY",
        about: "find models on the Hub, and say which of them run here",
        server: false,
        routed: true,
        words: &[Kind::Text],
        subs: &[],
        flags: &[],
        usage: None,
        notes: "The STATUS column is the answer to \"can I run this\": whether it is a chat or a\n\
                completion model and whether it fits in memory, or the reason it will not run.",
        examples: &["kvad search smollm", "kvad search qwen coder"],
    },
    Command {
        name: "pull",
        group: Group::Models,
        args: "REPO",
        about: "download a model",
        server: false,
        routed: true,
        words: &[Kind::Text],
        subs: &[],
        flags: &[&[sw("--dev", "LTX-2.5's dev model and its distilled LoRA, for guided videos (51 GB)")]],
        usage: None,
        notes: "REPO is a Hugging Face repo id. Two longer spellings name one file of a repo:\n\
                REPO:QUANT is a GGUF at that quantisation, such as Q4_K_S, and\n\
                REPO:FILE.safetensors is a checkpoint or a LoRA in one file.\n\n\
                The config is read before the weights, so a model this build cannot run is\n\
                refused before gigabytes of it arrive.",
        examples: &[
            "kvad pull HuggingFaceTB/SmolLM2-135M-Instruct",
            "kvad pull city96/FLUX.1-schnell-gguf:Q4_K_S",
        ],
    },
    Command {
        name: "ls",
        group: Group::Models,
        args: "",
        about: "list the models on disk: downloaded, and trained here",
        server: false,
        routed: true,
        words: &[],
        subs: &[],
        flags: &[],
        usage: None,
        notes: "`*` marks the model `kvad use` chose. On a server, `in memory` and `tools` say\n\
                which are loaded and which can be offered tools.",
        examples: &["kvad ls"],
    },
    Command {
        name: "use",
        group: Group::Models,
        args: "MODEL",
        about: "choose the model that `run` and `chat` use when none is named",
        server: false,
        routed: true,
        words: &[Kind::Model(Want::Text)],
        subs: &[],
        flags: &[],
        usage: None,
        notes: "",
        examples: &["kvad use Qwen/Qwen2.5-1.5B-Instruct"],
    },
    Command {
        name: "rm",
        group: Group::Models,
        args: "MODEL",
        about: "delete a downloaded or trained model",
        server: false,
        routed: true,
        words: &[Kind::Model(Want::Any)],
        subs: &[],
        flags: &[&[YES]],
        usage: None,
        notes: "It says what will go, and asks. A downloaded model can be pulled again; one\n\
                trained here is the only copy there is.",
        examples: &["kvad rm stabilityai/sdxl-turbo"],
    },
    Command {
        name: "info",
        group: Group::Models,
        args: "[MODEL]",
        about: "show a model's shape, without downloading its weights",
        server: false,
        routed: true,
        words: &[Kind::Model(Want::Any)],
        subs: &[],
        flags: &[&[f("--model", "MODEL", Kind::Model(Want::Any), "the same as naming it as a word")]],
        usage: None,
        notes: "",
        examples: &["kvad info Qwen/Qwen3-14B"],
    },
    Command {
        name: "cache",
        group: Group::Models,
        args: "[MODEL|clear]",
        about: "list the pre-quantised weight files, or delete them",
        server: false,
        routed: true,
        words: &[Kind::Model(Want::Text)],
        subs: &[Sub { name: "clear", args: "", about: "delete every one of them", words: &[], flags: &[] }],
        flags: &[],
        usage: None,
        notes: "The first load of a model at q8 or q4 writes its quantised weights to a file,\n\
                and later loads map it. Deleting one costs a few seconds on the next load of\n\
                that model and nothing else, which is why this does not ask.\n\n\
                `kvad cache MODEL` deletes that model's files.",
        examples: &["kvad cache", "kvad cache Qwen/Qwen3-14B", "kvad cache clear"],
    },
    Command {
        name: "arch",
        group: Group::Models,
        args: "",
        about: "list the architectures this build can run",
        server: false,
        routed: false,
        words: &[],
        subs: &[],
        flags: &[],
        usage: None,
        notes: "",
        examples: &["kvad arch"],
    },
    // -- talk to one -------------------------------------------------------
    RUN,
    Command {
        name: "chat",
        group: Group::Use,
        args: "",
        about: "a conversation in the terminal",
        server: false,
        routed: true,
        words: &[],
        subs: &[],
        flags: &[
            &[
                TEXT_MODEL,
                f("--system", "TEXT", Kind::Text, "a system prompt for the whole conversation"),
                sw("--save", "on a server: keep the conversation there"),
                f("--conversation", "ID", Kind::Id(Listing::Conversations), "on a server: carry on with one it kept"),
                QUANT,
                BACKEND,
            ],
            SAMPLING,
            ACROSS,
        ],
        usage: None,
        notes: "Inside it: /reset forgets the conversation so far, /quit leaves.",
        examples: &["kvad chat", "kvad chat --system \"Answer in one sentence.\" --save"],
    },
    Command {
        name: "tokenize",
        group: Group::Use,
        args: "TEXT",
        about: "show how a model splits a text into tokens",
        server: true,
        routed: true,
        words: &[Kind::Text],
        subs: &[],
        flags: &[&[TEXT_MODEL]],
        usage: None,
        notes: "",
        examples: &["kvad tokenize \"Hello, world\""],
    },
    // -- make pictures and clips -------------------------------------------
    Command {
        name: "images",
        group: Group::Make,
        args: "[ls|make|edit|rm]",
        about: "make pictures with an image model, and list the ones made",
        server: true,
        routed: true,
        words: &[],
        subs: &[
            Sub { name: "ls", args: "", about: "the pictures made so far", words: &[], flags: &[] },
            Sub { name: "make", args: "PROMPT", about: "draw one from a prompt", words: &[Kind::Text], flags: &[IMAGE_FLAGS] },
            Sub {
                name: "edit",
                args: "PROMPT --image PICTURE",
                about: "draw one over a picture you have",
                words: &[Kind::Text],
                flags: &[IMAGE_FLAGS, &[
                    f("--image", "PICTURE", Kind::File, "the picture to start from"),
                    f("--mask", "FILE", Kind::File, "draw anew only where this is transparent, or white"),
                    f("--strength", "F", Kind::Text, "how far to go from the picture, above 0 and at most 1 (default 0.75)"),
                ]],
            },
            Sub { name: "rm", args: "ID", about: "delete one", words: &[Kind::Id(Listing::Images)], flags: &[&[YES]] },
        ],
        flags: &[],
        usage: Some(super::api::IMAGES),
        notes: "",
        examples: &[
            "kvad images make \"a lighthouse at dusk, photograph\" --out lighthouse.png",
            "kvad images make \"a red fox\" --model stabilityai/sdxl-turbo --size 1024x1024 --seed 3",
            "kvad images edit \"the same street in snow\" --image street.png --strength 0.6",
        ],
    },
    Command {
        name: "videos",
        group: Group::Make,
        args: "[ls|make|show|watch|get|rm]",
        about: "make clips with a video model, and fetch the ones made",
        server: true,
        routed: true,
        words: &[],
        subs: &[
            Sub { name: "ls", args: "", about: "the clips made so far, and the ones being made", words: &[], flags: &[] },
            Sub { name: "make", args: "PROMPT", about: "make one from a prompt, and wait for it", words: &[Kind::Text], flags: &[VIDEO_FLAGS] },
            Sub { name: "show", args: "ID", about: "how far along one is", words: &[Kind::Id(Listing::Videos)], flags: &[] },
            Sub { name: "watch", args: "ID", about: "follow one until it ends", words: &[Kind::Id(Listing::Videos)], flags: &[] },
            Sub {
                name: "get",
                args: "ID",
                about: "fetch one that is done",
                words: &[Kind::Id(Listing::Videos)],
                flags: &[&[f("--out", "FILE", Kind::File, "where to write it (default video-ID.mp4)")]],
            },
            Sub { name: "rm", args: "ID", about: "delete one, stopping it if it is still being made", words: &[Kind::Id(Listing::Videos)], flags: &[&[YES]] },
        ],
        flags: &[],
        usage: Some(super::api::VIDEOS),
        notes: "",
        examples: &[
            "kvad videos make \"waves break against a lighthouse at dusk\" --seconds 3",
            "kvad videos make \"a paper boat in the rain\" --image boat.png --out boat.mp4",
            "kvad videos watch 63",
        ],
    },
    Command {
        name: "transcribe",
        group: Group::Make,
        args: "FILE --model MODEL",
        about: "write down what is said in a recording, with a Whisper",
        server: false,
        routed: true,
        words: &[Kind::File],
        subs: &[],
        flags: &[TRANSCRIBE],
        usage: None,
        notes: "Any Whisper on the Hub in transformers' layout: openai/whisper-*, the distilled\n\
                ones, and fine-tunes such as NbAiLab/nb-whisper-* for Norwegian. A WAV file is\n\
                read here; anything else (MP3, M4A, FLAC, the sound of a video) takes ffmpeg.\n\n\
                NB-Whisper ignores --translate; it was taught to write English when asked for\n\
                --language en, and Nynorsk for --language nn.\n\n\
                Long recordings are read 30 seconds at a time. As text, each stretch is printed\n\
                as soon as it is done.\n\n\
                With a server running, the recording is sent to it, and its Whisper writes it\n\
                down: on its GPU at gpu-q8 where it has one, unless --backend or --quant says\n\
                otherwise. --local runs it in this process, on the CPU, at f32 unless --quant\n\
                says otherwise.\n\n\
                --live - reads raw audio from stdin, 16 kHz mono f32 little-endian, and writes\n\
                the words down while it comes: settled text to stdout, the guess at the rest\n\
                dimmed after it on a terminal. It runs in this process; on the CPU only tiny\n\
                keeps up with speech.",
        examples: &[
            "kvad transcribe intervju.mp3 --model NbAiLab/nb-whisper-small",
            "kvad transcribe talk.wav --model openai/whisper-small --format srt --out talk.srt",
            "kvad transcribe nyheter.m4a --model NbAiLab/nb-whisper-large --backend gpu-q8 --language en",
            "ffmpeg -i talk.mp3 -f f32le -ac 1 -ar 16000 - | kvad transcribe --live - --model NbAiLab/nb-whisper-tiny --quant q8",
        ],
    },
    // -- train your own ----------------------------------------------------
    Command {
        name: "crawl",
        group: Group::Train,
        args: "URL",
        about: "read a documentation site into a text file to train on",
        server: false,
        routed: false,
        words: &[Kind::Text],
        subs: &[],
        flags: &[&[f("--out", "FILE", Kind::File, "where to write it (default: a name from the address)")], CRAWL],
        usage: None,
        notes: "Links are followed under the starting address's directory only, because\n\
                `/book/` links into the standard library's documentation on nearly every page\n\
                and that is a hundred times the book. robots.txt is obeyed. Ctrl-C stops it\n\
                and loses what it has read; the web UI's Stop keeps it.\n\n\
                It always runs here and writes a file. `kvad datasets crawl` is the server's\n\
                version, which makes a dataset there instead.",
        examples: &["kvad crawl https://doc.rust-lang.org/book/", "kvad crawl https://example.org/docs/ --out docs.txt --pages 100"],
    },
    Command {
        name: "train",
        group: Group::Train,
        args: "--data FILE --name NAME",
        about: "train a small language model of your own on a text file",
        server: false,
        routed: true,
        words: &[],
        subs: &[Sub { name: "options", args: "", about: "on a server: the sizes and defaults a run there can have", words: &[], flags: &[] }],
        flags: &[&[
            f("--data", "FILE", Kind::File, "plain text to learn from; a server is sent it as a dataset first"),
            f("--dataset", "ID|NAME", Kind::Id(Listing::Datasets), "on a server: train on a dataset it already has"),
            f("--name", "NAME", Kind::Text, "what to call the model; one word, no `/`"),
            f("--from", "MODEL", Kind::Model(Want::Trained), "train an existing model further, instead of a new one"),
            f("--size", "NAME", Kind::TrainSize, "the model's shape: small, medium or large (default small)"),
            f("--steps", "N", Kind::Text, "training steps (default 2000)"),
            f("--batch", "N", Kind::Text, "windows of text a step (default 16); here only"),
            f("--lr", "F", Kind::Text, "peak learning rate (default 0.003)"),
            f("--warmup", "N", Kind::Text, "steps spent climbing to it (default: a tenth of the run); here only"),
            f("--decay-to", "F", Kind::Text, "fraction of --lr left at the last step (default 0.1); here only"),
            f("--clip", "F", Kind::Text, "longest the whole gradient may be, 0 for no limit (default 1); here only"),
            f("--eval-every", "N", Kind::Text, "steps between checkpoints (default 250)"),
            f("--threads", "N", Kind::Text, "replicas to split each batch across (default: every core)"),
            f("--sample", "N", Kind::Text, "characters to write at each checkpoint, 0 for none (default 160)"),
            f("--seed", "N", Kind::Text, "the same seed trains the same model (default 7)"),
        ]],
        usage: Some(
            "usage: kvad train --data FILE --name NAME [options]\n       \
             kvad train --data FILE --from MODEL [--name NAME]   train an existing one further\n       \
             kvad train options                                  what a server's runs can be asked for",
        ),
        notes: "The model that scored best on text it was not trained on is the one kept, not\n\
                the last one.\n\n\
                `--from` has two limits worth knowing. The character vocabulary is fixed at\n\
                first training, so text with a character the model never saw is refused. And\n\
                the optimiser's state is not saved, so a resumed run restarts AdamW's running\n\
                averages: measured at up to 0.06 of training loss over 50 steps, gone by 100.",
        examples: &[
            "kvad train --data book.txt --name book",
            "kvad train --data book.txt --from book --steps 1000",
            "kvad run --model book --prompt \"Chapter 1\"",
        ],
    },
    Command {
        name: "tune",
        group: Group::Train,
        args: "--data DIR --name NAME",
        about: "train a LoRA for SDXL from a folder of pictures",
        server: true,
        routed: true,
        words: &[],
        subs: &[Sub { name: "options", args: "", about: "what a run can be asked for on that server", words: &[], flags: &[] }],
        flags: &[&[
            f("--data", "DIR", Kind::Dir, "a folder of pictures and their captions, sent as a dataset first"),
            f("--dataset", "ID|NAME", Kind::Id(Listing::Datasets), "a dataset of pictures the server already has"),
            f("--name", "NAME", Kind::Text, "what to call the LoRA"),
            f("--model", "MODEL", Kind::Model(Want::Image), "the model it is for (default: SDXL's base)"),
            f("--caption", "TEXT", Kind::Text, "the caption of every picture that has none"),
            f(
                "--size",
                "N",
                Kind::Choice(&[("512", "pixels a side"), ("768", "pixels a side"), ("1024", "pixels a side, the default")]),
                "pixels a side to train at: 512, 768 or 1024 (default 1024)",
            ),
            f("--rank", "N", Kind::Text, "the LoRA's rank (default 16)"),
            f("--steps", "N", Kind::Text, "training steps (default 1000)"),
            f("--lr", "F", Kind::Text, "learning rate (default 1e-4)"),
            f("--eval-every", "N", Kind::Text, "steps between measurements (default 100)"),
            f("--seed", "N", Kind::Text, "(default 1337)"),
            f("--sample", "TEXT", Kind::Text, "a prompt to draw before the first step and at every measurement; up to four"),
            f("--sample-size", "N", Kind::Text, "pixels a side of a sample (default 512)"),
            f("--sample-steps", "N", Kind::Text, "a sample's denoising steps (default 20)"),
        ]],
        usage: Some(super::api::TUNE),
        notes: "",
        examples: &[
            "kvad tune --data ./my-dog --name my-dog --caption \"a photo of sks dog\"",
            "kvad images make \"sks dog on the moon\" --lora my-dog",
        ],
    },
    Command {
        name: "datasets",
        group: Group::Train,
        args: "[ls|add|crawl|show|...]",
        about: "the texts and picture folders a server trains on",
        server: true,
        routed: true,
        words: &[],
        subs: &[
            Sub { name: "ls", args: "", about: "the datasets the server has", words: &[], flags: &[] },
            Sub {
                name: "add",
                args: "FILE|DIR",
                about: "send a text, or a folder of pictures",
                words: &[Kind::File],
                flags: &[&[f("--name", "NAME", Kind::Text, "what to call it (default: the file's or folder's name)")]],
            },
            Sub {
                name: "crawl",
                args: "URL --name NAME",
                about: "have the server read a documentation site into one",
                words: &[Kind::Text],
                flags: &[&[
                    f("--name", "NAME", Kind::Text, "what to call it"),
                    f("--pages", "N", Kind::Text, "most pages to read (default 400)"),
                    f("--mb", "F", Kind::Text, "most megabytes of text to collect (default 16)"),
                    f("--pause", "MS", Kind::Text, "milliseconds to wait between requests (default 250)"),
                    f("--drop-rare", "N", Kind::Text, "drop characters seen fewer than N times (default 10)"),
                    sw("--same-host", "follow links anywhere on the host"),
                ]],
            },
            Sub { name: "show", args: "ID", about: "what is in one", words: &[Kind::Id(Listing::Datasets)], flags: &[] },
            Sub {
                name: "check",
                args: "ID --model MODEL",
                about: "would its characters fit that model's vocabulary",
                words: &[Kind::Id(Listing::Datasets)],
                flags: &[&[f("--model", "MODEL", Kind::Model(Want::Trained), "the model that would be trained further on it")]],
            },
            Sub {
                name: "search",
                args: "ID QUESTION",
                about: "the passages of one nearest a question",
                words: &[Kind::Id(Listing::Datasets), Kind::Text],
                flags: &[&[f("--k", "N", Kind::Text, "how many passages")]],
            },
            Sub { name: "put", args: "ID FILE...", about: "add pictures and captions to a folder of them", words: &[Kind::Id(Listing::Datasets), Kind::File], flags: &[] },
            Sub {
                name: "get",
                args: "ID FILE",
                about: "fetch one file of it back",
                words: &[Kind::Id(Listing::Datasets), Kind::Text],
                flags: &[&[f("--out", "PATH", Kind::File, "where to write it")]],
            },
            Sub { name: "rm", args: "ID [FILE...]", about: "delete one, or those files of it", words: &[Kind::Id(Listing::Datasets), Kind::Text], flags: &[&[YES]] },
        ],
        flags: &[],
        usage: Some(super::api::DATASETS),
        notes: "",
        examples: &["kvad datasets add book.txt", "kvad datasets add ./my-dog --name my-dog", "kvad train --dataset book --name book"],
    },
    // -- the server --------------------------------------------------------
    SERVE_HANDOVER,
    Command {
        name: "service",
        group: Group::Server,
        args: "[status|start|stop|...]",
        about: "that server as a service that starts at login",
        server: false,
        routed: false,
        words: &[],
        subs: &[
            Sub { name: "status", args: "", about: "is it running, where is it listening, what is loaded", words: &[], flags: &[&[sw("--json", "its health document, as JSON")]] },
            Sub { name: "start", args: "", about: "start it, and wait for it to answer", words: &[], flags: &[] },
            Sub { name: "stop", args: "", about: "stop it until the next login", words: &[], flags: &[] },
            Sub { name: "restart", args: "", about: "stop it and start it", words: &[], flags: &[] },
            Sub {
                name: "logs",
                args: "",
                about: "what the server has written",
                words: &[],
                flags: &[&[
                    Flag { name: "--follow", short: "-f", value: "", kind: Kind::Text, about: "keep printing as it writes" },
                    Flag { name: "--lines", short: "-n", value: "N", kind: Kind::Text, about: "how many lines back to start" },
                ]],
            },
            Sub {
                name: "install",
                args: "",
                about: "run kvad-serve at login",
                words: &[],
                flags: &[&[
                    f("--host", "H", Kind::Text, "the address to listen on (default: the one it has, else 127.0.0.1)"),
                    f("--port", "P", Kind::Text, "the port (default: the one it has, else 5823)"),
                    sw("--force", "install over the objections: no accounts beyond this machine, a taken address"),
                ]],
            },
            Sub { name: "uninstall", args: "", about: "stop it, and stop it starting at login", words: &[], flags: &[] },
        ],
        flags: &[],
        usage: Some(super::service::USAGE),
        notes: "",
        examples: &["kvad service install", "kvad service status", "kvad service logs -f"],
    },
    Command {
        name: "ps",
        group: Group::Server,
        args: "",
        about: "the models in memory, and how much room is left",
        server: true,
        routed: true,
        words: &[],
        subs: &[],
        flags: &[],
        usage: None,
        notes: "",
        examples: &["kvad ps"],
    },
    Command {
        name: "load",
        group: Group::Server,
        args: "MODEL",
        about: "put a model in memory, downloading it first if need be",
        server: true,
        routed: true,
        words: &[Kind::Model(Want::Any)],
        subs: &[],
        flags: &[&[BACKEND, QUANT]],
        usage: None,
        notes: "A model that does not fit beside the ones already in memory is refused; nothing\n\
                is unloaded to make room. `kvad ps` shows what is there and `kvad unload` gives\n\
                some back.",
        examples: &["kvad load Qwen/Qwen3-14B", "kvad load Qwen/Qwen3-14B --backend gpu-q4"],
    },
    Command {
        name: "unload",
        group: Group::Server,
        args: "[ID]",
        about: "take a model out of memory, or all of them",
        server: true,
        routed: true,
        words: &[Kind::Resident],
        subs: &[],
        flags: &[],
        usage: None,
        notes: "ID is as `kvad ps` shows it: the model and the backend it is on.",
        examples: &["kvad unload", "kvad unload Qwen/Qwen3-14B@gpu-q8"],
    },
    Command {
        name: "cancel",
        group: Group::Server,
        args: "[ID]",
        about: "stop what you have generating, or one generation",
        server: true,
        routed: true,
        words: &[Kind::Text],
        subs: &[],
        flags: &[],
        usage: None,
        notes: "Stops your own generations, waiting or running, and nobody else's. ID is as\n\
                `kvad queue` shows it.",
        examples: &["kvad cancel", "kvad cancel 12"],
    },
    Command {
        name: "queue",
        group: Group::Server,
        args: "",
        about: "what is generating, and what is waiting to",
        server: true,
        routed: true,
        words: &[],
        subs: &[],
        flags: &[],
        usage: None,
        notes: "Your own generations, and how many there are altogether. An administrator is\n\
                shown everybody's. Accounts take turns, so the order is not arrival order.\n\
                When as many are waiting as the server allows, or as one account may have,\n\
                the next request is refused and told when to try again.",
        examples: &["kvad queue"],
    },
    Command {
        name: "jobs",
        group: Group::Server,
        args: "[ls|show|watch|cancel|pictures]",
        about: "downloads, training runs, evals and benchmarks, in one history",
        server: true,
        routed: true,
        words: &[],
        subs: &[
            Sub { name: "ls", args: "", about: "the most recent", words: &[], flags: &[&[f("--limit", "N", Kind::Text, "how many (default 50)")]] },
            Sub { name: "show", args: "ID", about: "what one did, and its measurements", words: &[Kind::Id(Listing::Jobs)], flags: &[] },
            Sub { name: "watch", args: "ID", about: "follow one until it ends", words: &[Kind::Id(Listing::Jobs)], flags: &[] },
            Sub { name: "cancel", args: "ID", about: "ask one to stop", words: &[Kind::Id(Listing::Jobs)], flags: &[] },
            Sub {
                name: "pictures",
                args: "ID",
                about: "fetch the samples a LoRA run drew",
                words: &[Kind::Id(Listing::Jobs)],
                flags: &[&[f("--out", "DIR", Kind::Dir, "where to put them (default: here)")]],
            },
        ],
        flags: &[],
        usage: Some(super::api::JOBS),
        notes: "",
        examples: &["kvad jobs", "kvad jobs watch 16"],
    },
    Command {
        name: "conversations",
        group: Group::Server,
        args: "[ls|show|edit|rm]",
        about: "the conversations a server has kept",
        server: true,
        routed: true,
        words: &[],
        subs: &[
            Sub { name: "ls", args: "", about: "all of them", words: &[], flags: &[] },
            Sub { name: "show", args: "ID", about: "read one", words: &[Kind::Id(Listing::Conversations)], flags: &[] },
            Sub {
                name: "edit",
                args: "ID",
                about: "change its title or its system prompt",
                words: &[Kind::Id(Listing::Conversations)],
                flags: &[&[f("--title", "T", Kind::Text, "a new title"), f("--system", "S", Kind::Text, "a new system prompt")]],
            },
            Sub { name: "rm", args: "ID", about: "delete one", words: &[Kind::Id(Listing::Conversations)], flags: &[&[YES]] },
        ],
        flags: &[],
        usage: Some(super::api::CONVERSATIONS),
        notes: "",
        examples: &["kvad conversations", "kvad chat --conversation 14"],
    },
    // -- measure -----------------------------------------------------------
    Command {
        name: "evals",
        group: Group::Measure,
        args: "[runs|show|suites|add|run|...]",
        about: "score models against a suite of prompts, or by perplexity",
        server: true,
        routed: true,
        words: &[],
        subs: &[
            Sub { name: "runs", args: "", about: "the runs so far", words: &[], flags: &[] },
            Sub { name: "show", args: "ID", about: "one run's scores", words: &[Kind::Id(Listing::EvalRuns)], flags: &[] },
            Sub { name: "suites", args: "", about: "the suites the server has", words: &[], flags: &[] },
            Sub { name: "add", args: "FILE", about: "add a suite from a JSON file", words: &[Kind::File], flags: &[&[f("--name", "NAME", Kind::Text, "call it this, whatever the file says")]] },
            Sub { name: "edit", args: "ID FILE", about: "replace a suite with a file", words: &[Kind::Id(Listing::Suites), Kind::File], flags: &[] },
            Sub { name: "rm", args: "ID", about: "delete a suite", words: &[Kind::Id(Listing::Suites)], flags: &[&[YES]] },
            Sub {
                name: "run",
                args: "SUITE MODEL[@BACKEND]...",
                about: "run a suite against one model or several",
                words: &[Kind::Id(Listing::Suites), Kind::Variant],
                flags: &[&[
                    f("--max-tokens", "N", Kind::Text, "most tokens an answer may be (default 256)"),
                    f("--seed", "N", Kind::Text, "sampling seed (default 7)"),
                    BACKEND,
                ]],
            },
            Sub {
                name: "perplexity",
                args: "DATASET MODEL[@BACKEND]...",
                about: "how surprised each model is by a dataset's text",
                words: &[Kind::Id(Listing::Datasets), Kind::Variant],
                flags: &[&[f("--window", "N", Kind::Text, "tokens read at a time"), BACKEND]],
            },
        ],
        flags: &[],
        usage: Some(super::api::EVALS),
        notes: "",
        examples: &["kvad evals add capitals.json", "kvad evals run capitals Qwen/Qwen3-14B@gpu-q8 Qwen/Qwen3-14B@gpu-q4"],
    },
    Command {
        name: "bench",
        group: Group::Measure,
        args: "[runs|show|run]",
        about: "time models against each other, in tokens a second",
        server: true,
        routed: true,
        words: &[],
        subs: &[
            Sub { name: "runs", args: "", about: "the runs so far", words: &[], flags: &[] },
            Sub { name: "show", args: "ID", about: "one run's numbers", words: &[Kind::Id(Listing::BenchRuns)], flags: &[] },
            Sub {
                name: "run",
                args: "MODEL[@BACKEND]...",
                about: "time one model or several",
                words: &[Kind::Variant],
                flags: &[&[
                    f("--prompt", "T", Kind::Text, "what every model continues"),
                    f("--rounds", "N", Kind::Text, "times to visit every model"),
                    f("--tokens", "N", Kind::Text, "tokens to write each visit"),
                    f("--seed", "N", Kind::Text, "sampling seed"),
                    BACKEND,
                ]],
            },
        ],
        flags: &[],
        usage: Some(super::api::BENCH),
        notes: "",
        examples: &["kvad bench run Qwen/Qwen3-14B@cpu-q8 Qwen/Qwen3-14B@gpu-q8 --rounds 3"],
    },
    Command {
        name: "metrics",
        group: Group::Measure,
        args: "[requests|log|usage]",
        about: "what the server is doing, its requests, and who used it",
        server: true,
        routed: true,
        words: &[],
        subs: &[
            Sub { name: "overview", args: "", about: "memory, load and the queue", words: &[], flags: &[] },
            Sub { name: "requests", args: "", about: "recent requests, and what each route costs", words: &[], flags: &[&[f("--limit", "N", Kind::Text, "how many")]] },
            Sub { name: "log", args: "", about: "the end of the server's log", words: &[], flags: &[&[f("--limit", "N", Kind::Text, "how many lines")]] },
            Sub { name: "usage", args: "", about: "tokens and waiting, by account", words: &[], flags: &[&[f("--days", "N", Kind::Text, "how far back (default 7)")]] },
        ],
        flags: &[],
        usage: Some(super::api::METRICS),
        notes: "",
        examples: &["kvad metrics", "kvad metrics requests --limit 20", "kvad metrics usage --days 30"],
    },
    Command {
        name: "cluster",
        group: Group::Measure,
        args: "[find|plan|status|listen|ping|link]",
        about: "measure the link between two machines, and say what it is",
        server: false,
        routed: false,
        words: &[Kind::Text],
        subs: &[
            Sub { name: "find", args: "", about: "who is on the other end of each cable", words: &[], flags: &[&[sw("--all", "ask Wi-Fi and the LAN too, not only cables")]] },
            Sub { name: "listen", args: "", about: "be the far end of a measurement", words: &[], flags: &[&[CLUSTER_PORT]] },
            Sub {
                name: "ping",
                args: "[HOST]",
                about: "measure the link to a machine that is listening",
                words: &[Kind::Text],
                flags: &[&[
                    CLUSTER_PORT,
                    f("--rounds", "N", Kind::Text, "rounds that count, after one that does not (default 7)"),
                    f("--sizes", "LIST", Kind::Text, "frame sizes to send, such as 16k,32k,2m"),
                    f("--stream-mb", "N", Kind::Text, "megabytes sent one way, for the ceiling (default 64)"),
                    f("--gap", "MS", Kind::Text, "leave the link idle this long before each frame, as a model's tokens do"),
                ]],
            },
            Sub { name: "link", args: "IFACE", about: "say what one interface's link is", words: &[Kind::Text], flags: &[] },
            Sub { name: "status", args: "", about: "a server's part in a cluster: whether it is a worker, and whose", words: &[], flags: &[] },
            Sub {
                name: "plan",
                args: "MODEL",
                about: "say which machine would hold which layers of a model, and load nothing",
                words: &[Kind::Model(Want::Text)],
                flags: &[&[QUANT], ACROSS],
            },
        ],
        flags: &[],
        usage: Some(CLUSTER),
        notes: "The first step towards one model across several machines; docs/cluster-plan.md\n\
                is the whole of it. What is here measures a link the way a split model will\n\
                use it: a frame out and the same frame back, which is what handing a machine\n\
                its layers' input and waiting for their output costs before any layer runs.\n\
                32 KB is a decode step of a 70B model and 2 MB a chunk of its prompt.\n\n\
                It says what the link is before it prints a number. A USB-C charging cable\n\
                between two Macs carries USB 2: it takes addresses, answers and carries SSH,\n\
                at a hundredth of what the ports can do, and looks like Thunderbolt to\n\
                everything but this.\n\n\
                `listen` sends back what it is sent to anybody who can reach its port. It\n\
                loads nothing and reads nothing, and is not meant to be left running.",
        examples: &["kvad cluster listen", "kvad cluster find", "kvad cluster ping", "kvad cluster ping 'fe80::1c%en12' --sizes 16k,32k,64k"],
    },
    Command {
        name: "worker",
        group: Group::Server,
        args: "",
        about: "hold some of a model's layers for another machine that runs it",
        server: false,
        routed: false,
        words: &[],
        subs: &[],
        flags: &[&[
            f("--host", "ADDR", Kind::Text, "the address to listen on (default: every one this machine has)"),
            f("--port", "N", Kind::Text, "the port to listen on (default 7420)"),
        ]],
        usage: None,
        notes: "A model too large for one machine can be run across several: its layers are\n\
                divided between them, and what passes from one to the next is the model's\n\
                working state for each token, a few kilobytes. This is the machine that\n\
                holds layers and nothing else. The one that has the prompt names it:\n\n\
                \x20 kvad run --model MODEL --workers THIS-MACHINE\n\n\
                and the layers are placed: as many as fit on the fastest machine, then the\n\
                next, and a machine the model has no need of is left out. That is the\n\
                placing that writes a reply quickest. A prompt is another matter: it is\n\
                read by every machine at once, and quickest with the work shared out, so\n\
                --place prompt gives a slower machine some layers where --place reply\n\
                gives it none, and --place 2000:200 places for that much reading to that\n\
                much writing. `kvad cluster\n\
                plan MODEL --workers ...` shows the placing and loads nothing. --split\n\
                0..12,12..24 places them by hand instead: 0 to 11 on the machine with the\n\
                prompt, 12 to 23 here. A worker fetches the files its own layers are in,\n\
                if it has not got them, and no others.\n\n\
                THIS-MACHINE is the name the worker prints when it starts, the id beside\n\
                it, or an address. By name or id it is found on the network, every way to\n\
                it is measured, and the quickest is used and said: a machine on a cable\n\
                and on Wi-Fi is two ways, and they are not the same speed. `kvad cluster\n\
                find` lists the workers there are.\n\n\
                It answers only a machine that holds the same cluster key. The first start\n\
                makes one, keeps it in the data directory as cluster.key and prints it;\n\
                copy that file to the other machines, or give them the key in\n\
                KVAD_CLUSTER_KEY. What crosses between machines is not encrypted.\n\n\
                Both ends have to be the same build of kvad, to the commit. Finding a\n\
                worker by name is written for macOS; elsewhere, give its address.",
        examples: &["kvad worker", "kvad worker --port 7421"],
    },
    Command {
        name: "remote",
        group: Group::Accounts,
        args: "[id|status|pair|join]",
        about: "pair devices with a server, and reach it from anywhere",
        server: false,
        routed: false,
        words: &[Kind::Text],
        subs: &[
            Sub { name: "id", args: "", about: "the id a server's [remote] allow lists for this machine", words: &[], flags: &[] },
            Sub { name: "status", args: "", about: "which path the tunnel to a kvad:// server is on", words: &[], flags: &[] },
            Sub {
                name: "pair",
                args: "",
                about: "on the server: a link and QR code that pair one device",
                words: &[],
                flags: &[&[sw("--admin", "let the device act as an administrator")]],
            },
            Sub {
                name: "join",
                args: "LINK",
                about: "on the device: pair with a link from `kvad remote pair`",
                words: &[Kind::Text],
                flags: &[&[f("--name", "NAME", Kind::Text, "what the server calls this device (default: hostname)")]],
            },
        ],
        flags: &[],
        usage: None,
        notes: "A server with [remote] in its kvad.toml prints a kvad:// address when it starts. On it, \
                `kvad remote pair` shows a link and a QR code that pair one device, once, within ten minutes; \
                on the device, `kvad remote join LINK` pairs and keeps a key that works from that device \
                alone. Every command then takes --remote kvad://<node id>.",
        examples: &["kvad remote pair", "kvad remote join 'kvad://…&pair=…'", "kvad remote status --remote kvad://…"],
    },
    Command {
        name: "tunnel",
        group: Group::Accounts,
        args: "",
        about: "a local port for any HTTP client, forwarded to a kvad:// server",
        server: false,
        routed: false,
        words: &[],
        subs: &[],
        flags: &[&[
            f("--remote", "kvad://ID", Kind::Text, "the paired server to forward to (default: the only one this device is paired with)"),
            f("--port", "N", Kind::Text, "the port to listen on, on 127.0.0.1 only (default 5824)"),
            sw("--with-key", "add this device's key to every request: anything that reaches the port acts as this device"),
        ]],
        usage: None,
        notes: "Listens on 127.0.0.1 and forwards each connection through the tunnel, so\n\
                that OpenCode, curl or the openai SDK work against\n\
                http://127.0.0.1:5824/v1 unmodified. Says which path the tunnel is on (lan,\n\
                direct or relay) when it starts and whenever that changes, and connects\n\
                again when it drops.\n\n\
                The port is open to every program on this machine and to every user of it,\n\
                and they all reach the server as this device. The server still wants a key\n\
                with each request, so by default each program sends one of its own\n\
                (`kvad keys add NAME`), which can be revoked on its own.\n\n\
                --with-key adds this device's key instead, so nothing else needs one, and\n\
                anything that can reach the port acts as this device, with its rights,\n\
                while it runs. Use it on a machine that is yours alone. With it on,\n\
                requests a browser sent from a web page are refused, and each request gets\n\
                a connection of its own.",
        examples: &["kvad tunnel --remote kvad://9ce63fe0…", "kvad tunnel --port 8080 --with-key"],
    },
    Command {
        name: "devices",
        group: Group::Accounts,
        args: "[ls|rm ID]",
        about: "the devices paired with a server, and unpairing one",
        server: true,
        routed: true,
        words: &[Kind::Text],
        subs: &[
            Sub { name: "ls", args: "", about: "your paired devices", words: &[], flags: &[&[sw("--all", "every account's, for an administrator")]] },
            Sub { name: "rm", args: "ID", about: "unpair one: its key is revoked and its connection closed", words: &[Kind::Text], flags: &[] },
        ],
        flags: &[],
        usage: None,
        notes: "",
        examples: &["kvad devices", "kvad devices rm 3"],
    },
    // -- accounts ----------------------------------------------------------
    Command {
        name: "auth",
        group: Group::Accounts,
        args: "[status|login|logout|...]",
        about: "sign in to a server, and keep a key for it",
        server: true,
        routed: true,
        words: &[],
        subs: &[
            Sub { name: "status", args: "", about: "whether the server has accounts, and who you are to it", words: &[], flags: &[] },
            Sub {
                name: "login",
                args: "",
                about: "sign in, and keep an API key for this server",
                words: &[],
                flags: &[&[
                    f("--name", "NAME", Kind::Text, "your account's name, rather than be asked"),
                    sw("--key", "paste a key made elsewhere, rather than sign in"),
                ]],
            },
            Sub { name: "logout", args: "", about: "revoke the kept key, and forget it", words: &[], flags: &[] },
            Sub { name: "setup", args: "TOKEN", about: "make the first account, with the token the server printed", words: &[Kind::Text], flags: &[&[f("--name", "NAME", Kind::Text, "the account's name, rather than be asked")]] },
            Sub { name: "password", args: "", about: "change your password", words: &[], flags: &[] },
        ],
        flags: &[],
        usage: Some(super::api::AUTH),
        notes: "",
        examples: &["kvad auth login --remote http://box:5823", "kvad auth status"],
    },
    Command {
        name: "users",
        group: Group::Accounts,
        args: "[ls|add|edit|rm]",
        about: "the accounts on a server",
        server: true,
        routed: true,
        words: &[],
        subs: &[
            Sub { name: "ls", args: "", about: "all of them", words: &[], flags: &[] },
            Sub { name: "add", args: "NAME", about: "make one; the password is asked for", words: &[Kind::Text], flags: &[&[f("--role", "admin|user", ROLES, "what it may do (default user)")]] },
            Sub {
                name: "edit",
                args: "ID",
                about: "change one's role or password",
                words: &[Kind::Id(Listing::Users)],
                flags: &[&[f("--role", "admin|user", ROLES, "what it may do"), sw("--password", "set a new password, which is asked for")]],
            },
            Sub { name: "rm", args: "ID", about: "delete one", words: &[Kind::Id(Listing::Users)], flags: &[&[YES]] },
        ],
        flags: &[],
        usage: Some(super::api::USERS),
        notes: "",
        examples: &["kvad users add ada --role admin"],
    },
    Command {
        name: "sessions",
        group: Group::Accounts,
        args: "[ls|rm]",
        about: "where you are signed in from a browser",
        server: true,
        routed: true,
        words: &[],
        subs: &[
            Sub { name: "ls", args: "", about: "all of them", words: &[], flags: &[] },
            Sub { name: "rm", args: "HASH", about: "sign one out", words: &[Kind::Id(Listing::Sessions)], flags: &[] },
        ],
        flags: &[],
        usage: Some(super::api::SESSIONS),
        notes: "",
        examples: &["kvad sessions"],
    },
    Command {
        name: "keys",
        group: Group::Accounts,
        args: "[ls|add|rm]",
        about: "API keys, for programs that call the server",
        server: true,
        routed: true,
        words: &[],
        subs: &[
            Sub { name: "ls", args: "", about: "all of yours", words: &[], flags: &[] },
            Sub { name: "add", args: "NAME", about: "make one; it is shown once", words: &[Kind::Text], flags: &[] },
            Sub { name: "rm", args: "ID", about: "revoke one", words: &[Kind::Id(Listing::Keys)], flags: &[] },
        ],
        flags: &[],
        usage: Some(super::api::KEYS),
        notes: "",
        examples: &["kvad keys add ci"],
    },
    // -- this program ------------------------------------------------------
    Command {
        name: "api",
        group: Group::Cli,
        args: "[METHOD] PATH [BODY]",
        about: "call any route of the server's API; with no words, list them",
        server: true,
        routed: true,
        words: &[Kind::Route, Kind::Route, Kind::Text],
        subs: &[],
        flags: &[],
        usage: Some(super::api::API),
        notes: "",
        examples: &["kvad api", "kvad api /api/health", "kvad api post /api/keys '{\"name\": \"ci\"}'"],
    },
    Command {
        name: "completions",
        group: Group::Cli,
        args: "[install|uninstall|status|SHELL]",
        about: "set up Tab completion for your shell",
        server: false,
        routed: false,
        words: &[SHELLS],
        subs: &[
            Sub { name: "install", args: "[SHELL...]", about: "set it up, for your login shell or the ones named", words: &[SHELLS], flags: &[] },
            Sub { name: "uninstall", args: "[SHELL...]", about: "take it out, of every shell or the ones named", words: &[SHELLS], flags: &[] },
            Sub { name: "status", args: "", about: "where it is set up, and where it would go", words: &[], flags: &[] },
        ],
        flags: &[],
        usage: None,
        notes: "`install` adds one line to your shell's rc file — ~/.zshrc, or ~/.bashrc\n\
                (~/.bash_profile on a Mac) — or, for fish, writes a file among its completions.\n\
                The line asks kvad for the script each time a shell starts, so an upgrade needs\n\
                nothing done again. `uninstall` removes exactly what `install` wrote.\n\n\
                With a shell's name alone, this prints that shell's script.\n\n\
                Once it is set up, Tab completes commands, their options, and the things only\n\
                this machine or its server knows: the models on disk and in memory, backends,\n\
                LoRAs, and the ids of jobs, conversations, pictures, clips and datasets.\n\n\
                It completes kvad-serve, kvad-gpu and kvad-tui as well.",
        examples: &["kvad completions install", "kvad completions status", "kvad completions zsh"],
    },
    Command {
        name: "help",
        group: Group::Cli,
        args: "[COMMAND]",
        about: "this list, or what one command takes",
        server: false,
        routed: false,
        words: &[Kind::Command],
        subs: &[],
        flags: &[],
        usage: None,
        notes: "`kvad COMMAND --help` says the same as `kvad help COMMAND`.",
        examples: &["kvad help", "kvad help videos"],
    },
];

pub const SHELLS: Kind = Kind::Choice(&[("zsh", ""), ("bash", ""), ("fish", "")]);

/// `kvad run`, and what `kvad --prompt TEXT` means with no command.
const RUN: Command = Command {
    name: "run",
    group: Group::Use,
    args: "",
    about: "answer one prompt and exit",
    server: false,
    routed: true,
    words: &[],
    subs: &[],
    flags: &[
        &[
            TEXT_MODEL,
            f("--prompt", "TEXT", Kind::Text, "what to answer (default: a prompt that shows the model works)"),
            f("--system", "TEXT", Kind::Text, "a system prompt before it"),
            sw("--raw", "on a server: continue the prompt as text, with no chat template"),
            QUANT,
            BACKEND,
        ],
        SAMPLING,
        ACROSS,
    ],
    usage: None,
    notes: "`kvad --prompt TEXT`, with no command, is `kvad run --prompt TEXT`.\n\n\
            The answer goes to stdout and everything else to stderr, so\n\
            `kvad run --prompt ... > answer.txt` keeps only the answer.\n\n\
            --workers and --split run one model across several machines, in this\n\
            process and theirs: see `kvad help worker`.",
    examples: &[
        "kvad run --prompt \"Why is the sky blue?\"",
        "kvad run --model Qwen/Qwen2.5-1.5B-Instruct --prompt \"...\" --greedy",
        "kvad run --local --quant q8 --prompt \"...\"",
    ],
};

/// `kvad serve`, and `kvad-serve` itself: the first hands every word after
/// it to the second.
const SERVE_HANDOVER: Command = Command {
    name: "serve",
    group: Group::Server,
    args: "",
    about: "run the HTTP server and web UI in this terminal",
    server: false,
    routed: false,
    words: &[],
    subs: &[],
    flags: &[SERVE_FLAGS],
    usage: None,
    notes: "This hands over to `kvad-serve`, a separate program found beside this one, and\n\
            everything after `serve` is its to read: `kvad serve --help` is its own help.\n\n\
            To keep it running after the terminal closes, and start it at login, use\n\
            `kvad service install` instead.",
    examples: &["kvad serve", "kvad serve --bind 127.0.0.1:8080"],
};

const IMAGE_FLAGS: &[Flag] = &[
    f("--out", "FILE", Kind::File, "where to write it (default image-ID.png)"),
    f("--model", "MODEL", Kind::Model(Want::Image), "the image model (default: the one in memory, else the server's)"),
    f("--size", "WxH", Kind::Text, "width and height in pixels, such as 1024x1024"),
    f("--steps", "N", Kind::Text, "denoising steps"),
    f("--guidance", "F", Kind::Text, "how hard to follow the prompt"),
    f("--negative", "TEXT", Kind::Text, "what to keep out of it"),
    f("--seed", "N", Kind::Text, "the same seed draws the same picture (default: a new one each time)"),
    LORA,
];

const VIDEO_FLAGS: &[Flag] = &[
    f("--out", "FILE", Kind::File, "where to write it (default video-ID.mp4)"),
    f("--model", "MODEL", Kind::Model(Want::Video), "the video model"),
    f("--size", "WxH", Kind::Text, "width and height in pixels, such as 768x512"),
    f("--seconds", "S", Kind::Text, "how long (default: LTX-2.5 chooses from the prompt)"),
    f("--frames", "N", Kind::Text, "or how many frames"),
    f("--fps", "N", Kind::Text, "frames a second; above 30 runs DFR"),
    f("--seed", "N", Kind::Text, "the same seed makes the same clip (default: a new one each time)"),
    sw("--silent", "no sound"),
    f("--image", "PICTURE", Kind::File, "a picture to be its first frame"),
    f("--steps", "N", Kind::Text, "denoising steps; runs the dev model, guided (default 30)"),
    f("--guidance", "G", Kind::Text, "how hard to follow the prompt; runs the dev model (default 3)"),
    f("--negative", "TEXT", Kind::Text, "what to keep out of it; runs the dev model"),
    f(
        "--decoder",
        "diffusion|conv",
        Kind::Choice(&[("diffusion", "the reference's, and the default"), ("conv", "lighter, about twice as fast to decode")]),
        "how latents become frames",
    ),
    f(
        "--pipeline",
        "fast|dfr",
        Kind::Choice(&[("fast", "the distilled model, and the default"), ("dfr", "keyframes, a detailing pass, a keyframe-aware decode: slower and finer")]),
        "which of LTX-2.5's unguided pipelines",
    ),
    sw("--epilogue", "end DFR with its spatial epilogue, for sizes its second stage cannot hold whole"),
    LORA,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `"--flag"` the parser's source matches on.
    ///
    /// Read out of `main.rs` the way `cli.rs` reads routes out of its
    /// modules: the text between `fn switch` and the end of `parse_from` is
    /// where a flag is something the program takes, and a string literal
    /// there that starts with a dash is one.
    fn parsed() -> Vec<String> {
        let source = include_str!("../main.rs");
        let from = source.find("fn switch(").expect("the parser's switches");
        let to = source.find("/// Which model to use").expect("the end of the parser");
        let mut found = Vec::new();
        for piece in source[from..to].split('"').skip(1).step_by(2) {
            let dashed = piece.len() > 2 && piece.starts_with("--") || piece.len() == 2 && piece.starts_with('-') && piece != "--";
            if dashed && piece.chars().all(|c| c == '-' || c.is_ascii_alphanumeric()) && !found.iter().any(|f| f == piece) {
                found.push(piece.to_string());
            }
        }
        found
    }

    /// `kvad serve`'s flags are `kvad-serve`'s, and this parser never sees
    /// them.
    const PASSED_THROUGH: &[&str] = &["--bind", "--config", "--db", "--insecure", "--version"];

    #[test]
    fn every_flag_the_parser_takes_is_in_the_table() {
        let parsed = parsed();
        assert!(parsed.len() > 60, "only found {} flags; the scanner is broken", parsed.len());
        // `--limit` is `--lines` by another name, and `--help` belongs to
        // every command rather than to any.
        let unlisted: Vec<_> = parsed
            .iter()
            .filter(|f| takes_value(f).is_none() && !matches!(f.as_str(), "--help" | "-h"))
            .collect();
        assert!(unlisted.is_empty(), "the parser takes these and no command lists them: {unlisted:?}");
    }

    #[test]
    fn every_flag_in_the_table_is_one_the_parser_takes_the_same_way() {
        let parsed = parsed();
        for command in COMMANDS {
            for flag in WHERE.iter().chain(command.all_flags()) {
                if command.name == "serve" && PASSED_THROUGH.contains(&flag.name) {
                    continue;
                }
                assert!(parsed.iter().any(|p| p == flag.name), "`kvad {}` lists {}, which the parser does not take", command.name, flag.name);
                let alone = crate::switch(&mut crate::Args::default(), flag.name);
                assert_eq!(alone, !flag.takes_value(), "{} of `kvad {}`: a switch to one, a value to the other", flag.name, command.name);
                if !flag.short.is_empty() {
                    assert!(parsed.iter().any(|p| p == flag.short), "{} has no {}", flag.name, flag.short);
                }
            }
        }
    }

    /// The table and the dispatch name the same commands.
    #[test]
    fn the_table_holds_every_command_and_only_those() {
        for command in COMMANDS {
            let handled = matches!(command.name, "arch" | "crawl" | "transcribe" | "service" | "cluster" | "worker" | "remote" | "tunnel" | "serve" | "help" | "completions");
            assert!(handled || crate::cli::known(command.name), "`{}` is in the table and nothing runs it", command.name);
            assert_eq!(command.server, crate::cli::remote_only(command.name), "`{}`: does it need a server?", command.name);
            // The parser gathers words only for the commands it knows take
            // them, and takes a word after any other for a flag.
            let takes_words = !command.words.is_empty() || !command.subs.is_empty();
            let gathers = crate::POSITIONAL.contains(&command.name) || matches!(command.name, "help" | "completions");
            assert_eq!(takes_words, gathers, "`{}`: words in the table, or in the parser, and not both", command.name);
            assert!(!command.about.is_empty() && !command.examples.is_empty(), "`{}` wants a line and an example", command.name);
        }
        for name in ["ls", "ps", "search", "info", "pull", "use", "rm", "cache", "load", "unload", "cancel", "tokenize", "run", "chat", "train"] {
            assert!(command(name).is_some(), "`{name}` runs and is not in the table");
        }
        for name in crate::POSITIONAL {
            assert!(command(name).is_some(), "`{name}` takes words and is not in the table");
        }
    }

    /// `kvad` is the first program, and its table is the one the rest of
    /// this file is about.
    #[test]
    fn kvad_is_the_first_program() {
        assert_eq!(PROGRAMS[0].name, "kvad");
        assert_eq!(PROGRAMS[0].bare.name, "run");
        assert!(std::ptr::eq(PROGRAMS[0].commands, COMMANDS) || PROGRAMS[0].commands.len() == COMMANDS.len());
        assert!(command("run").is_some() && command("serve").is_some());
    }

    /// `kvad tunnel`'s server, port and key switch, and the help that says
    /// what the switch gives away.
    #[test]
    fn tunnel_takes_a_server_a_port_and_the_key_switch() {
        let a = crate::parse_from(["tunnel", "--remote", "kvad://abc", "--port", "9000", "--with-key"].map(String::from).to_vec());
        assert_eq!(a.command, "tunnel");
        assert!(matches!(&a.choice, kvad::client::Choice::Remote(u) if u == "kvad://abc"));
        assert_eq!((a.port, a.with_key), (Some(9000), true));
        // Never on unless asked for.
        let a = crate::parse_from(vec!["tunnel".to_string()]);
        assert_eq!((a.port, a.with_key), (None, false));
        let tunnel = command("tunnel").unwrap();
        let with_key = tunnel.all_flags().into_iter().find(|f| f.name == "--with-key").expect("--with-key is listed");
        assert!(!with_key.takes_value());
        assert!(tunnel.notes.contains("--with-key") && tunnel.notes.contains("acts as this device"), "the help says what --with-key gives away");
    }

    /// A subcommand in the table is one its command's source matches on.
    #[test]
    fn every_subcommand_is_one_the_source_matches() {
        let sources = [include_str!("api.rs"), include_str!("models.rs"), include_str!("service.rs"), include_str!("cluster.rs"), include_str!("complete.rs"), include_str!("remote.rs"), include_str!("../main.rs")];
        for command in COMMANDS {
            for sub in command.subs {
                let arm = format!("\"{}\"", sub.name);
                assert!(sources.iter().any(|s| s.contains(&arm)), "`kvad {} {}` is in the table and no source matches it", command.name, sub.name);
            }
        }
    }
}
