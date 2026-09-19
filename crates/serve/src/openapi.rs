//! The API, described: `/api/openapi.json`, and the table it is built from.
//!
//! # What is described, and what is not
//!
//! Every endpoint, its method, its parameters, what may call it, whether it
//! streams, and — for the ones that do — the names of the events it sends.
//! Not the shape of every request and response body.
//!
//! That is a deliberate line rather than an unfinished job. Hand-written
//! JSON schemas for forty-odd endpoints are a second copy of the handlers,
//! and a second copy is a copy that drifts: the first time somebody adds a
//! field and the schema does not, the document has stopped being
//! documentation and started being a lie with a `$ref` in it. What is
//! enumerated here is exactly what a test can check against the router, and
//! [`tests::the_document_and_the_router_describe_the_same_server`] checks it.
//!
//! The one thing a schema would be worth is `/v1/chat/completions`, and there
//! the answer is better than a schema of ours: it is OpenAI's, it has not
//! changed in years, and anything that already speaks it speaks this.
//!
//! # Why not a generator
//!
//! `utoipa` and friends derive this from the handlers, which is the right
//! answer for a codebase whose handlers carry typed extractors for every
//! body. Half of these stream, several take a plain `String` body, and the
//! macro annotations would be about as long as the table below — with the
//! drift moved into attributes where no test can see it.

//! # Why this module registers no route of its own
//!
//! `/api/openapi.json` is registered in `api.rs` with everything else, and
//! the reason is the test below: it looks for route registrations by reading
//! the source of the files that make them. A scanner that scanned its own
//! source would find the string it searches *with*, and report a route called
//! `.route(`. It did, the first time.

use crate::api::Fail;
use crate::auth::Identity;
use axum::Json;
use serde_json::json;

/// Who may call an endpoint.
#[derive(Clone, Copy, PartialEq)]
pub enum Access {
    /// No credential needed. Three endpoints, and each one is a route that
    /// *has* to answer someone who has none yet.
    Anyone,
    /// Any signed-in account.
    SignedIn,
    /// An administrator. Everything that changes what the machine is doing.
    Admin,
}

pub struct Endpoint {
    pub method: &'static str,
    pub path: &'static str,
    pub tag: &'static str,
    pub summary: &'static str,
    pub description: &'static str,
    pub access: Access,
    /// `(name, required, what it means)`.
    pub query: &'static [(&'static str, bool, &'static str)],
    /// What the request body is, when there is one.
    pub body: Option<Body>,
    /// What comes back. `text/event-stream` means the `events` below.
    pub produces: &'static str,
    /// The named SSE events a streaming endpoint sends.
    pub events: &'static [(&'static str, &'static str)],
}

pub struct Body {
    pub content_type: &'static str,
    pub description: &'static str,
}

const JSON: &str = "application/json";
const SSE: &str = "text/event-stream";

/// A JSON request body, described in a sentence.
const fn json_body(description: &'static str) -> Option<Body> {
    Some(Body { content_type: JSON, description })
}

/// Every route this server answers.
///
/// In the order the sidebar visits them, because the reader of this document
/// is usually looking for the page they were just on.
pub const ENDPOINTS: &[Endpoint] = &[
    // -- Meta ---------------------------------------------------------------
    Endpoint {
        method: "get", path: "/api/health", tag: "Meta", access: Access::SignedIn,
        summary: "Is it up, and what is it holding",
        description: "Version, uptime, schema version, whether a UI is embedded, the \
                      loaded model, the queue depth, the auth mode, and who the server \
                      thinks you are. The first thing to call when something is wrong.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/openapi.json", tag: "Meta", access: Access::SignedIn,
        summary: "This document",
        description: "The OpenAPI 3.1 description of everything here.",
        query: &[], body: None, produces: JSON, events: &[],
    },

    // -- Models -------------------------------------------------------------
    Endpoint {
        method: "get", path: "/api/models", tag: "Models", access: Access::SignedIn,
        summary: "What is on this machine",
        description: "Downloaded models, models trained here, the quantised-weight \
                      cache, the active model, what the engine has loaded, and the \
                      backends this build can actually load. The filesystem is the \
                      truth: nothing here is read from the database.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "delete", path: "/api/models", tag: "Models", access: Access::Admin,
        summary: "Delete a model",
        description: "Refused while that model is loaded. Deleting it also forgets its \
                      quantised weights, which were derived from the files that just \
                      went.",
        query: &[("id", true, "A repo id or the name of a model trained here. In the \
                               query string rather than the path because repo ids \
                               contain slashes.")],
        body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/models/search", tag: "Models", access: Access::Admin,
        summary: "Search the Hub",
        description: "Each result says whether this engine can run it, and if not, why \
                      — read from the Hub's own config, before any download.",
        query: &[("q", true, "What to search for."), ("limit", false, "1 to 100, default 40.")],
        body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/models/load", tag: "Models", access: Access::Admin,
        summary: "Load a model, streaming the progress",
        description: "Loading takes tens of seconds and may download gigabytes first, \
                      so the answer is a stream rather than a request with nothing in \
                      it. `POST` and not `EventSource`, because the body names the model.",
        query: &[], body: json_body("`{ repo, backend? }`. The backend is an id from \
                                     `/api/models`; omitted means the last one used."),
        produces: SSE,
        events: &[
            ("progress", "A status line, or bytes for a download bar."),
            ("loaded", "The model is in memory; the payload describes it."),
            ("error", "It did not load, and why."),
        ],
    },
    Endpoint {
        method: "post", path: "/api/models/unload", tag: "Models", access: Access::Admin,
        summary: "Drop the loaded model",
        description: "Gives the memory back without being told what to spend it on next.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/models/active", tag: "Models", access: Access::Admin,
        summary: "Set the default model",
        description: "The one `kvad run` picks with no `--model`. Shared with the CLI \
                      and the TUI, so setting it here sets it there.",
        query: &[], body: json_body("`{ repo }`, or `{ repo: null }` to clear it."),
        produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/models/pull", tag: "Models", access: Access::Admin,
        summary: "Download a model without loading it",
        description: "A job, not a stream: a checkpoint of several gigabytes takes \
                      longer than a browser tab reliably stays open. Watch it at \
                      `/api/jobs/{id}/events`.",
        query: &[], body: json_body("`{ repo }`."), produces: JSON, events: &[],
    },
    Endpoint {
        method: "delete", path: "/api/qcache", tag: "Models", access: Access::Admin,
        summary: "Forget pre-quantised weights",
        description: "They rebuild on the next load, more slowly. Nothing is lost but time.",
        query: &[("repo", true, "The model whose cached weights to delete.")],
        body: None, produces: JSON, events: &[],
    },

    // -- Chat ---------------------------------------------------------------
    Endpoint {
        method: "post", path: "/v1/chat/completions", tag: "Chat", access: Access::SignedIn,
        summary: "Generate a reply (OpenAI-compatible)",
        description: "The compatibility surface, shaped by somebody else's \
                      documentation and kept that way. `stream: true` returns \
                      `text/event-stream` in OpenAI's chunk format, ending with \
                      `[DONE]`; otherwise one JSON object. Per-generation numbers — \
                      prefill and decode rates, cached tokens — are in an extension \
                      field, which is how this server's own UI gets them.",
        query: &[], body: json_body("OpenAI's request: `{ messages, stream?, \
                                     temperature?, top_p?, max_tokens?, seed? }`."),
        produces: "application/json or text/event-stream",
        events: &[("message", "An OpenAI chunk, or the literal `[DONE]`.")],
    },
    Endpoint {
        method: "get", path: "/v1/models", tag: "Chat", access: Access::SignedIn,
        summary: "What is loaded, in OpenAI's shape",
        description: "One entry, because the engine holds one model.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "delete", path: "/api/generation", tag: "Chat", access: Access::SignedIn,
        summary: "Stop generating",
        description: "Affects whatever the engine is doing now, not a particular \
                      request — with one generation at a time those are the same thing.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/conversations", tag: "Chat", access: Access::SignedIn,
        summary: "Your conversations",
        description: "Yours only: another account's are a 404 rather than a 403, \
                      because whether a conversation exists is itself private.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/conversations", tag: "Chat", access: Access::SignedIn,
        summary: "Start a conversation",
        description: "", query: &[],
        body: json_body("`{ title?, system? }`."), produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/conversations/{id}", tag: "Chat", access: Access::SignedIn,
        summary: "A conversation and its messages",
        description: "", query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "patch", path: "/api/conversations/{id}", tag: "Chat", access: Access::SignedIn,
        summary: "Rename it, or change its system prompt",
        description: "", query: &[],
        body: json_body("`{ title?, system? }`."), produces: JSON, events: &[],
    },
    Endpoint {
        method: "delete", path: "/api/conversations/{id}", tag: "Chat", access: Access::SignedIn,
        summary: "Delete a conversation",
        description: "Takes its messages with it.", query: &[], body: None,
        produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/conversations/{id}/messages", tag: "Chat",
        access: Access::SignedIn,
        summary: "Append a message",
        description: "Written by the client after a generation finishes, with the \
                      numbers that generation reported.",
        query: &[], body: json_body("`{ role, content, stats? }`."),
        produces: JSON, events: &[],
    },

    // -- Playground ---------------------------------------------------------
    Endpoint {
        method: "post", path: "/api/playground/complete", tag: "Playground",
        access: Access::SignedIn,
        summary: "Continue a prompt, with no chat template",
        description: "The base-model view of a model, which for an instruct model is a \
                      different and more revealing thing than talking to it. `explain` \
                      asks for the candidates behind each token — the model's own \
                      probabilities over the whole vocabulary, and a flag saying \
                      whether top-k and top-p left each one in play.",
        query: &[],
        body: json_body("`{ prompt, temperature?, top_k?, top_p?, max_tokens?, seed?, \
                         explain?, fresh? }`. `explain` is 0 to 20 candidates a token; \
                         `fresh` (default true) drops the KV cache first, so the same \
                         prompt twice is genuinely twice."),
        produces: SSE,
        events: &[
            ("token", "`{ text }`, or `{ id, text, top[] }` when `explain` was asked for."),
            ("done", "Prompt and generated token counts, prefill and decode seconds."),
            ("error", "It stopped, and why."),
        ],
    },
    Endpoint {
        method: "post", path: "/api/playground/tokenize", tag: "Playground",
        access: Access::SignedIn,
        summary: "How the loaded model splits a text",
        description: "Each token as the vocabulary entry (`Ġthe`, and all) and as the \
                      piece of your text it covers. The two disagree, and seeing how \
                      is most of the point.",
        query: &[], body: json_body("`{ text }`."), produces: JSON, events: &[],
    },

    // -- Training -----------------------------------------------------------
    Endpoint {
        method: "post", path: "/api/train", tag: "Training", access: Access::Admin,
        summary: "Start a training run",
        description: "Refused if one is already going: two runs would fight for the \
                      same cores and each take twice as long. Every check that can be \
                      made before an hour is committed — the dataset exists, the model \
                      to continue from has a token for every character in it — is made \
                      here rather than at step one.",
        query: &[],
        body: json_body("`{ dataset, name?, from?, size?, steps?, lr?, eval_every?, \
                         threads?, sample?, seed? }`."),
        produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/train/options", tag: "Training", access: Access::Admin,
        summary: "What a run can be asked for",
        description: "Model sizes, models that can be continued, defaults, the core \
                      count, and whether a run is already going.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/jobs", tag: "Jobs", access: Access::SignedIn,
        summary: "Everything long-running, newest first",
        description: "Downloads, training runs, evals and benchmarks in one history.",
        query: &[("limit", false, "1 to 500, default 50.")],
        body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/jobs/{id}", tag: "Jobs", access: Access::SignedIn,
        summary: "One job, with its chart",
        description: "The row, plus the training metrics and samples belonging to it.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "delete", path: "/api/jobs/{id}", tag: "Jobs", access: Access::Admin,
        summary: "Ask a job to stop",
        description: "Advisory: a download finishes the file it is on and a training \
                      run stops at its next step. The row stays — a cancelled run is \
                      history, not an absence.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/jobs/{id}/events", tag: "Jobs", access: Access::SignedIn,
        summary: "Follow a job",
        description: "Everything that has happened, then everything that does. The \
                      subscription is taken before the history is read, so an update \
                      landing between the two is seen rather than lost — and may \
                      therefore be seen twice, which is why every update is keyed.",
        query: &[], body: None, produces: SSE,
        events: &[("update", "One of `status`, `download`, `metric`, `sample`, `pace`, \
                              `progress`, `case`, `scored`, `timing`, or `ended`.")],
    },

    // -- Datasets -----------------------------------------------------------
    Endpoint {
        method: "get", path: "/api/datasets", tag: "Datasets", access: Access::Admin,
        summary: "The text files runs are trained on",
        description: "", query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/datasets", tag: "Datasets", access: Access::Admin,
        summary: "Upload a corpus",
        description: "The body is the text itself, not a multipart form: what is being \
                      sent is one file's contents and nothing else.",
        query: &[("name", true, "What to call it.")],
        body: Some(Body { content_type: "text/plain", description: "The corpus." }),
        produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/datasets/{id}", tag: "Datasets", access: Access::Admin,
        summary: "A dataset, with the start of it",
        description: "", query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "delete", path: "/api/datasets/{id}", tag: "Datasets", access: Access::Admin,
        summary: "Delete a dataset",
        description: "", query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/datasets/{id}/check", tag: "Datasets", access: Access::Admin,
        summary: "Would this text fit that model's vocabulary",
        description: "A character tokeniser is fixed at first training, so a corpus \
                      with a character the model has no token for cannot continue it. \
                      This is the answer before committing to a run, and it names every \
                      missing character rather than the first.",
        query: &[("model", true, "A model trained here.")],
        body: None, produces: JSON, events: &[],
    },

    // -- Evals --------------------------------------------------------------
    Endpoint {
        method: "get", path: "/api/evals/suites", tag: "Evals", access: Access::SignedIn,
        summary: "Saved prompt suites",
        description: "", query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/evals/suites", tag: "Evals", access: Access::Admin,
        summary: "Save a suite",
        description: "Checked when it is written rather than when it is run: a case \
                      with no expectation is a prompt, not a test.",
        query: &[],
        body: json_body("`{ name, cases: [{ prompt, expect, match? }] }`, where `match` \
                         is `contains` (the default) or `equals`."),
        produces: JSON, events: &[],
    },
    Endpoint {
        method: "patch", path: "/api/evals/suites/{id}", tag: "Evals", access: Access::Admin,
        summary: "Change a suite",
        description: "", query: &[],
        body: json_body("`{ name?, cases? }`."), produces: JSON, events: &[],
    },
    Endpoint {
        method: "delete", path: "/api/evals/suites/{id}", tag: "Evals", access: Access::Admin,
        summary: "Delete a suite",
        description: "", query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/evals/run", tag: "Evals", access: Access::Admin,
        summary: "Run a suite across variants",
        description: "A job. Cases decode greedily with a fixed seed, so a case that \
                      fails is a change in the model rather than a change in the dice.",
        query: &[],
        body: json_body("`{ suite, variants: [{ model, backend }], max_tokens?, seed? }`."),
        produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/evals/perplexity", tag: "Evals", access: Access::Admin,
        summary: "Score held-out text",
        description: "A job. The exponential of the model's mean surprise per token, \
                      scored in windows; the first token of each window is not graded, \
                      because nothing precedes it.",
        query: &[],
        body: json_body("`{ dataset, variants: [{ model, backend }], window? }`."),
        produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/evals/runs", tag: "Evals", access: Access::SignedIn,
        summary: "Eval runs, newest first",
        description: "", query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/evals/runs/{id}", tag: "Evals", access: Access::SignedIn,
        summary: "One run, with every verdict",
        description: "Per-case verdicts for a suite, per-variant scores for a \
                      perplexity run.",
        query: &[], body: None, produces: JSON, events: &[],
    },

    // -- Benchmarks ---------------------------------------------------------
    Endpoint {
        method: "post", path: "/api/bench/run", tag: "Benchmarks", access: Access::Admin,
        summary: "Measure variants against each other",
        description: "A job. Refused while a training run, an eval or another \
                      benchmark is going, because a measurement taken next to other \
                      work measures the other work. A round visits every variant once \
                      and a run is several rounds, so a machine that warms up shows as \
                      a trend rather than as a winner. Every timed generation starts \
                      from an empty KV cache.",
        query: &[],
        body: json_body("`{ variants: [{ model, backend }], prompt?, rounds?, tokens?, \
                         seed?, keep_text? }`."),
        produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/bench/runs", tag: "Benchmarks", access: Access::SignedIn,
        summary: "Benchmark runs, newest first",
        description: "", query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/bench/runs/{id}", tag: "Benchmarks", access: Access::SignedIn,
        summary: "Every sample of a run, and the summary from them",
        description: "The summary is recomputed from the samples rather than read \
                      back, so a run still going has one too.",
        query: &[], body: None, produces: JSON, events: &[],
    },

    // -- Monitoring ---------------------------------------------------------
    Endpoint {
        method: "get", path: "/api/metrics", tag: "Monitoring", access: Access::SignedIn,
        summary: "What the machine is doing",
        description: "The loaded model, decode and time-to-first-token summaries, \
                      queue depth, resident memory, the KV cache against what it would \
                      cost at full context, and disk broken into what can be downloaded \
                      again and what cannot.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/metrics/requests", tag: "Monitoring", access: Access::SignedIn,
        summary: "Recent requests, and what each route costs",
        description: "Percentiles are taken from the samples themselves, sorted. There \
                      are few enough that an estimate would be a worse number for no \
                      saving.",
        query: &[("limit", false, "How many recent requests to return.")],
        body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/metrics/log", tag: "Monitoring", access: Access::SignedIn,
        summary: "The server log tail",
        description: "Bounded, in memory, and gone on restart: a log that has to \
                      outlive the process belongs to whatever is running it.",
        query: &[("limit", false, "How many lines.")],
        body: None, produces: JSON, events: &[],
    },

    // -- Accounts -----------------------------------------------------------
    Endpoint {
        method: "get", path: "/api/auth", tag: "Accounts", access: Access::Anyone,
        summary: "How to sign in here, and whether you already are",
        description: "The one route that has to answer a request with no credential at \
                      all. Says which mode is in force and whether the server is still \
                      waiting for its first account.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/auth/login", tag: "Accounts", access: Access::Anyone,
        summary: "Sign in",
        description: "Sets an `HttpOnly; SameSite=Lax` session cookie. Wrong password \
                      and unknown account are the same answer.",
        query: &[], body: json_body("`{ name, password }`."), produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/auth/logout", tag: "Accounts", access: Access::SignedIn,
        summary: "Sign out",
        description: "Ends this session server-side, not just in the browser.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/auth/setup", tag: "Accounts", access: Access::Anyone,
        summary: "Create the first account",
        description: "Needs the one-time token printed to the terminal at first start, \
                      and works exactly once.",
        query: &[], body: json_body("`{ token, name, password }`."),
        produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/auth/oidc/start", tag: "Accounts", access: Access::Anyone,
        summary: "Begin an OIDC sign-in",
        description: "Authorization code with PKCE. Redirects to the provider.",
        query: &[], body: None, produces: "redirect", events: &[],
    },
    Endpoint {
        method: "get", path: "/api/auth/oidc/callback", tag: "Accounts", access: Access::Anyone,
        summary: "Where the provider sends them back",
        description: "", query: &[
            ("code", true, "The authorization code."),
            ("state", true, "The value this server issued."),
        ], body: None, produces: "redirect", events: &[],
    },
    Endpoint {
        method: "post", path: "/api/auth/password", tag: "Accounts", access: Access::SignedIn,
        summary: "Change your password",
        description: "Ends every other session for that account, which is the point of \
                      changing it.",
        query: &[], body: json_body("`{ current, new }`."), produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/users", tag: "Accounts", access: Access::Admin,
        summary: "Accounts on this server",
        description: "", query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/users", tag: "Accounts", access: Access::Admin,
        summary: "Create an account",
        description: "", query: &[],
        body: json_body("`{ name, password, role }`, where role is `admin` or `user`."),
        produces: JSON, events: &[],
    },
    Endpoint {
        method: "patch", path: "/api/users/{id}", tag: "Accounts", access: Access::Admin,
        summary: "Change an account's role or password",
        description: "", query: &[],
        body: json_body("`{ role?, password? }`."), produces: JSON, events: &[],
    },
    Endpoint {
        method: "delete", path: "/api/users/{id}", tag: "Accounts", access: Access::Admin,
        summary: "Delete an account",
        description: "Takes its sessions, keys and conversations with it. The last \
                      administrator cannot be deleted.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/sessions", tag: "Accounts", access: Access::SignedIn,
        summary: "Your sessions",
        description: "", query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "delete", path: "/api/sessions/{hash}", tag: "Accounts", access: Access::SignedIn,
        summary: "End a session",
        description: "By the hash the listing gives, because the token itself is never \
                      stored and never shown again.",
        query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "get", path: "/api/keys", tag: "Accounts", access: Access::SignedIn,
        summary: "Your API keys",
        description: "", query: &[], body: None, produces: JSON, events: &[],
    },
    Endpoint {
        method: "post", path: "/api/keys", tag: "Accounts", access: Access::SignedIn,
        summary: "Create an API key",
        description: "Shown once. Only its hash is kept, so a lost key is replaced \
                      rather than recovered.",
        query: &[], body: json_body("`{ name }`."), produces: JSON, events: &[],
    },
    Endpoint {
        method: "delete", path: "/api/keys/{id}", tag: "Accounts", access: Access::SignedIn,
        summary: "Revoke an API key",
        description: "", query: &[], body: None, produces: JSON, events: &[],
    },
];

impl Access {
    /// What OpenAPI's `security` says for this endpoint.
    ///
    /// An empty list means "no credential required"; anything else means one
    /// of the listed schemes. OpenAPI cannot express "and must be an
    /// administrator", so that is in the description, where a person will
    /// read it.
    fn security(self) -> serde_json::Value {
        match self {
            Access::Anyone => json!([{}]),
            _ => json!([{ "apiKey": [] }, { "session": [] }, { "basic": [] }]),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Access::Anyone => "no credential required",
            Access::SignedIn => "any signed-in account",
            Access::Admin => "administrators only",
        }
    }
}

/// The OpenAPI document, built from [`ENDPOINTS`].
pub fn build() -> serde_json::Value {
    let mut paths = serde_json::Map::new();
    for e in ENDPOINTS {
        let entry = paths
            .entry(e.path.to_string())
            .or_insert_with(|| json!({}));
        let operation = operation(e);
        if let Some(obj) = entry.as_object_mut() {
            obj.insert(e.method.to_string(), operation);
        }
    }

    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "kvad-serve",
            "version": env!("CARGO_PKG_VERSION"),
            "description": DESCRIPTION,
            "license": { "name": "MIT" },
        },
        "servers": [{ "url": "/", "description": "This server." }],
        "tags": TAGS.iter().map(|(name, about)| json!({
            "name": name, "description": about
        })).collect::<Vec<_>>(),
        "components": { "securitySchemes": {
            "apiKey": {
                "type": "http", "scheme": "bearer",
                "description": "An API key from `/api/keys`, as \
                                `Authorization: Bearer kvad_…`. Works in every auth \
                                mode, and is what `/v1` is meant to be called with.",
            },
            "session": {
                "type": "apiKey", "in": "cookie", "name": "kvad_session",
                "description": "The cookie `/api/auth/login` sets. Mutating requests \
                                authenticated this way are also checked for a matching \
                                `Origin`, which is the CSRF defence; a bearer token \
                                needs no such check because a browser will not attach \
                                one on its own.",
            },
            "basic": {
                "type": "http", "scheme": "basic",
                "description": "The same accounts over HTTP Basic, when the server is \
                                started in `basic` mode. For scripts and reverse \
                                proxies.",
            },
        }},
        "paths": paths,
    })
}

const DESCRIPTION: &str = "\
Two surfaces, kept apart on purpose.

`/v1/**` is OpenAI-compatible. It is shaped by somebody else's documentation and \
has to stay that way, which is the whole point of it: anything that already speaks \
to OpenAI speaks to this. This server's own web UI chats through it like any other \
client, so it is the path that gets exercised every day rather than the one that \
quietly rots.

`/api/**` is this server's own — models, jobs, evals, monitoring, accounts. It can \
change whenever the UI needs it to.

Request and response bodies are described in prose rather than as schemas. That is \
a deliberate line: a hand-written schema for forty-odd endpoints is a second copy of \
the handlers, and a second copy drifts. What is enumerated here — every path, method, \
parameter and stream — is checked against the router by a test.";

const TAGS: &[(&str, &str)] = &[
    ("Meta", "Is it up, and what is it."),
    ("Models", "What is on this machine, and what the engine holds."),
    ("Chat", "Generating, and the conversations kept around it."),
    ("Playground", "The model without the conversation: raw completion, the \
                    tokeniser, and the choice behind each token."),
    ("Training", "Starting runs on this machine's own cores."),
    ("Jobs", "Work that outlives the request that asked for it."),
    ("Datasets", "The text files runs are trained and scored on."),
    ("Evals", "Perplexity, and prompt suites as regression tests."),
    ("Benchmarks", "The measurement protocol: interleaved rounds, median and range."),
    ("Monitoring", "Requests, latencies and the log."),
    ("Accounts", "Signing in, users, sessions and API keys."),
];

fn operation(e: &Endpoint) -> serde_json::Value {
    let mut parameters: Vec<serde_json::Value> = path_params(e.path)
        .into_iter()
        .map(|name| {
            json!({
                "name": name, "in": "path", "required": true,
                "schema": { "type": "string" },
            })
        })
        .collect();
    parameters.extend(e.query.iter().map(|(name, required, about)| {
        json!({
            "name": name, "in": "query", "required": required,
            "description": about, "schema": { "type": "string" },
        })
    }));

    // The events a stream sends are documented where a person will look for
    // them — in the description — because OpenAPI has nowhere to put the
    // names of server-sent events.
    let mut description = format!("{}\n\nAccess: {}.", e.description.trim(), e.access.label());
    if !e.events.is_empty() {
        description.push_str("\n\nEvents:\n");
        for (name, about) in e.events {
            description.push_str(&format!("\n- `{name}` — {about}"));
        }
    }

    let mut operation = json!({
        "summary": e.summary,
        "description": description.trim(),
        "tags": [e.tag],
        "security": e.access.security(),
        "responses": {
            "200": { "description": "Success.",
                     "content": { e.produces: { "schema": { "type": "object" } } } },
            "4XX": { "description": "`{ \"error\": \"…\" }`, a sentence for a person to read.",
                     "content": { "application/json": { "schema": { "type": "object" } } } },
        },
    });
    if !parameters.is_empty() {
        operation["parameters"] = json!(parameters);
    }
    if let Some(body) = &e.body {
        operation["requestBody"] = json!({
            "required": true,
            "description": body.description,
            "content": { body.content_type: { "schema": { "type": "object" } } },
        });
    }
    operation
}

/// The `{name}` placeholders in a path, in order.
fn path_params(path: &str) -> Vec<&str> {
    path.split('{').skip(1).filter_map(|rest| rest.split('}').next()).collect()
}

pub async fn document(_: Identity) -> Result<Json<serde_json::Value>, Fail> {
    Ok(Json(build()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every file that calls `.route(`.
    ///
    /// Read as text at compile time rather than asked of the `Router`, which
    /// has no way to list what is in it. Scanning the source is cruder than
    /// asking, and it has the one property that matters: a route added to any
    /// of these files and not to `ENDPOINTS` fails this test, without anybody
    /// having to remember to register it in a third place.
    /// `openapi.rs` is deliberately not here — see the note at the top of this
    /// module — and it registers nothing, so there is nothing to miss.
    const SOURCES: &[(&str, &str)] = &[
        ("api.rs", include_str!("api.rs")),
        ("training.rs", include_str!("training.rs")),
        ("monitoring.rs", include_str!("monitoring.rs")),
        ("playground.rs", include_str!("playground.rs")),
        ("evals.rs", include_str!("evals.rs")),
        ("bench.rs", include_str!("bench.rs")),
    ];

    const METHODS: &[&str] = &["get", "post", "put", "patch", "delete"];

    /// Every `(method, path)` the router actually serves.
    fn routes_in_source() -> BTreeSet<(String, String)> {
        let mut found = BTreeSet::new();
        for (_, source) in SOURCES {
            let mut rest = *source;
            while let Some(at) = rest.find(".route(") {
                rest = &rest[at + ".route(".len()..];
                // The path may be on the next line — `rustfmt` wraps a
                // `.route(` whose handlers are long — so skip to the quote
                // rather than expecting it immediately.
                let Some(quote) = rest.find('"') else { break };
                rest = &rest[quote + 1..];
                let Some(end) = rest.find('"') else { break };
                let path = rest[..end].to_string();
                rest = &rest[end + 1..];

                // The handlers, up to the `)` that closes this `.route(` call.
                let mut depth = 1usize;
                let mut cut = rest.len();
                for (i, c) in rest.char_indices() {
                    match c {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                cut = i;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                let handlers = &rest[..cut];
                for method in METHODS {
                    if mentions(handlers, method) {
                        found.insert((method.to_string(), path.clone()));
                    }
                }
                rest = &rest[cut..];
            }
        }
        found
    }

    /// Does this text call `method(`, as a name of its own?
    ///
    /// The preceding character has to be something other than a letter, a
    /// digit or an underscore, or `budget(` would read as `get(` and
    /// `delete_suite(` — which is a handler, not a method — would read as a
    /// `delete`.
    fn mentions(handlers: &str, method: &str) -> bool {
        let needle = format!("{method}(");
        let mut from = 0;
        while let Some(at) = handlers[from..].find(&needle) {
            let at = from + at;
            let before = handlers[..at].chars().next_back();
            if !before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                return true;
            }
            from = at + 1;
        }
        false
    }

    /// The document and the server have to be the same server.
    ///
    /// This is the whole reason the description is a table rather than prose
    /// in a Markdown file: a table can be compared to the thing it describes.
    #[test]
    fn the_document_and_the_router_describe_the_same_server() {
        let served = routes_in_source();
        let described: BTreeSet<(String, String)> =
            ENDPOINTS.iter().map(|e| (e.method.to_string(), e.path.to_string())).collect();

        let missing: Vec<_> = served.difference(&described).collect();
        assert!(
            missing.is_empty(),
            "these routes are served and not described in ENDPOINTS: {missing:#?}"
        );
        let invented: Vec<_> = described.difference(&served).collect();
        assert!(
            invented.is_empty(),
            "these routes are described and not served: {invented:#?}"
        );

        // And the scanner has to have found something, or an empty set would
        // match an empty set and this test would pass on a broken parse.
        assert!(served.len() > 40, "only found {} routes; the scanner is broken", served.len());
    }

    /// A method that reads as another method's name would make the check
    /// above pass on a mismatch, so the scanner's own rule is tested.
    #[test]
    fn the_scanner_tells_a_method_from_a_handler_that_ends_in_one() {
        assert!(mentions("get(health)", "get"));
        assert!(mentions("axum::routing::patch(update)", "patch"));
        assert!(mentions("get(a).delete(b)", "delete"));
        assert!(!mentions("post(budget)", "get"), "`budget(` read as `get(`");
        assert!(!mentions("post(delete_suite)", "delete"), "a handler read as a method");
        // And a handler whose name *is* a method still does not count as one.
        assert!(!mentions("post(crate::conversations::get)", "get"));
    }

    /// The generated document has to be a document: every described endpoint
    /// present, tagged with a tag that exists, and secured.
    #[test]
    fn the_generated_document_covers_every_endpoint() {
        let doc = build();
        let paths = doc["paths"].as_object().expect("no paths");
        let tags: BTreeSet<&str> = TAGS.iter().map(|(n, _)| *n).collect();

        let mut operations = 0;
        for e in ENDPOINTS {
            let operation = &paths[e.path][e.method];
            assert!(!operation.is_null(), "{} {} is missing", e.method, e.path);
            assert_eq!(operation["summary"], e.summary);
            assert!(tags.contains(e.tag), "unknown tag `{}`", e.tag);
            assert!(operation["security"].is_array());
            // The access rule is in the description, because OpenAPI has
            // nowhere to say "and must be an administrator".
            let described = operation["description"].as_str().unwrap_or("");
            assert!(described.contains(e.access.label()), "{} {}", e.method, e.path);
            operations += 1;
        }
        assert_eq!(operations, ENDPOINTS.len());

        // Path placeholders become parameters, or a generated client builds
        // URLs with braces in them.
        let one = &paths["/api/jobs/{id}"]["get"]["parameters"][0];
        assert_eq!(one["name"], "id");
        assert_eq!(one["in"], "path");
        assert_eq!(one["required"], true);

        // Streams say so, and say what they send.
        let load = &paths["/api/models/load"]["post"];
        assert!(load["responses"]["200"]["content"]["text/event-stream"].is_object());
        assert!(load["description"].as_str().unwrap().contains("`loaded`"));
    }

    /// Three routes answer someone with no credential, and there must be no
    /// fourth by accident.
    #[test]
    fn only_the_routes_that_have_to_be_open_are_open() {
        let open: Vec<&str> = ENDPOINTS
            .iter()
            .filter(|e| e.access == Access::Anyone)
            .map(|e| e.path)
            .collect();
        assert_eq!(
            open,
            [
                "/api/auth",
                "/api/auth/login",
                "/api/auth/setup",
                "/api/auth/oidc/start",
                "/api/auth/oidc/callback",
            ],
            "the set of unauthenticated routes changed"
        );
    }

    #[test]
    fn path_parameters_are_read_out_of_the_path() {
        assert_eq!(path_params("/api/jobs/{id}/events"), ["id"]);
        assert_eq!(path_params("/api/health"), Vec::<&str>::new());
        assert_eq!(path_params("/a/{x}/b/{y}"), ["x", "y"]);
    }
}
