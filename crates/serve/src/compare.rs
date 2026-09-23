//! Running the same thing against several models, one after another.
//!
//! Evals and benchmarks ask different questions and share one awkward fact:
//! "side by side" is a lie about the hardware. The server can hold several
//! models at once, but a model measured beside another is measured on the
//! memory they share — `engine::preferred` records a CPU model decoding at
//! half speed because a GPU model beside it had evicted its weights. So a
//! comparison runs each variant *alone*: whatever else is in memory is
//! unloaded first, and what actually happens is a sequence — load, measure,
//! load the next. Everything in this file exists to make that sequence
//! honest.
//!
//! # Rounds, not runs
//!
//! A benchmark that measured A five times and then B five times would blame
//! the model for anything that changed about the machine in between: a fan
//! spinning up, a browser opening, the quantised-weight cache warming. So a
//! round visits *every* variant once, and the run is several rounds. Each
//! variant is then sampled across the whole span, and drift shows as a trend
//! in the rounds rather than as a winner.
//!
//! The price is a model load per variant per round, which is why a run of
//! three variants and five rounds is minutes rather than seconds. It is the
//! honest price of one model at a time.
//!
//! # Everything is seeded
//!
//! Comparisons fix the sampler's seed, and suites run greedily. Two variants
//! that differ only in their random draw are not a comparison, and a
//! regression test that fails one time in five is not a test.
//!
//! # The cache is dropped before every timed generation
//!
//! The same prompt twice in a row would otherwise be prefilled out of the
//! first run's KV cache and report a time to first token no first run would
//! ever see. See [`kvad::service::Cmd::Complete`].

use crate::jobs::{Case, Jobs, Scored, Timing, Update};
use crate::scheduler::{Key, Piece, Scheduler};
use kvad::chat::Message;
use kvad::runtime::Stats;
use kvad::service::{Backend, Sampling};
use rusqlite::params;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::broadcast;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// One model at one precision: the thing a comparison compares.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Variant {
    /// A repo id or the name of a model trained here.
    pub model: String,
    /// A backend id from `/api/models`, e.g. `cpu-q8`.
    pub backend: String,
}

impl Variant {
    /// What the results are filed under, and what a person reads.
    ///
    /// Text rather than a pair of foreign keys, because a result has to
    /// outlive the model it was measured on: deleting a checkpoint should not
    /// quietly empty last week's benchmark.
    pub fn label(&self) -> String {
        format!("{} · {}", self.model, self.backend)
    }
}

/// Check every variant before starting, so a run of three does not fail on
/// the third after ten minutes of the first two.
pub fn resolve(variants: &[Variant]) -> Result<Vec<(Variant, Backend)>, String> {
    if variants.is_empty() {
        return Err("a comparison needs at least one model".into());
    }
    let mut seen = Vec::new();
    let mut out = Vec::new();
    for v in variants {
        if v.model.trim().is_empty() {
            return Err("a variant with no model in it".into());
        }
        let backend = crate::engine::parse(&v.backend).ok_or_else(|| {
            format!(
                "`{}` is not a backend this build can load; it has {}",
                v.backend,
                crate::engine::available().iter().map(|c| c.id.clone()).collect::<Vec<_>>().join(", ")
            )
        })?;
        let label = v.label();
        if seen.contains(&label) {
            return Err(format!("`{label}` is in this comparison twice"));
        }
        seen.push(label);
        out.push((v.clone(), backend));
    }
    Ok(out)
}

/// Make `variant` the only model in memory, and say which resident it is.
///
/// Everything else is unloaded first, including models somebody loaded for
/// other reasons; see the module docs for why a measurement has to be taken
/// alone. Reloading a variant that is already there would cost a minute and
/// change nothing, so the check is worth the branch — and it is what makes a
/// suite of forty cases against one variant load once rather than forty
/// times.
async fn switch(
    engine: &Arc<Scheduler>,
    events: &broadcast::Sender<Update>,
    variant: &Variant,
    backend: Backend,
) -> Result<Key, String> {
    let key = Key { repo: variant.model.clone(), backend };
    for other in engine.residents().into_iter().filter(|r| r.key != key) {
        let _ = events.send(Update::Status { message: format!("unloading {}", other.id) });
        engine.unload(Some(other.key)).await?;
    }
    if engine.residents().iter().any(|r| r.key == key) {
        return Ok(key);
    }
    let _ = events.send(Update::Status { message: format!("loading {}", variant.label()) });
    let (progress, mut updates) = tokio::sync::mpsc::channel(32);
    let forward = {
        let events = events.clone();
        tokio::spawn(async move {
            while let Some(p) = updates.recv().await {
                let _ = events.send(Update::from(p));
            }
        })
    };
    let outcome = engine.load(variant.model.clone(), backend, progress).await;
    forward.abort();
    outcome.map(|_| key).map_err(String::from)
}

/// What a generation that stopped before its first token is called.
///
/// The model chose its end token first. It is not an engine failure, and it
/// is not an answer either.
const WROTE_NOTHING: &str = "the model wrote nothing: it ended its turn before the first token";

/// Read one generation to its end, and return what was written and what it
/// cost.
///
/// The caller decides how the prompt reaches the model — a suite as a chat
/// turn, a benchmark as raw text — and asks for a cold cache either way.
async fn once(
    pieces: Result<tokio::sync::mpsc::Receiver<Piece>, String>,
) -> Result<(String, Stats), String> {
    let mut pieces = pieces?;
    let mut text = String::new();
    while let Some(piece) = pieces.recv().await {
        match piece {
            Piece::Token(t) => text.push_str(&t),
            Piece::Chose(c) => text.push_str(&c.text),
            Piece::Done(stats) => return Ok((text, stats)),
            Piece::Failed(why) => return Err(why),
        }
    }
    Err("the generation ended without saying so".into())
}

/// How a case's expectation is checked.
///
/// Two ways, because a third would be a regular-expression dependency and a
/// fourth would be a language. `contains` is what almost every case wants;
/// `equals` is for the ones where trailing chatter is the bug.
fn matches(got: &str, expect: &str, how: &str) -> bool {
    let (got, expect) = (got.trim(), expect.trim());
    match how {
        "equals" => got.eq_ignore_ascii_case(expect),
        // The default, including for anything unrecognised: a suite with a
        // typo in its match kind should be lenient rather than all-failing.
        _ => got.to_lowercase().contains(&expect.to_lowercase()),
    }
}

/// One case of a prompt suite, as it is written down.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Expectation {
    pub prompt: String,
    pub expect: String,
    #[serde(default, rename = "match")]
    pub how: Option<String>,
}

/// Run a prompt suite across every variant.
#[allow(clippy::too_many_arguments)]
pub fn suite(
    jobs: &Arc<Jobs>,
    engine: &Arc<Scheduler>,
    name: String,
    cases: Vec<Expectation>,
    variants: Vec<(Variant, Backend)>,
    max_tokens: usize,
    seed: u64,
    owner: Option<i64>,
) -> Res<crate::jobs::Job> {
    if cases.is_empty() {
        return Err(format!("`{name}` has no cases in it").into());
    }
    let params = serde_json::json!({
        "suite": name,
        "cases": cases.len(),
        "variants": variants.iter().map(|(v, _)| v).collect::<Vec<_>>(),
        "max_tokens": max_tokens,
        "seed": seed,
    });
    let id = jobs.create("eval", &name, &params, owner)?;
    let (cancel, events) = jobs.register(id);
    let worker = Arc::clone(jobs);
    let engine = Arc::clone(engine);

    tokio::spawn(async move {
        let mut passed = 0usize;
        let mut total = 0usize;
        let mut per_variant = Vec::new();
        let mut failure: Option<String> = None;

        'variants: for (variant, backend) in &variants {
            let on = match switch(&engine, &events, variant, *backend).await {
                Ok(on) => on,
                Err(why) => {
                    failure = Some(format!("{}: {why}", variant.label()));
                    break;
                }
            };
            let mut here = 0usize;
            for (idx, case) in cases.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    break 'variants;
                }
                // Greedy, seeded: a regression test that disagrees with
                // itself on a rerun is not evidence of anything.
                let sampling = Sampling {
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 1.0,
                    seed: Some(seed),
                    max_tokens,
                };
                // A case is a question put to the model, so it goes through
                // the model's own template as a user turn. As raw text, an
                // instruct model reads "Capital of France?" as a finished
                // user message and its first token is end-of-turn: an empty
                // answer to every case. A base model has no template and gets
                // the text as it is.
                let asked = vec![Message::user(case.prompt.clone())];
                let pieces = engine.chat(&on, asked, Vec::new(), sampling, true);
                let (got, stats) = match once(pieces).await {
                    Ok(answer) => answer,
                    // A case that could not run at all is a failed case with
                    // the reason in it, not a failed run: the other thirty
                    // still say something.
                    Err(why) => (format!("<{why}>"), Stats::default()),
                };
                let how = case.how.clone().unwrap_or_else(|| "contains".into());
                let right = matches(&got, &case.expect, &how);
                // An empty answer reads like a page that failed to show one,
                // so it says what happened instead — after the verdict, which
                // is about what the model wrote.
                let got = match stats.generated_tokens {
                    0 if got.is_empty() => format!("<{WROTE_NOTHING}>"),
                    _ => got,
                };
                let verdict = Case {
                    variant: variant.label(),
                    idx: idx as i64,
                    prompt: case.prompt.clone(),
                    expect: case.expect.clone(),
                    passed: right,
                    got,
                    decode_per_sec: (stats.generated_tokens > 0)
                        .then(|| stats.tokens_per_sec() as f64),
                    generated_tokens: Some(stats.generated_tokens as i64),
                };
                total += 1;
                if verdict.passed {
                    passed += 1;
                    here += 1;
                }
                let _ = worker.db.with(|c| {
                    c.execute(
                        "INSERT OR REPLACE INTO eval_results
                         (job, variant, idx, prompt, expect, got, passed,
                          decode_per_sec, generated_tokens)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                        params![
                            id,
                            verdict.variant,
                            verdict.idx,
                            verdict.prompt,
                            verdict.expect,
                            verdict.got,
                            verdict.passed as i64,
                            verdict.decode_per_sec,
                            verdict.generated_tokens
                        ],
                    )
                });
                let _ = events.send(Update::Case(verdict));
                let _ = events.send(Update::Progress { done: total, total: cases.len() * variants.len() });
            }
            per_variant.push(serde_json::json!({
                "variant": variant.label(),
                "passed": here,
                "of": cases.len(),
            }));
        }

        let result = serde_json::json!({
            "suite": name,
            "passed": passed,
            "of": total,
            "variants": per_variant,
        });
        match (failure, cancel.load(Ordering::Relaxed)) {
            (Some(why), _) => worker.finish(id, "failed", Some(why), Some(result)),
            (None, true) => worker.finish(id, "cancelled", None, Some(result)),
            (None, false) => worker.finish(id, "done", None, Some(result)),
        }
    });

    jobs.get(id)?.ok_or_else(|| "the job vanished as it started".into())
}

/// Score a held-out text on every variant.
pub fn score(
    jobs: &Arc<Jobs>,
    engine: &Arc<Scheduler>,
    dataset: String,
    text: String,
    variants: Vec<(Variant, Backend)>,
    window: usize,
    owner: Option<i64>,
) -> Res<crate::jobs::Job> {
    let params = serde_json::json!({
        "dataset": dataset,
        "characters": text.chars().count(),
        "window": window,
        "variants": variants.iter().map(|(v, _)| v).collect::<Vec<_>>(),
    });
    let id = jobs.create("eval", &format!("perplexity · {dataset}"), &params, owner)?;
    let (cancel, events) = jobs.register(id);
    let worker = Arc::clone(jobs);
    let engine = Arc::clone(engine);

    tokio::spawn(async move {
        let mut scores = Vec::new();
        let mut failure: Option<String> = None;

        for (variant, backend) in &variants {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let on = match switch(&engine, &events, variant, *backend).await {
                Ok(on) => on,
                Err(why) => {
                    failure = Some(format!("{}: {why}", variant.label()));
                    break;
                }
            };
            let _ = events.send(Update::Status { message: format!("scoring on {}", variant.label()) });

            let (progress, mut ticks) = tokio::sync::mpsc::channel(32);
            let relay = {
                let events = events.clone();
                tokio::spawn(async move {
                    while let Some((done, total)) = ticks.recv().await {
                        let _ = events.send(Update::Progress { done, total });
                    }
                })
            };
            let started = Instant::now();
            let scored = engine.score(&on, text.clone(), window, progress).await;
            relay.abort();

            match scored {
                Ok(p) => {
                    let s = Scored {
                        variant: variant.label(),
                        tokens: p.tokens,
                        scored: p.scored,
                        perplexity: p.perplexity,
                        bits_per_token: p.bits_per_token,
                        took_secs: started.elapsed().as_secs_f64(),
                    };
                    scores.push(serde_json::to_value(&s).unwrap_or_default());
                    let _ = events.send(Update::Scored(s));
                }
                Err(why) => {
                    failure = Some(format!("{}: {why}", variant.label()));
                    break;
                }
            }
        }

        let result = serde_json::json!({ "dataset": dataset, "scores": scores });
        match (failure, cancel.load(Ordering::Relaxed)) {
            (Some(why), _) => worker.finish(id, "failed", Some(why), Some(result)),
            (None, true) => worker.finish(id, "cancelled", None, Some(result)),
            (None, false) => worker.finish(id, "done", None, Some(result)),
        }
    });

    jobs.get(id)?.ok_or_else(|| "the job vanished as it started".into())
}

/// What a benchmark was asked for.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchParams {
    pub prompt: String,
    pub rounds: usize,
    pub tokens: usize,
    pub seed: u64,
    /// Keep what each variant wrote. A side-by-side comparison wants it; a
    /// measurement over five rounds does not, because four of them are the
    /// same text again.
    #[serde(default)]
    pub keep_text: bool,
}

/// Time every variant, several rounds each, interleaved.
pub fn bench(
    jobs: &Arc<Jobs>,
    engine: &Arc<Scheduler>,
    p: BenchParams,
    variants: Vec<(Variant, Backend)>,
    owner: Option<i64>,
) -> Res<crate::jobs::Job> {
    let label = match variants.len() {
        1 => variants[0].0.label(),
        n => format!("{n} variants, {} rounds", p.rounds),
    };
    let params = serde_json::json!({
        "prompt": p.prompt,
        "rounds": p.rounds,
        "tokens": p.tokens,
        "seed": p.seed,
        "keep_text": p.keep_text,
        "variants": variants.iter().map(|(v, _)| v).collect::<Vec<_>>(),
    });
    let id = jobs.create("bench", &label, &params, owner)?;
    let (cancel, events) = jobs.register(id);
    let worker = Arc::clone(jobs);
    let engine = Arc::clone(engine);

    tokio::spawn(async move {
        let mut failure: Option<String> = None;
        let total = p.rounds * variants.len();
        let mut done = 0usize;

        'rounds: for round in 1..=p.rounds as i64 {
            for (variant, backend) in &variants {
                if cancel.load(Ordering::Relaxed) {
                    break 'rounds;
                }
                let on = match switch(&engine, &events, variant, *backend).await {
                    Ok(on) => on,
                    Err(why) => {
                        failure = Some(format!("{}: {why}", variant.label()));
                        break 'rounds;
                    }
                };
                let _ = events.send(Update::Status {
                    message: format!("round {round} · {}", variant.label()),
                });
                let sampling = Sampling {
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 1.0,
                    seed: Some(p.seed),
                    max_tokens: p.tokens,
                };
                // Raw text, not a chat turn: a benchmark measures the
                // arithmetic, and a template would make the same prompt a
                // different number of tokens on every model.
                let pieces = engine.complete(&on, p.prompt.clone(), sampling, 0, true);
                let (text, stats) = match once(pieces).await {
                    // Not a sample: a rate over no tokens is a zero that would
                    // sit in the median looking like a very slow model.
                    Ok((_, stats)) if stats.generated_tokens == 0 => {
                        failure = Some(format!(
                            "{}: {WROTE_NOTHING}. An instruct model does that with a \
                             finished question; a benchmark wants text to continue, \
                             such as the start of a sentence",
                            variant.label()
                        ));
                        break 'rounds;
                    }
                    Ok(answer) => answer,
                    Err(why) => {
                        failure = Some(format!("{}: {why}", variant.label()));
                        break 'rounds;
                    }
                };
                let timing = Timing {
                    variant: variant.label(),
                    round,
                    decode_per_sec: stats.tokens_per_sec() as f64,
                    prefill_per_sec: stats.prompt_tokens as f64
                        / stats.prefill_secs.max(1e-6) as f64,
                    ttft_millis: stats.prefill_secs as f64 * 1000.0,
                    generated_tokens: stats.generated_tokens as i64,
                    text: p.keep_text.then_some(text),
                };
                let _ = worker.db.with(|c| {
                    c.execute(
                        "INSERT OR REPLACE INTO bench_samples
                         (job, variant, round, decode_per_sec, prefill_per_sec,
                          ttft_millis, generated_tokens, text)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                        params![
                            id,
                            timing.variant,
                            timing.round,
                            timing.decode_per_sec,
                            timing.prefill_per_sec,
                            timing.ttft_millis,
                            timing.generated_tokens,
                            timing.text
                        ],
                    )
                });
                let _ = events.send(Update::Timing(timing));
                done += 1;
                let _ = events.send(Update::Progress { done, total });
            }
        }

        let summary =
            worker.db.with(|c| summarise(c, id)).unwrap_or(serde_json::Value::Null);
        let result = serde_json::json!({ "rounds": p.rounds, "variants": summary });
        match (failure, cancel.load(Ordering::Relaxed)) {
            (Some(why), _) => worker.finish(id, "failed", Some(why), Some(result)),
            (None, true) => worker.finish(id, "cancelled", None, Some(result)),
            (None, false) => worker.finish(id, "done", None, Some(result)),
        }
    });

    jobs.get(id)?.ok_or_else(|| "the job vanished as it started".into())
}

/// Median and range per variant, computed from the samples rather than kept
/// running — five numbers are cheaper to sort than to summarise carefully.
///
/// The range is the whole spread, not a standard deviation. With five samples
/// a standard deviation is a number with more decimal places than meaning,
/// and "32 to 41" is what somebody actually needs to know before believing a
/// difference.
pub fn summarise(c: &rusqlite::Connection, job: i64) -> rusqlite::Result<serde_json::Value> {
    let mut q = c.prepare(
        "SELECT variant, decode_per_sec, ttft_millis FROM bench_samples
         WHERE job = ?1 ORDER BY variant, round",
    )?;
    let rows: Vec<(String, f64, f64)> =
        q.query_map([job], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<Result<_, _>>()?;

    let mut out: Vec<serde_json::Value> = Vec::new();
    let mut names: Vec<String> = rows.iter().map(|(v, _, _)| v.clone()).collect();
    names.dedup();
    for name in names {
        let mut decode: Vec<f64> =
            rows.iter().filter(|(v, _, _)| *v == name).map(|(_, d, _)| *d).collect();
        let mut ttft: Vec<f64> =
            rows.iter().filter(|(v, _, _)| *v == name).map(|(_, _, t)| *t).collect();
        decode.sort_by(f64::total_cmp);
        ttft.sort_by(f64::total_cmp);
        out.push(serde_json::json!({
            "variant": name,
            "runs": decode.len(),
            "decode_median": middle(&decode),
            "decode_low": decode.first(),
            "decode_high": decode.last(),
            "ttft_median": middle(&ttft),
            "ttft_low": ttft.first(),
            "ttft_high": ttft.last(),
        }));
    }
    Ok(serde_json::Value::Array(out))
}

/// The middle of a sorted list; with an even count, the lower of the two.
///
/// Not the mean of the middle pair: every number here was measured, and an
/// average of two of them is a reading nothing produced.
fn middle(sorted: &[f64]) -> Option<f64> {
    sorted.get(sorted.len() / 2).copied()
}

/// Whether anything is on the machine that would make a measurement a lie.
///
/// The "idle check first" of the protocol, as a refusal rather than a
/// footnote: a benchmark run against a machine that is also training reports
/// numbers about the training run.
pub fn machine_is_busy(jobs: &Jobs) -> Option<String> {
    jobs.live_kind(&["train", "bench", "eval"])
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::db::Db;
    use std::path::PathBuf;

    /// Keep the tests out of the developer's own configuration.
    ///
    /// Loading a model sets the active one, and the active one lives in
    /// `$XDG_CONFIG_HOME/kvad/state.json` — which without this is the real
    /// file, so a test run would leave `kvad run` pointing at a temporary
    /// directory it had already deleted. It did, once, which is why this
    /// exists.
    ///
    /// Once for the whole process: the tests are threads in one binary, and
    /// an environment variable is process-wide however many of them there
    /// are.
    fn somewhere_harmless() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let dir = std::env::temp_dir().join(format!("kvad-test-config-{}", std::process::id()));
            let _ = std::fs::create_dir_all(&dir);
            std::env::set_var("XDG_CONFIG_HOME", &dir);
            // And the quantised weights of a model loaded at q8, which would
            // otherwise be written beside the developer's real ones under a
            // name nobody can tell apart from theirs.
            std::env::set_var("KVAD_QUANT_CACHE", dir.join("quant"));
        });
    }

    /// A memory budget no test model comes near.
    pub fn roomy() -> crate::memory::Budget {
        crate::memory::Budget { total: 1 << 40, context: 1024 }
    }

    /// A real model, trained here, in about a second.
    ///
    /// The plan asks for the API tests to run against a `nervus`-trained
    /// model rather than a downloaded one, so that CI needs no network and no
    /// gigabytes. Everything below this line is the real engine: the real
    /// loader, the real scheduler, the real tokenizer.
    pub fn tiny_model(name: &str) -> PathBuf {
        trained_on(name, &"the cat sat on the mat. ".repeat(40)).0
    }

    /// A tiny *instruct* model: trained on turns, with a template that writes
    /// them and a character that ends them.
    ///
    /// Its whole world is `u:the cat?|a:on the mat.|` — a user turn, an
    /// answer, and `|` as end-of-turn. So, like a real instruct model, it
    /// answers when it is asked through its template, and when handed the
    /// bare question it reads a finished user turn and ends it at once.
    pub fn tiny_instruct_model(name: &str) -> PathBuf {
        let (dir, tok) = trained_on(name, &"u:the cat?|a:on the mat.|".repeat(40));
        let template = "{% for m in messages %}u:{{ m['content'] }}|{% endfor %}\
                        {% if add_generation_prompt %}a:{% endif %}";
        let config = serde_json::json!({ "chat_template": template });
        std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
        let end = tok.encode("|").unwrap()[0];
        let generation = serde_json::json!({ "eos_token_id": end });
        std::fs::write(dir.join("generation_config.json"), generation.to_string()).unwrap();
        dir
    }

    /// Train a small GPT on `text` and save it where the loader can read it.
    fn trained_on(name: &str, text: &str) -> (PathBuf, nervus::text::CharTokenizer) {
        somewhere_harmless();
        use nervus::model::{Gpt, GptConfig};
        use nervus::optim::AdamW;
        use nervus::rng::Rng;
        use nervus::text::{train_step, CharTokenizer};

        let tok = CharTokenizer::from_text(text);
        let tokens = tok.encode(text).unwrap();
        let config =
            GptConfig { vocab: tok.vocab(), context: 32, d_model: 32, n_heads: 4, n_layers: 2 };

        let mut rng = Rng::new(3);
        let mut model = Gpt::new(config, &mut rng);
        let mut opt = AdamW::new(3e-3);
        for _ in 0..300 {
            train_step(&mut model, &mut opt, &tokens, 4, &mut rng);
        }

        let dir = std::env::temp_dir()
            .join(format!("kvad-serve-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        nervus::checkpoint::save(&dir, &mut model).unwrap();
        tok.save(&dir).unwrap();
        (dir, tok)
    }

    /// Wait for a job to reach a terminal state, or give up.
    pub async fn finished(jobs: &Arc<Jobs>, id: i64) -> crate::jobs::Job {
        for _ in 0..600 {
            let job = jobs.get(id).unwrap().expect("the job vanished");
            if !job.live() {
                return job;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        panic!("the job never finished");
    }

    fn variant(model: &str, backend: &str) -> Variant {
        Variant { model: model.into(), backend: backend.into() }
    }

    /// Every variant is checked before anything is loaded, because the
    /// alternative is a run of three that fails on the third after ten
    /// minutes of the first two.
    #[test]
    fn a_comparison_is_checked_before_it_starts() {
        assert!(resolve(&[]).unwrap_err().contains("at least one"));
        assert!(resolve(&[variant("", "cpu-q8")]).unwrap_err().contains("no model"));

        let unknown = resolve(&[variant("a/b", "cpu-q9")]).unwrap_err();
        assert!(unknown.contains("cpu-q9") && unknown.contains("cpu-q8"), "{unknown}");

        // The same thing twice is a comparison of nothing, and it would write
        // its two results to one primary key.
        let twice = resolve(&[variant("a/b", "cpu-q8"), variant("a/b", "cpu-q8")]).unwrap_err();
        assert!(twice.contains("twice"), "{twice}");

        // The same model at two precisions is the point of the page.
        let both = resolve(&[variant("a/b", "cpu-q8"), variant("a/b", "cpu-f32")]).unwrap();
        assert_eq!(both.len(), 2);
        assert_eq!(both[0].0.label(), "a/b · cpu-q8");
    }

    /// `contains` is lenient about case and whitespace; `equals` is not
    /// lenient about anything else. A match kind nobody recognises falls back
    /// to `contains`, because a typo should not fail every case in a suite.
    #[test]
    fn a_case_passes_on_what_it_says_it_wants() {
        assert!(matches("  Paris, obviously.  ", "paris", "contains"));
        assert!(!matches("Lyon", "paris", "contains"));
        assert!(matches("Paris", "  paris ", "equals"));
        assert!(!matches("Paris, obviously", "paris", "equals"));
        assert!(matches("Paris, obviously", "paris", "who knows"));
    }

    /// The median and the range come from the samples, and the median is one
    /// of them rather than an average of two.
    #[test]
    fn a_summary_is_made_of_numbers_that_were_measured() {
        let db = Db::in_memory().unwrap();
        db.with(|c| {
            c.execute(
                "INSERT INTO jobs (id, kind, state, label, params)
                 VALUES (1, 'bench', 'done', 'b', '{}')",
                [],
            )?;
            for (round, decode, ttft) in
                [(1, 30.0), (2, 41.0), (3, 32.0), (4, 37.0), (5, 25.0)]
                    .iter()
                    .enumerate()
                    .map(|(i, &(r, d))| (r, d, 10.0 + i as f64))
            {
                c.execute(
                    "INSERT INTO bench_samples
                     (job, variant, round, decode_per_sec, prefill_per_sec, ttft_millis,
                      generated_tokens)
                     VALUES (1, 'a · cpu-q8', ?1, ?2, 100.0, ?3, 64)",
                    params![round, decode, ttft],
                )?;
            }
            Ok(())
        })
        .unwrap();

        let summary = db.with(|c| summarise(c, 1)).unwrap();
        let row = &summary[0];
        assert_eq!(row["variant"], "a · cpu-q8");
        assert_eq!(row["runs"], 5);
        // 25 30 32 37 41 -> the middle one, and the ends.
        assert_eq!(row["decode_median"], 32.0);
        assert_eq!(row["decode_low"], 25.0);
        assert_eq!(row["decode_high"], 41.0);

        // The median of an even count is a measurement, not the average of
        // two of them.
        assert_eq!(middle(&[1.0, 2.0, 3.0, 4.0]), Some(3.0));
        assert_eq!(middle(&[]), None);
    }

    /// A benchmark next to a training run measures the training run, so it is
    /// refused — and the refusal names what is in the way.
    #[test]
    fn a_busy_machine_is_named_rather_than_measured() {
        let jobs = Jobs::new(Db::in_memory().unwrap());
        assert!(machine_is_busy(&jobs).is_none());

        let id = jobs.create("train", "shakespeare", &serde_json::json!({}), None).unwrap();
        let busy = machine_is_busy(&jobs).expect("a training run was not in the way");
        assert!(busy.contains("train") && busy.contains("shakespeare"), "{busy}");

        jobs.finish(id, "done", None, None);
        assert!(machine_is_busy(&jobs).is_none());

        // A download is not in the way: it is waiting on a network, not on
        // the cores the measurement needs.
        jobs.create("pull", "a/b", &serde_json::json!({}), None).unwrap();
        assert!(machine_is_busy(&jobs).is_none());
    }
}

#[cfg(test)]
mod against_a_real_model {
    use super::tests::*;
    use super::*;
    use crate::db::Db;

    /// A suite against a model this test trained: one case it can pass and
    /// one it cannot.
    ///
    /// The model's whole world is "the cat sat on the mat", so greedy
    /// decoding from "the cat" is a known answer — which is what makes this a
    /// test of the eval machinery rather than a test of a language model.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_suite_runs_against_a_model_trained_in_this_test() {
        let dir = tiny_model("suite");
        let jobs = Arc::new(Jobs::new(Db::in_memory().unwrap()));
        let engine = Arc::new(Scheduler::spawn(kvad::service::cpu_loader, roomy()));
        let variants = resolve(&[Variant {
            model: dir.to_string_lossy().into_owned(),
            backend: "cpu-f32".into(),
        }])
        .unwrap();

        let cases = vec![
            Expectation {
                prompt: "the cat".into(),
                expect: "sat on the mat".into(),
                how: None,
            },
            Expectation {
                prompt: "the cat".into(),
                expect: "went to the moon".into(),
                how: None,
            },
        ];
        let job =
            suite(&jobs, &engine, "cats".into(), cases, variants, 16, 1337, None).unwrap();
        let job = finished(&jobs, job.id).await;

        assert_eq!(job.state, "done", "{:?}", job.error);
        assert_eq!(job.result.as_ref().unwrap()["passed"], 1);
        assert_eq!(job.result.as_ref().unwrap()["of"], 2);

        let verdicts = jobs.cases(job.id).unwrap();
        assert_eq!(verdicts.len(), 2);
        assert!(verdicts[0].passed, "got `{}`", verdicts[0].got);
        assert!(!verdicts[1].passed, "got `{}`", verdicts[1].got);
        // The numbers come along with the verdict, because a suite that
        // passes twice as slowly is news.
        assert!(verdicts[0].decode_per_sec.unwrap() > 0.0);
        assert_eq!(verdicts[0].generated_tokens, Some(16));

        // Greedy and seeded: the same case twice gives the same answer, which
        // is the only thing that makes a regression test evidence.
        assert_eq!(verdicts[0].got, verdicts[1].got);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A suite asks an instruct model its questions through the model's own
    /// template.
    ///
    /// It used to send the bare prompt as raw text, and an instruct model
    /// reads `the cat?` as a finished user turn: its first token was
    /// end-of-turn, and every case of every suite came back empty. The model
    /// above has no template, so nothing here noticed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_suite_asks_an_instruct_model_through_its_template() {
        let dir = tiny_instruct_model("instruct");
        let jobs = Arc::new(Jobs::new(Db::in_memory().unwrap()));
        let engine = Arc::new(Scheduler::spawn(kvad::service::cpu_loader, roomy()));
        let variants = resolve(&[Variant {
            model: dir.to_string_lossy().into_owned(),
            backend: "cpu-f32".into(),
        }])
        .unwrap();

        let cases = vec![Expectation {
            prompt: "the cat?".into(),
            expect: "on the mat.".into(),
            how: Some("equals".into()),
        }];
        let job =
            suite(&jobs, &engine, "turns".into(), cases, variants, 16, 1337, None).unwrap();
        let job = finished(&jobs, job.id).await;
        assert_eq!(job.state, "done", "{:?}", job.error);

        let verdicts = jobs.cases(job.id).unwrap();
        assert_eq!(verdicts.len(), 1);
        let v = &verdicts[0];
        assert!(v.passed, "got `{}` from {:?} tokens", v.got, v.generated_tokens);
        // The answer and nothing after it: end-of-turn stopped it.
        assert_eq!(v.generated_tokens, Some(11));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A case the model answered with nothing says so, rather than showing
    /// an empty answer that looks like a page that failed to draw one.
    ///
    /// The model is the instruct one with its template taken away: a base
    /// model with an end token, handed a question it has only ever seen
    /// ended.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_case_answered_with_nothing_says_so() {
        let dir = tiny_instruct_model("silent");
        std::fs::remove_file(dir.join("tokenizer_config.json")).unwrap();
        let jobs = Arc::new(Jobs::new(Db::in_memory().unwrap()));
        let engine = Arc::new(Scheduler::spawn(kvad::service::cpu_loader, roomy()));
        let variants = resolve(&[Variant {
            model: dir.to_string_lossy().into_owned(),
            backend: "cpu-f32".into(),
        }])
        .unwrap();

        // `nothing` is in the note, and a note is not an answer.
        let cases = vec![Expectation {
            prompt: "the cat?".into(),
            expect: "nothing".into(),
            how: None,
        }];
        let job =
            suite(&jobs, &engine, "silent".into(), cases, variants, 16, 1337, None).unwrap();
        let job = finished(&jobs, job.id).await;
        assert_eq!(job.state, "done", "{:?}", job.error);

        let v = &jobs.cases(job.id).unwrap()[0];
        assert_eq!(v.generated_tokens, Some(0));
        assert_eq!(v.got, format!("<{WROTE_NOTHING}>"));
        assert!(!v.passed);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A benchmark whose prompt the model ends at once fails, and says why,
    /// rather than keeping a sample of no tokens at no tokens a second.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_benchmark_of_nothing_is_refused_rather_than_timed() {
        let dir = tiny_instruct_model("bench-silent");
        let jobs = Arc::new(Jobs::new(Db::in_memory().unwrap()));
        let engine = Arc::new(Scheduler::spawn(kvad::service::cpu_loader, roomy()));
        let variants = resolve(&[Variant {
            model: dir.to_string_lossy().into_owned(),
            backend: "cpu-f32".into(),
        }])
        .unwrap();

        // Raw text, so the question arrives as a finished user turn.
        let p = BenchParams {
            prompt: "the cat?".into(),
            rounds: 2,
            tokens: 8,
            seed: 1337,
            keep_text: false,
        };
        let job = bench(&jobs, &engine, p, variants, None).unwrap();
        let job = finished(&jobs, job.id).await;

        assert_eq!(job.state, "failed");
        let why = job.error.unwrap_or_default();
        assert!(why.contains(WROTE_NOTHING) && why.contains("text to continue"), "{why}");
        assert!(jobs.timings(job.id).unwrap().is_empty(), "a sample of nothing was kept");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A benchmark of two rounds: every round visits every variant, every
    /// sample is kept, and the summary is computed from them.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_benchmark_keeps_every_sample_and_summarises_from_them() {
        let dir = tiny_model("bench");
        let jobs = Arc::new(Jobs::new(Db::in_memory().unwrap()));
        let engine = Arc::new(Scheduler::spawn(kvad::service::cpu_loader, roomy()));
        let model = dir.to_string_lossy().into_owned();
        let variants =
            resolve(&[Variant { model: model.clone(), backend: "cpu-f32".into() }]).unwrap();

        let p = BenchParams {
            prompt: "the cat".into(),
            rounds: 3,
            tokens: 8,
            seed: 1337,
            keep_text: true,
        };
        let job = bench(&jobs, &engine, p, variants, None).unwrap();
        let job = finished(&jobs, job.id).await;
        assert_eq!(job.state, "done", "{:?}", job.error);

        let samples = jobs.timings(job.id).unwrap();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples.iter().map(|s| s.round).collect::<Vec<_>>(), [1, 2, 3]);
        for s in &samples {
            assert_eq!(s.generated_tokens, 8);
            assert!(s.decode_per_sec > 0.0, "{s:?}");
            // Every timed generation starts from an empty cache, or the
            // second round would report a time to first token that no first
            // round would ever see.
            assert!(s.ttft_millis > 0.0, "{s:?}");
            assert!(s.text.is_some(), "a comparison run kept no text");
        }
        // Seeded and greedy, so the rounds differ in their timing and in
        // nothing else.
        assert_eq!(samples[0].text, samples[2].text);

        let summary = &job.result.as_ref().unwrap()["variants"][0];
        assert_eq!(summary["runs"], 3);
        let median = summary["decode_median"].as_f64().unwrap();
        let (low, high) =
            (summary["decode_low"].as_f64().unwrap(), summary["decode_high"].as_f64().unwrap());
        assert!(low <= median && median <= high, "{low} {median} {high}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Scoring a text the model did know against one it did not.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn perplexity_runs_as_a_job_and_tells_the_two_texts_apart() {
        let dir = tiny_model("score");
        let jobs = Arc::new(Jobs::new(Db::in_memory().unwrap()));
        let engine = Arc::new(Scheduler::spawn(kvad::service::cpu_loader, roomy()));
        let model = dir.to_string_lossy().into_owned();
        let variants =
            resolve(&[Variant { model, backend: "cpu-f32".into() }]).unwrap();

        let learnt = score(
            &jobs,
            &engine,
            "cats".into(),
            "the cat sat on the mat. ".repeat(20),
            variants.clone(),
            32,
            None,
        )
        .unwrap();
        let learnt = finished(&jobs, learnt.id).await;
        assert_eq!(learnt.state, "done", "{:?}", learnt.error);

        let nonsense = score(
            &jobs,
            &engine,
            "backwards".into(),
            "tam. eht no tas tac eht ".repeat(20),
            variants,
            32,
            None,
        )
        .unwrap();
        let nonsense = finished(&jobs, nonsense.id).await;

        let of = |job: &crate::jobs::Job| {
            job.result.as_ref().unwrap()["scores"][0]["perplexity"].as_f64().unwrap()
        };
        assert!(
            of(&learnt) < of(&nonsense),
            "its own text scored {:.2} and nonsense {:.2}",
            of(&learnt),
            of(&nonsense)
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
