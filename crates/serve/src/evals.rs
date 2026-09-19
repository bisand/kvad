//! Two ways of asking whether a model is any good, and one way of asking
//! whether it got worse.
//!
//! **Perplexity** is a number: how surprised a model is by text it did not
//! write. It compares quantisations honestly, because they are the same model
//! and the question is what the arithmetic cost — and it is the only measure
//! here that needs no opinion about what a good answer looks like.
//!
//! **Prompt suites** are the regression test. A case is a prompt and
//! something the answer has to contain; a suite is a list of them; a run is
//! that suite against one or more variants. Cases run greedily and seeded, so
//! a failure is a change in the model rather than a change in the dice.
//!
//! Both are [`crate::jobs`] jobs, because both mean loading models one after
//! another — see [`crate::compare`] for why that is the shape of it.

use crate::api::{blocking, Fail};
use crate::auth::{Admin, Identity, State};
use crate::compare::{self, Expectation, Variant};
use crate::jobs::{Case, Job, Scored};
use axum::extract::{Path, State as St};
use axum::routing::{get, post};
use axum::{Json, Router};
use rusqlite::{params, OptionalExtension};
use serde_json::json;

pub fn routes() -> Router<State> {
    Router::new()
        .route("/api/evals/suites", get(list_suites).post(create_suite))
        .route("/api/evals/suites/{id}", axum::routing::patch(update_suite).delete(delete_suite))
        .route("/api/evals/run", post(run_suite))
        .route("/api/evals/perplexity", post(run_perplexity))
        .route("/api/evals/runs", get(list_runs))
        .route("/api/evals/runs/{id}", get(results))
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Serialize)]
pub struct Suite {
    pub id: i64,
    pub name: String,
    pub cases: Vec<Expectation>,
    pub created_at: String,
    pub updated_at: String,
}

fn suite_row(r: &rusqlite::Row) -> rusqlite::Result<Suite> {
    let raw: String = r.get("cases")?;
    Ok(Suite {
        id: r.get("id")?,
        name: r.get("name")?,
        // A suite whose JSON cannot be read is an empty suite rather than a
        // failed listing: one bad row should not hide the others.
        cases: serde_json::from_str(&raw).unwrap_or_default(),
        created_at: r.get("created_at")?,
        updated_at: r.get("updated_at")?,
    })
}

async fn list_suites(_: Identity, St(state): St<State>) -> Result<Json<Vec<Suite>>, Fail> {
    let db = state.db.clone();
    blocking(move || {
        db.with(|c| {
            let mut q = c.prepare(
                "SELECT id, name, cases, created_at, updated_at FROM eval_suites ORDER BY name",
            )?;
            let rows = q.query_map([], suite_row)?.collect();
            rows
        })
    })
    .await
    .map(Json)
}

#[derive(serde::Deserialize)]
pub struct SuiteBody {
    name: String,
    cases: Vec<Expectation>,
}

/// Cases as they will be stored, with the obvious mistakes named.
fn checked(cases: &[Expectation]) -> Result<String, String> {
    if cases.is_empty() {
        return Err("a suite needs at least one case".into());
    }
    for (i, c) in cases.iter().enumerate() {
        if c.prompt.trim().is_empty() {
            return Err(format!("case {} has no prompt", i + 1));
        }
        if c.expect.trim().is_empty() {
            return Err(format!("case {} says nothing about what the answer should be", i + 1));
        }
        if let Some(how) = &c.how {
            if !matches!(how.as_str(), "contains" | "equals") {
                return Err(format!(
                    "case {} matches by `{how}`; there are two ways, `contains` and `equals`",
                    i + 1
                ));
            }
        }
    }
    serde_json::to_string(cases).map_err(|e| e.to_string())
}

async fn create_suite(
    who: Admin,
    St(state): St<State>,
    Json(body): Json<SuiteBody>,
) -> Result<Json<Suite>, Fail> {
    let db = state.db.clone();
    let owner = who.0.id;
    blocking(move || {
        let name = body.name.trim().to_string();
        if name.is_empty() {
            return Err("a suite needs a name".into());
        }
        let cases = checked(&body.cases)?;
        let id = db.with(|c| {
            c.execute(
                "INSERT INTO eval_suites (name, cases, owner) VALUES (?1, ?2, ?3)",
                params![name, cases, owner],
            )?;
            Ok(c.last_insert_rowid())
        })?;
        db.with(|c| {
            c.query_row(
                "SELECT id, name, cases, created_at, updated_at FROM eval_suites WHERE id = ?1",
                [id],
                suite_row,
            )
        })
    })
    .await
    .map(Json)
    .map_err(|e| Fail::bad(e.1))
}

#[derive(serde::Deserialize)]
pub struct SuitePatch {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    cases: Option<Vec<Expectation>>,
}

async fn update_suite(
    _: Admin,
    St(state): St<State>,
    Path(id): Path<i64>,
    Json(body): Json<SuitePatch>,
) -> Result<Json<Suite>, Fail> {
    let db = state.db.clone();
    blocking(move || {
        if let Some(name) = &body.name {
            let name = name.trim();
            if name.is_empty() {
                return Err("a suite needs a name".into());
            }
            db.with(|c| {
                c.execute("UPDATE eval_suites SET name = ?2 WHERE id = ?1", params![id, name])
            })?;
        }
        if let Some(cases) = &body.cases {
            let cases = checked(cases)?;
            db.with(|c| {
                c.execute("UPDATE eval_suites SET cases = ?2 WHERE id = ?1", params![id, cases])
            })?;
        }
        db.with(|c| {
            c.execute(
                "UPDATE eval_suites SET updated_at = datetime('now') WHERE id = ?1",
                [id],
            )?;
            c.query_row(
                "SELECT id, name, cases, created_at, updated_at FROM eval_suites WHERE id = ?1",
                [id],
                suite_row,
            )
            .optional()
        })?
        .ok_or_else(|| format!("there is no suite {id}").into())
    })
    .await
    .map(Json)
    .map_err(|e| Fail::bad(e.1))
}

async fn delete_suite(
    _: Admin,
    St(state): St<State>,
    Path(id): Path<i64>,
) -> Result<Json<serde_json::Value>, Fail> {
    let db = state.db.clone();
    let gone = blocking(move || db.with(|c| c.execute("DELETE FROM eval_suites WHERE id = ?1", [id])))
        .await?;
    match gone {
        0 => Err(Fail::missing(format!("there is no suite {id}"))),
        _ => Ok(Json(json!({ "deleted": id }))),
    }
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct RunRequest {
    suite: i64,
    variants: Vec<Variant>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
}

async fn run_suite(
    who: Admin,
    St(state): St<State>,
    Json(body): Json<RunRequest>,
) -> Result<Json<Job>, Fail> {
    let variants = compare::resolve(&body.variants).map_err(Fail::bad)?;
    let db = state.db.clone();
    let jobs = state.jobs.clone();
    let engine = state.engine.clone();
    let owner = who.0.id;

    // Read the suite before starting anything: the one failure that should be
    // an error rather than a job that fails a second later.
    let suite = blocking(move || {
        db.with(|c| {
            c.query_row(
                "SELECT id, name, cases, created_at, updated_at FROM eval_suites WHERE id = ?1",
                [body.suite],
                suite_row,
            )
            .optional()
        })
    })
    .await?
    .ok_or_else(|| Fail::missing(format!("there is no suite {}", body.suite)))?;

    if let Some(busy) = compare::machine_is_busy(&jobs) {
        return Err(Fail::bad(format!("{busy} is running; wait for it or stop it")));
    }
    compare::suite(
        &jobs,
        &engine,
        suite.name,
        suite.cases,
        variants,
        body.max_tokens.unwrap_or(64).clamp(1, 1024),
        body.seed.unwrap_or(1337),
        owner,
    )
    .map(Json)
    .map_err(|e| Fail::bad(e.to_string()))
}

#[derive(serde::Deserialize)]
pub struct PerplexityRequest {
    dataset: i64,
    variants: Vec<Variant>,
    /// Tokens per scoring window. Clamped to the model's context by the
    /// engine, which is the only place that knows what that is.
    #[serde(default)]
    window: Option<usize>,
}

async fn run_perplexity(
    who: Admin,
    St(state): St<State>,
    Json(body): Json<PerplexityRequest>,
) -> Result<Json<Job>, Fail> {
    let variants = compare::resolve(&body.variants).map_err(Fail::bad)?;
    let db = state.db.clone();
    let jobs = state.jobs.clone();
    let engine = state.engine.clone();
    let owner = who.0.id;

    let (dataset, text) = blocking(move || crate::datasets::read(&db, body.dataset))
        .await
        .map_err(|e| Fail::missing(e.1))?;

    if let Some(busy) = compare::machine_is_busy(&jobs) {
        return Err(Fail::bad(format!("{busy} is running; wait for it or stop it")));
    }
    compare::score(
        &jobs,
        &engine,
        dataset.name,
        text,
        variants,
        body.window.unwrap_or(512).clamp(2, 8192),
        owner,
    )
    .map(Json)
    .map_err(|e| Fail::bad(e.to_string()))
}

async fn list_runs(_: Identity, St(state): St<State>) -> Result<Json<Vec<Job>>, Fail> {
    let jobs = state.jobs.clone();
    blocking(move || jobs.db.with(|c| kind_rows(c, "eval", 50))).await.map(Json)
}

/// Jobs of one kind, newest first. The listing both eval and bench pages use.
pub fn kind_rows(c: &rusqlite::Connection, kind: &str, limit: i64) -> rusqlite::Result<Vec<Job>> {
    let mut q = c.prepare(
        "SELECT id, kind, state, label, params, result, error,
                created_at, started_at, ended_at
         FROM jobs WHERE kind = ?1 ORDER BY id DESC LIMIT ?2",
    )?;
    let rows = q.query_map(params![kind, limit], crate::jobs::row)?.collect();
    rows
}

/// Everything a finished run has to show: the job, its per-case verdicts, and
/// the scores if it was a perplexity run.
#[derive(serde::Serialize)]
pub struct Results {
    #[serde(flatten)]
    job: Job,
    cases: Vec<Case>,
    scores: Vec<Scored>,
}

async fn results(
    _: Identity,
    St(state): St<State>,
    Path(id): Path<i64>,
) -> Result<Json<Results>, Fail> {
    let jobs = state.jobs.clone();
    let found = blocking(move || {
        let Some(job) = jobs.get(id)? else { return Ok(None) };
        let cases = jobs.cases(id)?;
        // Perplexity has no rows of its own — one number per variant is the
        // whole of it, and it lives in the job's result.
        let scores = job
            .result
            .as_ref()
            .and_then(|r| r.get("scores"))
            .and_then(|s| serde_json::from_value::<Vec<Scored>>(s.clone()).ok())
            .unwrap_or_default();
        Ok(Some(Results { job, cases, scores }))
    })
    .await?;
    found.map(Json).ok_or_else(|| Fail::missing(format!("there is no run {id}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(prompt: &str, expect: &str, how: Option<&str>) -> Expectation {
        Expectation {
            prompt: prompt.into(),
            expect: expect.into(),
            how: how.map(str::to_string),
        }
    }

    /// A suite is refused for the things that would make its results
    /// meaningless, and the refusal says which case.
    #[test]
    fn a_suite_is_checked_when_it_is_written_rather_than_when_it_is_run() {
        assert!(checked(&[]).unwrap_err().contains("at least one case"));
        assert!(checked(&[case("  ", "paris", None)]).unwrap_err().contains("case 1 has no prompt"));

        // A case with nothing to check is a prompt, not a test.
        let vague = checked(&[case("capital of France?", " ", None)]).unwrap_err();
        assert!(vague.contains("case 1") && vague.contains("should be"), "{vague}");

        // A match kind that does not exist would silently fall back to
        // `contains` at run time, so it is caught at the form instead.
        let typo = checked(&[case("q", "a", Some("countains"))]).unwrap_err();
        assert!(typo.contains("countains") && typo.contains("`contains`"), "{typo}");

        // And a good one round-trips.
        let stored = checked(&[case("capital of France?", "Paris", Some("contains"))]).unwrap();
        let back: Vec<Expectation> = serde_json::from_str(&stored).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].expect, "Paris");
        assert_eq!(back[0].how.as_deref(), Some("contains"));
    }
}
