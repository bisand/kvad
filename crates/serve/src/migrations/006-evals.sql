-- Measuring models instead of talking to them.
--
-- Two questions, both of which need the same awkward thing: the engine holds
-- one model at a time, so comparing two means loading each in turn. That
-- makes every comparison long-running, which makes it a job — and so this
-- migration's first act is to let the jobs table say so.

-- SQLite cannot alter a CHECK constraint, so the table is rebuilt: the
-- procedure from the SQLite docs, minus the steps that do not apply — there
-- are no triggers or views on `jobs`, and its one index is recreated below.
--
-- `DROP TABLE jobs` with foreign keys enforced would run every ON DELETE
-- CASCADE first and take `train_metrics` and `train_samples` with it, so the
-- migration runner turns enforcement off around each migration. It cannot be
-- done from in here: `PRAGMA foreign_keys` is a no-op inside a transaction,
-- and every migration runs inside one.
CREATE TABLE jobs_new (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL CHECK (kind IN ('pull', 'train', 'eval', 'bench')),
    state      TEXT NOT NULL CHECK (state IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    label      TEXT NOT NULL,
    params     TEXT NOT NULL,
    result     TEXT,
    error      TEXT,
    owner      INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    ended_at   TEXT
) STRICT;

INSERT INTO jobs_new
SELECT id, kind, state, label, params, result, error, owner, created_at, started_at, ended_at
FROM jobs;

DROP TABLE jobs;
ALTER TABLE jobs_new RENAME TO jobs;
CREATE INDEX jobs_by_created ON jobs (created_at DESC, id DESC);

-- A saved prompt suite: the regression test of this project.
--
-- The cases are one JSON array rather than a table of their own, because they
-- are written, read and edited as a unit — there is no query that wants case
-- 7 of suite 3 — and a form that posts a list should not have to work out
-- which rows to delete.
CREATE TABLE eval_suites (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    -- [{"prompt": "...", "expect": "...", "match": "contains"|"equals"}]
    cases      TEXT NOT NULL,
    owner      INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;

-- What one case produced on one variant. Rows rather than a blob in the job's
-- result, because the interesting view is across variants — "which of these
-- three quantisations still gets case 4 right" — and that is a GROUP BY, not
-- a JSON parse.
CREATE TABLE eval_results (
    job       INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    -- Which model and precision produced it, e.g. `HuggingFaceTB/SmolLM2-135M`
    -- at `cpu-q8`. Stored as text so a run outlives the model it ran on.
    variant   TEXT NOT NULL,
    idx       INTEGER NOT NULL,
    prompt    TEXT NOT NULL,
    expect    TEXT NOT NULL,
    got       TEXT NOT NULL,
    passed    INTEGER NOT NULL,
    -- Kept beside the verdict because a suite that passes twice as slowly is
    -- news, and nobody would run it twice to find out.
    decode_per_sec REAL,
    generated_tokens INTEGER,
    PRIMARY KEY (job, variant, idx)
) STRICT;

-- One timed generation. The protocol from the README made a table: several
-- rounds over several variants, every sample kept, so the median and the
-- range are computed from the numbers rather than remembered as a summary.
--
-- `round` is which pass over the variants this was. Round 1 of every variant
-- happens before round 2 of any of them, which is what makes a slow machine
-- warming up show as a trend instead of as a winner.
CREATE TABLE bench_samples (
    job       INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    variant   TEXT NOT NULL,
    round     INTEGER NOT NULL,
    decode_per_sec   REAL NOT NULL,
    prefill_per_sec  REAL NOT NULL,
    ttft_millis      REAL NOT NULL,
    generated_tokens INTEGER NOT NULL,
    -- What it wrote, for the side-by-side view. Null when the run is a
    -- measurement rather than a comparison: five rounds of the same text is
    -- four copies nobody reads.
    text      TEXT,
    PRIMARY KEY (job, variant, round)
) STRICT;
