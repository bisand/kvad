-- Long work, and what it left behind.
--
-- A download takes minutes and a training run can take an hour, which is
-- longer than any HTTP request should live and longer than a browser tab is
-- likely to stay open. So the work belongs to the server and a request only
-- watches it: these rows are what somebody comes back to.
CREATE TABLE jobs (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL CHECK (kind IN ('pull', 'train')),
    -- queued -> running -> one of done, failed, cancelled.
    state      TEXT NOT NULL CHECK (state IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    -- A one-line description for a list: the repo being pulled, the model
    -- being trained.
    label      TEXT NOT NULL,
    -- What it was asked to do, as JSON, so that a run can be read back and
    -- repeated without this table growing a column per training flag.
    params     TEXT NOT NULL,
    -- What it produced: a Summary for a training run, null otherwise.
    result     TEXT,
    error      TEXT,
    owner      INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    ended_at   TEXT
) STRICT;

CREATE INDEX jobs_by_created ON jobs (created_at DESC, id DESC);

-- One row per validation checkpoint. These are the loss chart: a run that was
-- watched in a tab yesterday is still a chart today.
--
-- Columns rather than a JSON blob, because averaging JSON is how you end up
-- with a second schema nobody wrote down.
CREATE TABLE train_metrics (
    job            INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    step           INTEGER NOT NULL,
    train_loss     REAL NOT NULL,
    val_loss       REAL NOT NULL,
    chars_per_sec  REAL NOT NULL,
    elapsed_secs   REAL NOT NULL,
    -- True at the step whose model is the one on disk.
    saved          INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (job, step)
) STRICT;

-- What the model wrote at each checkpoint. The part anyone actually watches.
CREATE TABLE train_samples (
    job  INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (job, step)
) STRICT;

-- Text files to train on.
--
-- The file on disk is the truth, as it is for models; this table holds what a
-- listing wants to show without reading every file again. Both counts are
-- taken once at upload, because how much text there is and how many distinct
-- characters are in it is the only interesting thing about a corpus before it
-- is trained on. (`distinct` is a keyword, hence the longer column name.)
CREATE TABLE datasets (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    bytes      INTEGER NOT NULL,
    characters INTEGER NOT NULL,
    distinct_chars INTEGER NOT NULL,
    owner      INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;
