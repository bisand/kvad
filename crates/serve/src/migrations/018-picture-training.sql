-- Training on pictures: a LoRA for an image model, from the web UI.
--
-- A run of it is a job of a new kind, `tune`, so the jobs table learns the
-- word the way 006 and 007 taught it theirs: SQLite cannot alter a CHECK,
-- and the table is rebuilt. `db::migrate` turns foreign keys off around
-- every migration for exactly this, so the charts hanging off a job stay.
CREATE TABLE jobs_new (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL CHECK (kind IN ('pull', 'train', 'eval', 'bench', 'crawl', 'tune')),
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

-- A dataset is text, as every one before this was, or `pictures`: a folder
-- under `datasets/` of pictures, each with its caption in a `.txt` of the
-- same name beside it, which is what `kvad-gpu tune` reads. The folder is
-- the truth, as the file is for a text: how many pictures there are and
-- which have no caption are counted from it when asked, and not kept here
-- to go stale as files are added one at a time.
ALTER TABLE datasets ADD COLUMN kind TEXT NOT NULL DEFAULT 'text';

-- What a run drew at each measurement: one picture a prompt, a file under
-- the data directory's `tune-samples/JOB/`. The prompts are in the job's
-- params, numbered from 0 as they were given; step 0 is the model without
-- the LoRA, which is what the rest of a prompt's row is held against.
CREATE TABLE train_pictures (
    job    INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    step   INTEGER NOT NULL,
    prompt INTEGER NOT NULL,
    file   TEXT NOT NULL,
    PRIMARY KEY (job, step, prompt)
) STRICT;
