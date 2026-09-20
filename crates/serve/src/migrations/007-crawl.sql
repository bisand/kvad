-- Corpora that came from a website rather than from a file picker.
--
-- Reading a documentation site takes minutes and hundreds of requests, which
-- is a job, so the first act here is the same as 006's: let the jobs table
-- say a new word. Same rebuild, same reason — SQLite cannot alter a CHECK.
CREATE TABLE jobs_new (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL CHECK (kind IN ('pull', 'train', 'eval', 'bench', 'crawl')),
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

-- Where the text came from: the address a crawl started at, or null for a
-- file somebody uploaded.
--
-- One column rather than a read of the manifest beside every file, for the
-- reason the counts are columns: a listing should not have to open anything.
-- The manifest — `datasets/<name>.crawl.json` — holds the rest of it, every
-- page in the order it was fetched, and is written and deleted with the text.
ALTER TABLE datasets ADD COLUMN source TEXT;
