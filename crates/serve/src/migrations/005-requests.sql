-- What the server was asked, and how long it took.
--
-- The live view of this is a ring buffer in memory; see `metrics`. This table
-- is what survives a restart, and it is deliberately the *smaller* of the
-- two in what it keeps per row: a dashboard reads the ring buffer, and what
-- is wanted from here is "what did last Tuesday look like".
--
-- Rows are pruned rather than kept forever. A server answering a request a
-- second would write 86,400 rows a day, which is nothing for SQLite and
-- everything for a person trying to read it.
CREATE TABLE requests (
    id       INTEGER PRIMARY KEY,
    at       TEXT NOT NULL DEFAULT (datetime('now')),
    method   TEXT NOT NULL,
    -- The route pattern where axum knows one — `/api/jobs/{id}` rather than
    -- `/api/jobs/17` — so that a histogram groups what belongs together and
    -- an id cannot turn one route into a thousand.
    path     TEXT NOT NULL,
    status   INTEGER NOT NULL,
    millis   REAL NOT NULL,
    -- Generation only, null elsewhere: what a request cost the engine, kept
    -- beside the timing so a slow reply can be told from a long one.
    tokens          INTEGER,
    decode_per_sec  REAL,
    ttft_millis     REAL
) STRICT;

CREATE INDEX requests_by_time ON requests (at DESC, id DESC);
