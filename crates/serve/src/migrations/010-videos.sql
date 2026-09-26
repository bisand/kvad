-- Videos this server made, or is making, one row per video.
--
-- As `images`, with one difference: a video takes minutes, so the row is
-- written when the video is asked for, not when it is done, and it is the
-- job's state as well as its record. `status` is OpenAI's: `queued`,
-- `in_progress`, `completed` or `failed`. The file is `videos/<id>.mp4`,
-- with a poster frame beside it as `videos/<id>.png`; see `videos.rs`.
--
-- AUTOINCREMENT from the start, for the reason 009 gave images: a deleted
-- video's id must not be handed out again while a browser still holds it.
CREATE TABLE videos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    owner           INTEGER REFERENCES users(id) ON DELETE CASCADE,
    model           TEXT NOT NULL,
    backend         TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    frames          INTEGER NOT NULL,
    fps             INTEGER NOT NULL,
    -- A u64 stored bit for bit in SQLite's i64, as in `images`.
    seed            INTEGER NOT NULL,
    -- 1 when the file has sound.
    audio           INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',
    -- From 0 to 1, weighted by what each part of a generation takes.
    progress        REAL NOT NULL DEFAULT 0,
    -- What is running: `text`, `stage 1`, `upsample`, `stage 2`, `decode`.
    phase           TEXT,
    error           TEXT,
    bytes           INTEGER,
    encode_secs     REAL,
    denoise_secs    REAL,
    decode_secs     REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    -- When the engine reached it, which is not when it was asked for if
    -- something was ahead of it.
    started_at      TEXT,
    completed_at    TEXT
) STRICT;

CREATE INDEX videos_by_owner ON videos (owner, id DESC);
