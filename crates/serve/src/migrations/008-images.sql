-- Images this server made, one row per image.
--
-- The picture itself is a PNG file named by the row's id, in `images/` under
-- the data directory; see `images.rs`. A row is what the gallery lists and
-- what makes a picture reproducible — every setting that went into it, the
-- seed above all — and the file is what is shown. Neither is any use without
-- the other, so they are written and deleted together.
--
-- Owned like a conversation: `owner` is the account, NULL when the server
-- runs without accounts, and every query says `owner IS ?`.
CREATE TABLE images (
    id              INTEGER PRIMARY KEY,
    owner           INTEGER REFERENCES users(id) ON DELETE CASCADE,
    model           TEXT NOT NULL,
    backend         TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    negative_prompt TEXT,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    steps           INTEGER NOT NULL,
    guidance        REAL NOT NULL,
    -- A u64 stored bit for bit in SQLite's i64; `images.rs` casts both ways.
    seed            INTEGER NOT NULL,
    bytes           INTEGER NOT NULL,
    -- Encode, denoise and decode together: what the image cost.
    secs            REAL NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;

CREATE INDEX images_by_owner ON images (owner, id DESC);
