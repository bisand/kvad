-- Image ids that are never handed out twice.
--
-- 008 made `id` a plain INTEGER PRIMARY KEY, and SQLite gives such a table
-- the largest id plus one: delete the newest image and the next one gets its
-- id back. The picture is served from `/api/images/<id>.png` as immutable, so
-- a browser that had seen the deleted picture went on showing it in place of
-- the new one. AUTOINCREMENT remembers the largest id ever used instead.
--
-- SQLite cannot add AUTOINCREMENT to a table, so this is the rebuild its
-- documentation describes: a new table, the rows copied with their ids (which
-- also starts `sqlite_sequence` at the largest of them), the old one dropped.
-- Nothing references `images`, so there is no cascade to worry about.
CREATE TABLE images_new (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
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

INSERT INTO images_new (id, owner, model, backend, prompt, negative_prompt, width, height, steps,
                        guidance, seed, bytes, secs, created_at)
    SELECT id, owner, model, backend, prompt, negative_prompt, width, height, steps,
           guidance, seed, bytes, secs, created_at
    FROM images;

DROP TABLE images;
ALTER TABLE images_new RENAME TO images;

CREATE INDEX images_by_owner ON images (owner, id DESC);
