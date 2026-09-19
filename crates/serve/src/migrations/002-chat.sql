-- Conversations, and the messages in them.
--
-- The server does not need this to answer /v1/chat/completions: that endpoint
-- is stateless, takes the whole transcript in the request, and is exactly
-- what any OpenAI client already sends. What is stored here is what the web
-- UI needs in order to still be there tomorrow, and it is written by the UI
-- around that call rather than by the call itself. One code path through the
-- engine, one place that remembers.
CREATE TABLE conversations (
    id         INTEGER PRIMARY KEY,
    title      TEXT NOT NULL,
    -- The system prompt for this conversation. NULL is "none", which is not
    -- the same as the empty string: an empty system message is still a turn
    -- the model sees.
    system     TEXT,
    -- What was loaded when it was last spoken to. Advisory — the model may be
    -- gone by the time anyone opens this again.
    model      TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;

CREATE TABLE messages (
    id           INTEGER PRIMARY KEY,
    conversation INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content      TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),

    -- What it cost, for assistant messages; NULL everywhere else. Columns
    -- rather than a JSON blob because these are the numbers the Monitoring
    -- and Benchmarks pages will want to average, and averaging JSON is how
    -- you end up with a second schema nobody wrote down.
    model            TEXT,
    backend          TEXT,
    prompt_tokens    INTEGER,
    cached_tokens    INTEGER,
    generated_tokens INTEGER,
    prefill_secs     REAL,
    decode_secs      REAL
) STRICT;

-- Every read of this table is "the messages of one conversation, in order",
-- and `id` ascending is that order: SQLite hands out rowids increasing.
CREATE INDEX messages_by_conversation ON messages (conversation, id);
