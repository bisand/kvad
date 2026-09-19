-- Who may use this server, and what they may do.
--
-- The tables exist whatever `auth.mode` says. With mode "none" they stay
-- empty and nothing reads them; turning auth on is then a config change and a
-- first sign-in rather than a migration.
CREATE TABLE users (
    id            INTEGER PRIMARY KEY,
    -- NOCASE so that `Ada` and `ada` cannot both be created and then be told
    -- apart only by whoever typed them.
    name          TEXT NOT NULL UNIQUE COLLATE NOCASE,
    -- An argon2id PHC string, or NULL for somebody who signs in another way
    -- — through an identity provider, say. NULL is not "any password will
    -- do"; `secret::verify_password` refuses an empty hash.
    password_hash TEXT,
    role          TEXT NOT NULL CHECK (role IN ('admin', 'user')),
    -- What an identity provider's `email` claim is matched against. NULL for
    -- an account that only ever signs in with a password.
    email         TEXT UNIQUE COLLATE NOCASE,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at  TEXT
) STRICT;

-- Server-side sessions, so that signing somebody out actually signs them out.
-- A self-contained token — a JWT in a cookie — cannot be revoked before it
-- expires, and "revoke this session" is a thing a settings page has to be able
-- to do.
CREATE TABLE sessions (
    -- sha256(token), never the token. Somebody who reads this table has a
    -- list of hashes, not a set of live sessions.
    token_hash TEXT PRIMARY KEY,
    user       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL,
    -- For the list in Settings, so a session can be recognised before it is
    -- revoked. Client-supplied and shown as text, never interpreted.
    user_agent TEXT
) STRICT;

CREATE INDEX sessions_by_user ON sessions (user);

-- Bearer tokens for scripts and for anything speaking the OpenAI API. Hashed
-- the same way and for the same reason; shown once, when made.
CREATE TABLE api_keys (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    token_hash   TEXT NOT NULL UNIQUE,
    -- The first few characters of the key, so a list can say which key is
    -- which without the list being a list of keys.
    prefix       TEXT NOT NULL,
    user         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    last_used_at TEXT
) STRICT;

CREATE INDEX api_keys_by_user ON api_keys (user);

-- Conversations belong to somebody, once there is somebody for them to belong
-- to. NULL means "made before anyone signed in", which is every conversation
-- from a server that has only ever run with mode "none". Those are adopted by
-- the first account created, rather than left visible to whoever signs up
-- next: see `users::bootstrap`.
ALTER TABLE conversations ADD COLUMN owner INTEGER REFERENCES users(id) ON DELETE CASCADE;

CREATE INDEX conversations_by_owner ON conversations (owner);
