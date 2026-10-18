-- The devices paired with this server through the tunnel (docs/relay-plan.md,
-- Phase 2): each a key of its own, which the tunnel lets in, and an API key
-- of its own, which only that device may present, and only through the
-- tunnel.
--
-- `admin` is what pairing granted, not what the account is: a device acts as
-- `user` unless it was paired to act as more, so that a lost phone is a lost
-- microphone, not a lost administrator.
CREATE TABLE devices (
    id        INTEGER PRIMARY KEY,
    -- The device's public key, 64 hex digits, proved by every handshake.
    node      TEXT NOT NULL UNIQUE,
    name      TEXT NOT NULL,
    user      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- Its key. Revoking the key alone leaves a device that can reach the
    -- server and must sign in like anybody else.
    api_key   INTEGER UNIQUE REFERENCES api_keys(id) ON DELETE SET NULL,
    admin     INTEGER NOT NULL DEFAULT 0 CHECK (admin IN (0, 1)),
    paired_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT
) STRICT;

CREATE INDEX devices_by_user ON devices (user);
