-- Runtime settings: everything somebody can change from the UI while the
-- server is running. What has to be known before the server starts is in
-- kvad.toml instead; see `config`.
--
-- Key/value rather than a column per setting, because the alternative is a
-- migration every time the UI grows a checkbox. The value is JSON so that a
-- setting can be a number, a string or a list without three columns to say
-- which.
CREATE TABLE settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;
