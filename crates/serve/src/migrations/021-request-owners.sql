-- Whose a generation was, and what it read as well as what it wrote.
--
-- A shared server gets asked two things this table could not answer: who is
-- using it, and why it was slow at three o'clock. Both want the account
-- beside the tokens. The reason to add it before anybody asks is that the
-- answer only exists from here forward: a row written without its owner
-- cannot be given one later.
--
-- Null is a row from before this, a request that is not a generation, or a
-- server with no accounts — where there is one person and nothing to tell
-- apart.
ALTER TABLE requests ADD COLUMN owner INTEGER;
-- `tokens` is what was generated. A long prompt costs the engine too, and
-- what of it came out of the cache is the difference between a conversation
-- and the same conversation read again from the start.
ALTER TABLE requests ADD COLUMN prompt_tokens INTEGER;
ALTER TABLE requests ADD COLUMN cached_tokens INTEGER;

CREATE INDEX requests_by_owner ON requests (owner, at DESC);
