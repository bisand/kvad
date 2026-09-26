-- A video's length can be the model's choice: LTX-2.5's duration head reads
-- the prompt and says how long its clip should be, when a request gives no
-- length.
--
-- 1 when the model chose it. `frames` is 0 until it has, which is after the
-- text phase, and what it chose after that.
ALTER TABLE videos ADD COLUMN chosen INTEGER NOT NULL DEFAULT 0;
