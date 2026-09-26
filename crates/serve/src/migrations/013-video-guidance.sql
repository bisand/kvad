-- A video can be made by a guided pipeline: LTX-2.5's dev model, which a
-- request asks for by giving steps, a guidance scale or a negative prompt.
--
-- All three NULL for a video made without guidance; otherwise what it was
-- made with, blanks filled, except a negative prompt left to the model,
-- which stays NULL.
ALTER TABLE videos ADD COLUMN steps INTEGER;
ALTER TABLE videos ADD COLUMN guidance REAL;
ALTER TABLE videos ADD COLUMN negative_prompt TEXT;
