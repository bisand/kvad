-- A request can apply LoRAs to LTX-2.5's DiT, as to an image model.
--
-- The ones that made the video, as JSON, `[{"name": …, "scale": …}]`, each by
-- the name it was asked for by; NULL for a video made without one, and every
-- video made before there were any.
ALTER TABLE videos ADD COLUMN loras TEXT;
