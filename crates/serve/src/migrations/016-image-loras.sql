-- A request can apply LoRAs to the model that makes its image.
--
-- The ones that made it, as JSON, `[{"name": …, "scale": …}]`, each by the
-- name it was asked for by; NULL for an image made without one, and every
-- image made before there were any.
ALTER TABLE images ADD COLUMN loras TEXT;
