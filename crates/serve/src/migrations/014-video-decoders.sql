-- LTX-2.5 has two video decoders for one latent space, and a request can
-- name one: `diffusion`, the reference's default, or `conv`.
--
-- The one that made the video; NULL for a model with no choice, and for
-- every video made before there was one, all of which were `conv`.
ALTER TABLE videos ADD COLUMN decoder TEXT;
