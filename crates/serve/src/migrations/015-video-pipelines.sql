-- LTX-2.5 has two unguided pipelines: `fast`, its two distilled stages, and
-- `dfr`, the reference's production pipeline, which also makes more than
-- 30 fps by doubling a clip's frame rate.
--
-- The one that made the video; NULL for a guided video, a model with no
-- choice, and every video made before there was one, all of which were fast.
ALTER TABLE videos ADD COLUMN pipeline TEXT;
