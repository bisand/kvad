-- A video can start from a picture: its first frame.
--
-- 1 when it does. The picture is kept as it was sent, beside the video as
-- `videos/<id>.input`, whatever its format; see `videos.rs`.
ALTER TABLE videos ADD COLUMN picture INTEGER NOT NULL DEFAULT 0;
