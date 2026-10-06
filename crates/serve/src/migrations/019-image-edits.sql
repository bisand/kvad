-- An image can be made from a picture: `/v1/images/edits`.
--
-- How far the picture was noised before it was drawn over, 0 to 1; NULL for
-- an image made from a prompt alone, and every image made before there were
-- edits. The picture itself is a file beside the image's own, `<id>.input`,
-- as it was sent, so that an edit can be looked at beside what it was made
-- from and made again.
ALTER TABLE images ADD COLUMN strength REAL;

-- Whether a mask kept part of the picture. The mask is `<id>.mask`.
ALTER TABLE images ADD COLUMN masked INTEGER NOT NULL DEFAULT 0;
