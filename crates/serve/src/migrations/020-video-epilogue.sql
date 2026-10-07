-- DFR can end with its spatial epilogue: the clip made at half its width and
-- height, then upsampled once more and detailed in spatial tiles at the size
-- asked for.
--
-- Whether it did; 0 for every video made before it could.
ALTER TABLE videos ADD COLUMN epilogue INTEGER NOT NULL DEFAULT 0;
