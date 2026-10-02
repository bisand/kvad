-- Training an image model from the same place a text model is trained (#77).
--
-- A dataset is text, or a folder of captioned pictures. `kind` says which,
-- and every row from before there was a choice is text. `items` is how many
-- pictures a folder holds; a text has none to count.
ALTER TABLE datasets ADD COLUMN kind TEXT NOT NULL DEFAULT 'text' CHECK (kind IN ('text', 'pictures'));
ALTER TABLE datasets ADD COLUMN items INTEGER;

-- One row per measurement of a LoRA run's validation loss.
--
-- A table of its own rather than more rows of `train_metrics`, because the
-- two loops measure different things. A text run has a training loss at every
-- checkpoint and a speed in characters; a diffusion run has neither at step 0,
-- where the model is measured before it has taken a step, and its training
-- loss says mostly which noise level a step drew (docs/tune.md). The columns
-- that may be absent are allowed to be.
CREATE TABLE tune_metrics (
    job           INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    step          INTEGER NOT NULL,
    val_loss      REAL NOT NULL,
    -- The mean over the steps since the last measurement; none at step 0.
    train_loss    REAL,
    secs_per_step REAL,
    elapsed_secs  REAL NOT NULL,
    -- True at the step whose LoRA is the one on disk.
    saved         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (job, step)
) STRICT;

-- A picture a run drew at a measurement: which prompt, counted from 0, and
-- the file, under the data directory. The file is the picture; the row is
-- how a browser arriving late learns there is one.
CREATE TABLE tune_samples (
    job    INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    step   INTEGER NOT NULL,
    prompt INTEGER NOT NULL,
    file   TEXT NOT NULL,
    PRIMARY KEY (job, step, prompt)
) STRICT;
