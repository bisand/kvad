#!/usr/bin/env python3
"""Reference scores for kvad's diarization scorer, from pyannote.metrics.

`crates/llm/src/speech/score.rs` writes out the diarization error rate, the
Jaccard error rate, the word diarization error and cpWER by hand. This makes
pairs of reference and hypothesis RTTM, and scores each pair with the tools
the field reports with, at the three protocols of the VBx paper:

- DER and JER by `pyannote.metrics` (MIT), with each component of DER;
- cpWER by `meeteval` (MIT), the CHiME challenges' scorer;
- WDER, which no package scores: Shafey et al.'s (2019) definition, written
  here a second time, apart from the Rust, with scipy's assignment. Which
  words a transcript's words are paired with is the fewest edits, and where
  two pairings cost as much, WDER can differ by a word: both sides take a
  substitution before a deletion before an insertion, from the end back.
  (jiwer's alignment, rapidfuzz's, breaks ties in no one order of the
  three, and moved one meeting here by 0.2 points.)

The cases are made from a fixed seed, so the file is the same every run:
hand-made ones, the diarization probe's two-voice dialogue (the true turns
`scripts/dialogue.py` writes for its default script), random meetings with
overlap, and edges (an empty hypothesis, one speaker for everything, a speaker
for every turn, turns shorter than the collar). A hypothesis is a reference
with its boundaries moved, turns dropped, speakers merged or split, and false
alarms added, as a real diarization's mistakes look.

    /opt/homebrew/bin/python3.14 -m venv /tmp/der-venv
    /tmp/der-venv/bin/pip install pyannote.metrics meeteval scipy
    /tmp/der-venv/bin/python scripts/der-fixtures.py
    cargo test --release -p kvad speech::score

It writes `crates/llm/src/speech/der-fixtures.json`, which is committed, so
the test needs none of this. `--rttm REF HYP` adds a pair of one's own (a
real recording's), and `--out` writes somewhere else.

pyannote's `collar` is the collar's whole width, half either side of a
boundary; the protocols here say seconds either side, as `md-eval` does, and
so pass twice that.
"""
import argparse
import json
import os
import random
import warnings
from importlib.metadata import version

import numpy as np
from meeteval.wer import cp_word_error_rate
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from scipy.optimize import linear_sum_assignment

# Seconds either side of a boundary, and whether overlap is scored.
PROTOCOLS = {"full": (0.0, False), "fair": (0.25, False), "forgiving": (0.25, True)}

# The diarization probe's dialogue: who spoke when, to the millisecond, and
# what they said, as `scripts/dialogue.py` writes it for its default script.
DIALOGUE = [
    ("A", 0.5, 5.469, "Hei, og velkommen til møtet. I dag skal vi snakke om budsjettet for neste år."),
    ("B", 5.719, 9.868, "Takk. Jeg har sett på tallene, og de ser bedre ut enn i fjor."),
    ("A", 10.168, 11.267, "Det er godt å høre."),
    ("B", 11.467, 14.384, "Men vi må fortsatt kutte litt i reisekostnadene."),
    ("A", 14.734, 15.088, "Ja."),
    ("B", 15.338, 17.64, "Kan vi ta det opp med styret på torsdag?"),
    ("A", 17.94, 24.408, "Det tror jeg er lurt. Jeg sender en innkalling i ettermiddag, så får alle lest saken før møtet."),
    ("B", 24.608, 26.615, "Flott, da gjør vi det slik."),
]

VOCABULARY = "hei takk ja nei vi må det er godt og så men kan ta opp med på i dag møtet tallene budsjettet styret saken".split()


def rttm(turns):
    return "".join(f"SPEAKER fx 1 {s:.3f} {e - s:.3f} <NA> <NA> {who} <NA> <NA>\n" for who, s, e in turns)


def read(text):
    """An RTTM as pyannote holds one: a track a line, so that two turns of
    one speaker that overlap stay two, as `load_rttm` keeps them."""
    a = Annotation(uri="fx")
    for i, line in enumerate(text.splitlines()):
        f = line.split()
        start, dur = float(f[3]), float(f[4])
        if dur > 0:
            a[Segment(start, start + dur), i] = f[7]
    return a


def scores(ref_text, hyp_text):
    ref, hyp = read(ref_text), read(hyp_text)
    out = {}
    for name, (collar, skip) in PROTOCOLS.items():
        der = DiarizationErrorRate(collar=2 * collar, skip_overlap=skip)(ref, hyp, detailed=True)
        jer = JaccardErrorRate(collar=2 * collar, skip_overlap=skip)(ref, hyp)
        out[name] = {
            "der": 100 * der["diarization error rate"],
            "jer": 100 * jer,
            "total": der["total"],
            "missed": der["missed detection"],
            "false_alarm": der["false alarm"],
            "confusion": der["confusion"],
        }
    return out


def words(text):
    """As `score::words`: lower case, punctuation stripped from each end."""
    out = []
    for w in text.split():
        i, j = 0, len(w)
        while i < j and not w[i].isalnum():
            i += 1
        while j > i and not w[j - 1].isalnum():
            j -= 1
        if j > i:
            out.append(w[i:j].lower())
    return out


def paired(r, h):
    """The words of `h` paired with words of `r` (the same or substituted),
    by the fewest edits, ties broken as the docstring says."""
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(len(r) + 1), np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j - 1] + (r[i - 1] != h[j - 1]), d[i - 1, j] + 1, d[i, j - 1] + 1)
    i, j, pairs = len(r), len(h), []
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            i, j = i - 1, j - 1
            pairs.append((i, j))
        elif i and d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def word_scores(reference, hypothesis):
    """cpWER by meeteval, and WDER over the pairing above."""
    join = lambda ws: {s: " ".join(w for t, w in ws if t == s) for s in dict.fromkeys(t for t, _ in ws)}
    cp = cp_word_error_rate(reference=join(reference), hypothesis=join(hypothesis))
    pairs = paired([w for _, w in reference], [w for _, w in hypothesis])
    refs = list(dict.fromkeys(s for s, _ in reference))
    hyps = list(dict.fromkeys(s for s, _ in hypothesis))
    agree = np.zeros((len(hyps), len(refs)))
    for i, j in pairs:
        agree[hyps.index(hypothesis[j][0]), refs.index(reference[i][0])] += 1
    mapped = {hyps[h]: refs[r] for h, r in zip(*linear_sum_assignment(-agree))}
    wrong = sum(mapped.get(hypothesis[j][0]) != reference[i][0] for i, j in pairs)
    return {
        # `speaker:word`, a word at a time.
        "reference": " ".join(f"{s}:{w}" for s, w in reference),
        "hypothesis": " ".join(f"{s}:{w}" for s, w in hypothesis),
        "cpwer_errors": int(cp.errors),
        "cpwer_length": int(cp.length),
        "wder": 100 * wrong / len(pairs),
    }


def meeting(rng, speakers, seconds, overlap):
    """Turns of exponential length, a speaker after another, the next one
    sometimes starting before the last has stopped."""
    turns, t, last = [], rng.uniform(0, 2), None
    while t < seconds:
        who = rng.choice([s for s in speakers if s != last])
        length = min(rng.expovariate(1 / 3.5) + 0.3, 20)
        turns.append((who, t, t + length))
        last = who
        t += length + (-rng.uniform(0.2, 1.5) if rng.random() < overlap else rng.uniform(0.05, 1.2))
    return turns


def mistake(rng, turns, jitter=0.2, drop=0.05, merge=False, split=False, alarms=3, extra=False):
    """A diarization's mistakes, on a reference: what a hypothesis looks like."""
    names = sorted({w for w, _, _ in turns})
    label = {n: f"spk{i}" for i, n in enumerate(rng.sample(names, len(names)))}
    if merge and len(names) > 2:
        label[names[1]] = label[names[0]]
    out = []
    for who, s, e in turns:
        if rng.random() < drop:
            continue
        s, e = s + rng.gauss(0, jitter), e + rng.gauss(0, jitter)
        if e - s < 0.05:
            continue
        name = label[who]
        if split and who == names[-1] and rng.random() < 0.4:
            name = "spk_split"
        out.append((name, max(0.0, s), e))
    end = max(e for _, _, e in turns)
    for _ in range(alarms):
        s = rng.uniform(0, end)
        out.append((rng.choice(list(label.values())), s, s + rng.uniform(0.2, 2)))
    if extra:
        s = rng.uniform(0, end)
        out.append(("spk_extra", s, s + rng.uniform(1, 4)))
    return out


def worded(rng, turns):
    """Each turn given a few words from a small vocabulary."""
    return [(who, rng.choice(VOCABULARY)) for who, s, e in turns for _ in range(max(1, int((e - s) * 2)))]


def misheard(rng, reference, sub=0.08, dele=0.05, ins=0.04, flip=0.06):
    """A transcript's mistakes on words and speakers, with fresh labels."""
    names = list(dict.fromkeys(s for s, _ in reference))
    label = {n: f"h{i}" for i, n in enumerate(rng.sample(names, len(names)))}
    out = []
    for i, (who, w) in enumerate(reference):
        if rng.random() < dele:
            continue
        if rng.random() < sub:
            w = rng.choice(VOCABULARY) + "x"
        speaker = label[who]
        # A speaker error is likeliest at a turn's edge.
        edge = (i > 0 and reference[i - 1][0] != who) or (i + 1 < len(reference) and reference[i + 1][0] != who)
        if rng.random() < (5 * flip if edge else flip / 3):
            speaker = rng.choice(list(label.values()))
        out.append((speaker, w))
        if rng.random() < ins:
            out.append((speaker, rng.choice(VOCABULARY)))
    return out


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "crates", "llm", "src", "speech", "der-fixtures.json"))
    ap.add_argument("--rttm", nargs=2, action="append", default=[], metavar=("REF", "HYP"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*'uem' was approximated.*")
    rng = random.Random(20261018)
    cases = []

    def add(name, ref, hyp, words=None):
        ref_text, hyp_text = (ref if isinstance(ref, str) else rttm(ref)), (hyp if isinstance(hyp, str) else rttm(hyp))
        case = {"name": name, "reference": ref_text, "hypothesis": hyp_text, "scores": scores(ref_text, hyp_text)}
        if words:
            case["words"] = word_scores(*words)
        cases.append(case)
        print(f"{name:24} " + "  ".join(f"{k} DER {v['der']:6.2f} JER {v['jer']:6.2f}" for k, v in case["scores"].items()))

    add("by-hand", [("A", 0, 10), ("B", 8, 20)], [("x", 0, 9), ("y", 9, 20), ("z", 20, 22)])

    dialogue = [(w, s, e) for w, s, e, _ in DIALOGUE]
    dialogue_words = [(w, x) for w, _, _, text in DIALOGUE for x in words(text)]
    add("dialogue-perfect", dialogue, [("s1" if w == "A" else "s0", s, e) for w, s, e in dialogue],
        (dialogue_words, [("s1" if w == "A" else "s0", x) for w, x in dialogue_words]))
    # The probe's kind of mistake: the one-word turn swallowed by its
    # neighbours, and boundaries a little out.
    hyp = [("s1" if w == "A" else "s0", s + rng.gauss(0, 0.08), e + rng.gauss(0, 0.08)) for w, s, e in dialogue if e - s > 0.5]
    add("dialogue-mistaken", dialogue, hyp, (dialogue_words, misheard(rng, dialogue_words)))

    for k, (n, seconds, overlap, kw) in enumerate([
        (2, 60, 0.05, {}),
        (3, 120, 0.15, {"merge": True}),
        (4, 180, 0.2, {"split": True}),
        (5, 240, 0.25, {"extra": True, "jitter": 0.4}),
        (3, 90, 0.4, {"drop": 0.2, "alarms": 10}),
        (6, 200, 0.1, {"merge": True, "split": True, "extra": True}),
    ]):
        names = [f"S{i}" for i in range(n)]
        ref = meeting(rng, names, seconds, overlap)
        said = worded(rng, ref)
        add(f"meeting-{k}", ref, mistake(rng, ref, **kw), (said, misheard(rng, said)))

    ref = meeting(rng, ["S0", "S1", "S2"], 60, 0.2)
    add("empty-hypothesis", ref, [])
    add("one-speaker-for-all", ref, [("x", 0.0, 61.0)])
    add("a-speaker-a-turn", ref, [(f"t{i}", s + 0.05, e - 0.05) for i, (_, s, e) in enumerate(ref)])
    add("shorter-than-collar", [("A", 1.0, 1.3), ("B", 1.4, 1.6), ("A", 2.0, 6.0)], [("x", 1.1, 1.5), ("y", 1.5, 2.2), ("x", 2.2, 6.4)])
    add("hypothesis-past-reference", [("A", 5.0, 9.0), ("B", 9.5, 12.0)], [("x", 0.0, 9.2), ("y", 9.2, 15.0)])
    add("three-at-once", [("A", 0, 6), ("B", 2, 8), ("C", 3, 5), ("A", 9, 12)], [("x", 0, 7), ("y", 2.5, 8.5), ("x", 9.2, 11.8)])
    for i, (r, h) in enumerate(args.rttm):
        add(f"given-{i}-{os.path.basename(r)}", open(r).read(), open(h).read())

    with open(args.out, "w") as f:
        json.dump(
            {"made_by": "scripts/der-fixtures.py", "pyannote.metrics": version("pyannote.metrics"), "meeteval": version("meeteval"), "cases": cases},
            f,
            ensure_ascii=False,
            indent=1,
        )
        f.write("\n")
    print(f"{len(cases)} cases to {os.path.relpath(args.out)}")


if __name__ == "__main__":
    main()
