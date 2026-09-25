# The Kvad icon

An open book seen from its end: the covers make a V, and the pages fan up out
of the spine inside them — the text a model is trained on, and the *v* in
*kvad*.

| File | For |
|---|---|
| `kvad.svg` | The icon, from 48px up. `kvad-1024.png` is it rendered for places that want a bitmap. |
| `kvad-small.svg` | 32px and below: one page instead of two, and heavier strokes, so the V survives at 16px. It is also `web/public/favicon.svg`. |
| `kvad-mark.svg` | One colour, `currentColor`, no tile — beside the name, on either theme. The web UI's sidebar carries it inline. |

`web/public/apple-touch-icon.png` is `kvad.svg` at 180px with square corners,
because iOS rounds them itself.

| | |
|---|---|
| Ink (tile) | `#1b1f2e` |
| Gold (covers) | `#f0b35a` |
| Parchment (pages) | `#efe3c8`, and `#b9ad93` for the page further back |

The PNGs come from Inkscape:

```bash
inkscape kvad.svg -w 1024 -h 1024 -o kvad-1024.png
```
