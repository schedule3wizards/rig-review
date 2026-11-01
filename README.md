# KayKit Knight rig review

A rig review script and its rule schema, with the stock KayKit Knight as the worked example.

## Contents

| Path | What it is |
| --- | --- |
| `knight/Knight.glb` | KayKit Adventurers Character Pack 1.0, Knight: skinned, 76 animation clips, texture embedded |
| `knight/knight_texture.png` | the Knight's texture as a separate file |
| `knight/LICENSE-KayKit.txt` | the pack's licence: CC0, by Kay Lousberg (www.kaylousberg.com) |
| `rig_review/rig_review.py` | the review script |
| `rig_review/rig_schema.json` | 35 rules, profiles `humanoid` and `ogre` |
| `rig_review/LICENSE` | MIT, for the script and the schema |
| `example_review/` | the script's output on the stock Knight: sheet (PNG), results (JSON), console log |

## Run it

Get it from github.com/schedule3wizards/rig-review (clone it, or Code > Download ZIP).

Python 3 with numpy and Pillow. scipy is optional; without it `motion.deltoid_volume` reports SKIP.

```
python rig_review/rig_review.py knight/Knight.glb --profile humanoid --out-dir out --name kaykit_knight
```

Exit code 0 means every rule passed, 1 means at least one FAIL, 2 means the file could not be read, 3 means no rule
failed but at least one could not be measured (SKIP; the verdict reads INCOMPLETE), and 4 means the review raised on a
file that loaded, which is a defect in the script rather than in the input. It writes
`<name>_rig_review.json` (every rule with its value, threshold and status) and `<name>_rig_review.png` (front and side
bind-pose overlays with neck-base, shoulder and hip guide lines, the rule table, and a six-frame pose strip skinned by
linear blend from the glb's own animations).

## What it expects

- A skinned glb with a T-pose bind (inverse bind matrices in a T-pose).
- KayKit bone names: `hips`, `spine`, `chest`, `head`, `upperarm.l/.r`, `lowerarm`, `wrist`, `hand`, `upperleg`,
  `lowerleg`, `foot`, `toes`. A clavicle is any joint named `clav*`, `collar*` or `shoulder*` that parents an upper arm;
  a neck is any joint named `neck*` between the top spine joint and the head. Other naming conventions need edits.
- Y up, +X is the figure's left (`.l`), +Z forward, feet at y = 0.

## The example result

The stock Knight fails 5 of 35 rules, all expected:

- `topology.clavicles`, `topology.neck_bone`, `topology.head_span`: the KayKit armature has no clavicles, no neck bone
  and no head end joint.
- `placement.chest_rib_base`: no spine joint sits at the base of the ribcage. Measured from the hips to the shoulder
  line, the spine joint is at 0.27 and the chest joint at 0.81; the rule wants one between 0.40 and 0.68 and reports the
  nearer one, the spine joint.
- `arm.upper_fore_ratio`: upper arm 0.93 times the forearm; anthropometry puts it near 1.27 (Drillis and Contini via
  Winter), so the band is 1.05 to 1.40.

## Limits

- A pass means no measured rule fired, not that the rig looks right. Deformation quality is measured only as a convex-hull
  volume ratio around the deltoid.
- Mesh landmarks (arm tube, torso edge, neck level, crotch) assume horizontal arms in the bind pose and are corrupted by
  cloaks, robes, wings and tails.
- Thresholds are anchored to the stock Knight, anthropometry and two standard humanoid rig references; no published
  joint-placement numbers exist for stylized bodies.
- The self-test that plants defects is not included: it loads rigs that cannot be shared.

## Licences

- KayKit Knight: CC0 1.0 (see `knight/LICENSE-KayKit.txt`).
- `rig_review.py`, `rig_schema.json`: MIT, copyright (c) 2026 Mattlock of Schedule 3 Wizards (see `rig_review/LICENSE`).
