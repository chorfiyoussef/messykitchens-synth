# messykitchens-synth: MessyKitchens-Synthetic, recreated

A generator for synthetic cluttered-kitchen scenes following the MessyKitchens-Synthetic recipe
(arXiv 2603.16868, §3.2 and App. C). Its output plugs straight into MOD (Multi-Object Decoder,
github.com/MessyKitchens/MessyKitchens): every scene comes with a MOD raw manifest record per
kept view, integer instance masks and a posed GT scene (`gt_scene.glb`) for
`prepare.sh --generate-training-gt`.

An independent, unofficial recreation: not affiliated with or endorsed by the MessyKitchens
authors (Ansari et al., ECCV 2026).

The generated dataset (1,890 scenes, 10,800 MOD training records) is on Hugging Face:
[chorfiyoussef/messykitchens-synth](https://huggingface.co/datasets/chorfiyoussef/messykitchens-synth).
Recipe summary, MOD format findings and every choice the paper leaves open:
[docs/M1_recipe_and_format.md](docs/M1_recipe_and_format.md). Dataset statistics for the full run:
see [Results](#results).

## Quick start

```bash
git clone --recursive https://github.com/chorfiyoussef/messykitchens-synth.git   # MOD as a submodule
cd messykitchens-synth
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m mksynth.assets.fetch               # GSO assets (CC BY 4.0) -> data/assets/raw
.venv/bin/python -m mksynth.assets.preprocess --coacd  # proxies, COM, shape props, convex parts
.venv/bin/python -m mksynth.assets.fetch_env           # Poly Haven HDRIs + table textures (CC0)

.venv/bin/python generate.py --level easy --n_scenes 600 --seed 0 --out out/full/easy
.venv/bin/python validate.py --out out/full/easy
.venv/bin/python -m mksynth.viz out/full/easy/scenes/train_easy_00000   # overlay panels
bash scripts/run_full.sh                                # all three levels, validation, merged manifests
```

BlenderProc 2.8.0 downloads its pinned Blender (4.2.1) into `.blender/` on first use.
`generate.py` options: `--n_val` (validation scenes, default 5% of `--n_scenes`), `--gpus 0,1`
(one Blender worker per slot, `--workers_per_gpu` slots per GPU), `--device_type OPTIX|CUDA|METAL`
(GPU only; with OptiX the OptiX denoiser is used), `--chunk` (scenes per Blender process),
`--max_retries`, `--set key.path=value` (config overrides). Each worker gets an equal share of the
machine's CPU quota (cgroup-aware): physics runs on the CPU while other workers render.
Runs are resumable: finished scenes are skipped, failed ones are retried with fresh layouts and
then listed in `failures.jsonl`. `--redo SCENE_ID ...` regenerates finished scenes the same way
(fresh layouts), e.g. the ones `validate.py` rejects.

For a GPU machine (RunPod etc.), `scripts/pod/setup.sh generator gaps sam3d weights env_file`
installs the generator, GAPS `mshalign` and the pinned SAM3D environment used by MOD.

## How a scene is made

1. **Assets** (`configs/assets.yaml`). 42 training and 10 validation-only Google Scanned Objects
   (all CC BY 4.0; licence, URL and file hashes recorded per asset in `source.json`). Each scan
   gets a watertight proxy (SDF remesh), a solid-volume centre of mass, shape properties for
   stacking and nesting (top face, outer radius profile, cavity profile), and a convex
   decomposition (CoACD, parts shrunk back onto the scanned surface). The 134 MessyKitchens-Real
   benchmark objects are held out by source and file hash. Other sources (OmniObject3D,
   RoboCasa) plug in through `sources` + a fetcher.
2. **Plan** (`mksynth/layout.py`, `mksynth/compat.py`). Easy: 4 objects, about 50% upright, spread
   out. Medium: 4 jittered base objects + 2 stacked. Hard: 4 base + 2 nested + 2 stacked, with
   chains allowed. Stack/nest pairs come from a compatibility table (top-face area, volume,
   never-on-top list, and a cavity-profile fit for nesting).
3. **Physics** (`blender/scene_worker.py`). Bullet through Blender, margins 0.01 mm
   (objects) and 1 mm (table), restitution 0, no deactivation, and an active table held by a fixed
   constraint (paper App. C.2). Base objects are convex compounds dropped on the table (parts
   shrunk 0.4 mm, given back as margin). After each stage settles, its objects are frozen as
   static meshes of their scans (decimated to ≤ 8k triangles, p99 deviation ≤ 0.4 mm), and the
   next stage is dropped onto them. Convex parts bulge over cavities and would hold objects above,
   or across, the real surface. A scene is rejected if anything glitches (> 3 m/s), keeps moving,
   falls off, breaks its planned relation, moves during a 1 s hold, would tip a support
   (combined centre of mass outside its base), overlaps another object by > 2 mm, or hovers over
   its planned parent (gap > 0.5 mm between the watertight proxies, i.e. ~1.3 mm between the scans:
   the child's own convex parts can bridge a recess such as a foot ring). A rejected layout is
   replaced by a new one.
4. **Render.** Cycles, original GSO textures, 512², 10 views (azimuth 0–360°, elevation 45–90°),
   random Poly Haven HDRI, extra area light, table material, FOV 45–65°, camera distance fitted
   so all objects are in frame.
5. **Annotate** (`mksynth/annotate.py`). Amodal masks by raycasting each object alone, contact
   (≤ 2.5 mm) and penetration (1 mm voxels) labels on the faces of the original meshes, support
   graph, 6 kept views (most objects visible, then highest visible fraction), MOD masks and records.

## Output layout

```
out/full/
  mod_train.json, mod_val.json      MOD raw manifests, all levels (paths relative to out/full)
  stats.json, STATS.md              dataset statistics
  easy/ medium/ hard/
    config.yaml                     resolved config snapshot
    manifest.jsonl                  one line per scene (summary statistics)
    mod_train.json, mod_val.json    MOD raw manifests for this level
    failures.jsonl                  scenes that could not be generated, with reasons
    validation.json                 validate.py report
    contact_sheet.png
    scenes/<split>_<level>_<index>/
      rgb/vNN.png                   10 views, 512x512
      depth/vNN.png                 uint16 z-depth, 0.1 mm units (0 = invalid)
      mask_visible/vNN.png          uint8 instance labels: 0 background, k = object k-1
      masks_amodal.npz              amodal bool [view, object, H, W]
      mod_mask/vNN.png              MOD instance mask for the 6 kept views
      gt_scene.glb                  original meshes, world frame, nodes obj_000, obj_001, ...
      contacts.npz                  per object, per face of the original mesh: contact / penetration
                                    fraction, partner (index; n = table), samples
      scene.json                    objects (asset, licence, attribution, mesh path, T_world_mesh,
                                    scale), cameras (K, T_world_cam, OpenCV), per-view 6D poses
                                    T_cam_mesh + quaternion, visibility, support graph, contacts,
                                    physics log, render settings, self-checks
```

World frame: z up, metres, table top at z = 0. `T_a_b` maps points from frame b to frame a.
Object poses refer to the original GSO mesh frame (`mesh_path`), with uniform scale 1.

## Validation

`validate.py` re-derives what MOD will consume from the saved files alone. It checks:
- **Masks:** loads `gt_scene.glb` the way MOD's `training_gt.py` does, raycasts it with the
  saved cameras, and compares the result with the Blender masks and depth.
- **Physics:** no floating or sunken objects, and penetration below 2 mm.
- **Consistency:** poses are consistent, and planned stacks and nests show up in the
  contact-based support graph.
- **MOD rules:** the same rules `mod-prepare` enforces (labels match `gt_object_map`,
  `num_parts`, one-to-one GLB nodes).
- **Licences and split:** allowed licences only, and no object shared between train and val.
- **Renders:** no black RGB views (Cycles renders black, without an error, when it runs out of
  GPU memory).

With `--mod_prepare N` it also runs `prepare.sh --generate-training-gt` on N scenes (all kept
views, or the first `--mod_prepare_views K` of each). That needs SAM3D and GAPS on a Linux NVIDIA
GPU with ~25 GB free: SAM3D does not fit next to 8 Blender workers on one 48 GB L40S, so run it
after generation, not alongside.

Tests: `pytest tests` (compatibility table, planner, physics checks, annotation). Set
`MKSYNTH_BLENDER_TESTS=1` to also run the Blender physics integration tests.

## Results

### Full run (`out/full`, seed 0, October 2026)

1,890 scenes (600 train + 30 val per level), all 1,890 passing `validate.py`; 10,800 training and
540 validation MOD records (`out/full/mod_train.json`, `mod_val.json`); 9.0 GB.

| level / split | scenes | MOD records | objects / scene | assets | scenes with nest | with stack | pen / contact (ours) | pen / contact (paper, real) | max pen. p95 (mm) | layouts tried / scene |
|---|---|---|---|---|---|---|---|---|---|---|
| easy / train | 600 | 3600 | 4 | 42 | 1 | 54 | 0.000 | 0.072 | 0.17 | 1.14 |
| easy / val | 30 | 180 | 4 | 10 | 0 | 2 | 0.000 | 0.072 | 0.05 | 1.17 |
| medium / train | 600 | 3600 | 6 | 42 | 0 | 600 | 0.006 | 0.153 | 1.05 | 1.94 |
| medium / val | 30 | 180 | 6 | 10 | 0 | 30 | 0.003 | 0.153 | 0.04 | 1.20 |
| hard / train | 600 | 3600 | 8 | 42 | 600 | 600 | 0.005 | 0.135 | 1.14 | 2.09 |
| hard / val | 30 | 180 | 8 | 9 | 30 | 30 | 0.002 | 0.135 | 0.15 | 1.50 |

Validation, worst scene per level (thresholds in brackets):

| level | 1-px mask agreement (≥ 0.9) | mean mask IoU, median | depth error (≤ 2 mm) | deepest overlap (≤ 2 mm) |
|---|---|---|---|---|
| easy | 1.000 | 0.979 | 1.44 mm | 0.26 mm |
| medium | 0.946 | 0.965 | 1.43 mm | 1.71 mm |
| hard | 0.951 | 0.957 | 1.41 mm | 1.87 mm |

Penetration / contact is 25–70× below the paper's real-data ratio for medium and hard and
essentially zero for easy: simulated contacts are cleaner than the reconstructed real scenes.
"Easy with stack" counts easy scenes where a dropped object came to rest on another.
`gso_aroma_frother` (a milk frother, no stack / nest / support role) is never drawn in the 30
hard validation scenes, so hard/val uses 9 of the 10 validation objects.

Compute (full table with per-scene physics and render times: `STATS.md` in the dataset):

| level | machine | workers | wall time | per scene |
|---|---|---|---|---|
| easy | MacBook Pro M3 Pro, Metal | 3 | 2 h 27 min | 14 s |
| medium | MacBook Pro M3 Pro, Metal | 3 | ~3 h 18 min (two runs) | ~19 s |
| hard | 1× NVIDIA L40S (RunPod, $1.09/h), OptiX | 8 | ~2 h 36 min (paused for M4) | ~15 s |

RunPod total for M4 and the hard level: about $10 over two pods (the first was stopped when the
account balance ran out; outputs on `/workspace` survived).

Regenerated with `--redo` after scanning finished scenes (7 of 1,890): 4 scenes made before the
`max_parent_gap` check (an object hovering ~2.5 mm over its support: `train_hard_00009`, `00054`,
`00491`, `train_medium_00133`), 2 rendered black while SAM3D shared the GPU (`train_hard_00101`,
`00102`, before the black-frame check), and 1 whose support edge was reversed (`train_hard_00259`,
see limitations).

### M4: 20 scenes per level through MOD prepare (`m4_mod_prepare_check/` in the dataset)

22 scenes per level (20 train + 2 val) generated on one L40S with 8 workers, then
`validate.py --mod_prepare 20 --mod_prepare_views 1`: `prepare.sh --generate-training-gt` (SAM3D +
GAPS `mshalign`) on one kept view of each of the 20 training scenes.

| level | generation (22 scenes) | validated | MOD target poses | MOD prepare per record |
|---|---|---|---|---|
| easy | 211 s | 22/22 | 19/20 | 60 s |
| medium | 472 s | 22/22 | 19/20 | 86 s |
| hard | 647 s | 21/22 | 20/20 | 115 s |

The two missing targets are records where GAPS reported `Converged = 0` for one object's
SAM3D reconstruction; MOD drops such records (`--continue-on-error`), as its authors also lost
records this way. The hard validation failure was the hovering object that led to the
`max_parent_gap` check. Prepare time grows with the number of objects (SAM3D runs per object);
for all 10,800 training records it extrapolates to about 3,600 × (60 + 86 + 115) s ≈ 260
L40S-hours, run by whoever trains MOD.

### Known limitations

- **Support direction between thin objects.** A sample counts as support when it lies within
  2.5 mm of the partner, so for two flat objects in full contact (a saucer on a plate) the
  lower one's underside can count too and the edge comes out reversed. About 0.3% of the 3,820
  object-on-object edges have a parent sitting higher than its child (some of them legitimate:
  a tilted tray resting on a rim). Scenes where this hides a planned relation fail validation
  and were regenerated; other edges are left as annotated. The support graph is an auxiliary
  label: MOD does not read it.
- **Renderer differs by level.** Easy and medium were rendered on Apple Metal with the OIDN
  denoiser, hard on NVIDIA with OptiX; same Cycles settings (128 samples, threshold 0.01).
- **Blender 4.2.1 on macOS** occasionally aborts while compiling Metal kernels at start-up
  (heap corruption in `MetalKernelPipeline::compile`, 5 of ~250 worker launches). No scene is
  affected; the chunk is retried.
- **Cycles out of GPU memory** renders black frames without an error. The worker now rejects
  them (`render_failed`, retried) and `validate.py` checks for them.

## Licences

Assets: Google Scanned Objects, CC BY 4.0 (attribution: Google Research, Downs et al. 2022);
per-asset records in `data/assets/raw/<id>/source.json` and in every `scene.json`. HDRIs and
textures: Poly Haven, CC0. MOD/SAM3D are used as external tools under their own licences
(SAM License for the SAM3D weights).

## Citation

If you use this generator or dataset, cite the MessyKitchens paper whose recipe it follows, and
Google Scanned Objects for the assets:

```bibtex
@inproceedings{ansari2026messykitchens,
  title     = {MessyKitchens: Contact-rich Object-level 3D Scene Reconstruction},
  author    = {Ansari, Junaid Ahmed and Ding, Ran and Pizzati, Fabio and Laptev, Ivan},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2603.16868}
}
```

Google Scanned Objects: Downs et al., "Google Scanned Objects: A High-Quality Dataset of 3D
Scanned Household Items", ICRA 2022.

Code: MIT ([LICENSE](LICENSE)).
