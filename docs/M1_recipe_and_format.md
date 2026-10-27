# Milestone 1 — recipe summary and MOD format findings

Sources: MessyKitchens repo `465fe2211a35cb54df224a4e208aa0971fc2c458` (2026-09-29), paper
arXiv 2603.16868v2 §3.1, §3.2, §4.2, §5.2 and Appendix C, GAPS `mshalign.cpp` @ `f78eb23e`.

## Recipe (paper)

1. Assets: 42 GSO kitchenware objects. Remesh to uniform triangles; fix centre of mass where needed (C.1).
2. Physics: Blender with concave mesh collision. Margins: 0.01 mm object-object, 1.0 mm object-plane. Restitution 0. Velocity deactivation off (C.2).
3. Plane: an *active* rigid body with translation and rotation fully constrained. The authors report this penetrates less than a passive collider (C.2).
4. Easy: 4 objects placed or dropped on a plane, ~50% upright, the rest at arbitrary orientations (C.3).
5. Medium: 4 base objects with jittered (x, y) around the centre (jitter controls tightness), then 2 objects placed above supports and dropped. Kept only if stable (§3.2, C.3).
6. Hard: medium plus several stacks and nests. Feasibility comes from a precomputed table of upright top-face bbox area and volume, plus a never-on-top list (pitchers, kettles) (C.3).
7. Render: Cycles with the original GSO textures, 10 views, azimuth in [0, 2π], elevation in [π/4, π/2]. Instance maps are exported with the RGB (§3.2).
8. Training uses 600 scenes per level × 6 images = 10,800 images (§5.1).
9. MOD targets: Sim(3) ICP from the SAM3D scene to the GT scene, then per-object Sim(3) ICP, decomposed into q, t, σ (§4.2). In the repo this is `prepare.sh --generate-training-gt` (GAPS mshalign).
10. Contacts: contact area = surface within 2.5 mm of another object; penetration area = intersecting surface found by voxelisation. The paper reports the ratio **penetration / contact** (real data: 0.14 overall; easy 0.072, medium 0.153, hard 0.135) (§5.2, Table 1).

## Not specified by the paper (my defaults, all in config)

| Item | Default chosen |
|---|---|
| Which 42 GSO objects | candidate list from Fuel (all 1,033 GSO models are CC BY 4.0) |
| Remesh target edge length / method; COM correction rule | voxel/isotropic remesh to ~2-3 mm edges; COM = solid-volume centroid of the watertight hull |
| Object count in synthetic hard scenes | 8 (matches real hard) |
| Drop heights, jitter radius, settle time, "still moving" threshold, "fell off" rule | explicit config values + tests |
| Nesting compatibility thresholds (volume ratio, footprint ratio, inner-cavity fit) | config |
| Lighting / HDRI source, table materials, camera distance, FOV, Cycles samples/denoiser | Poly Haven CC0 HDRIs + CC0 textures; FOV/distance chosen so the scene fits; 128 samples, adaptive threshold 0.01, OptiX denoiser on NVIDIA (OIDN elsewhere), GPU-only devices |
| Which 6 of the 10 views are kept | the 6 with best per-object visibility (deterministic) |
| Minimum visible pixels for an object to be given to SAM3D | config (`min_visible_px`) |
| Contact/penetration voxel size, surface sampling density | 1 mm voxels, area-weighted surface samples |
| Train/val split | none in the paper (MOD's own val used 10 external records) |
| Collision shape fallback when MESH glitches | CoACD (per the brief). As built: falling objects are CoACD compounds whose parts are shrunk 0.4 mm and get it back as Bullet margin (shape = true surface); once a stage settles, its objects become static colliders of their scans (decimated to ≤ 8k triangles, p99 deviation ≤ 0.4 mm) with the paper's 0.01 mm margin, and the next stage is dropped onto them. Convex parts bridge cavities and plate wells, so dropping onto them would hold objects above the real surface. |

Inconsistencies: Appendix C.3 says "For hard scenes, we then select two additional objects…" but describes
medium (§3.2 is unambiguous). The internal training manifest the repo mentions has **9,090** records
(= 1,515 scenes × 6), not 10,800, and 2,882 GSO-hard raw records are lost, so the paper's actual
training set was probably smaller than 1,800 scenes.

## MOD input format (what `prepare.sh --generate-training-gt` needs)

Raw manifest = JSON **list, one record per view**, paths relative to the manifest's directory:

```json
{
  "id": "easy_000123__v03",                 // unique per VIEW (becomes the output dir name)
  "scene_id": "easy_000123",                // physical scene
  "image_path": "scenes/easy/000123/rgb/v03.png",
  "instance_mask_path": "scenes/easy/000123/mod_mask/v03.png",  // uint8/uint16 integer labels
  "num_parts": 4,                           // MUST equal number of labels present (checked)
  "mesh_path": "scenes/easy/000123/gt_scene.glb",
  "gt_object_map": {"0": "obj_000", "2": "obj_002", "3": "obj_003"}
}
```

- Mask: 0 = background. Label k means SAM3D object id k-1. Missing labels are **not** renumbered, so a
  label stays fixed per scene object across all views.
- `gt_object_map` must cover **exactly** the labels present in that view's mask and be one-to-one.
  Values are **scene-graph node names** in the GLB.
- GLB: one geometry-bearing node per object, posed. mshalign's default is PCA (all 8 axis flips) plus
  Sim(3) ICP, so the GT frame and units can be anything. World frame in metres is fine.
- Only mapped nodes enter registration, so fully occluded objects can stay in the GLB.
- The step writes `target_poses.json` itself. Object-keyed `translation [x,y,z]`, quaternion
  `[w,x,y,z]`, `scale [sx,sy,sz]` in the SAM3D prediction frame (row-vector convention). We do **not**
  write MOD targets. Our own world-frame poses are for validation and other consumers.
- A scene fails if any object's registration fails or doesn't converge. `--continue-on-error` collects
  failures in `prepare_manifest.json`.
- Requirements: Linux, NVIDIA GPU ≥32 GB (A100 40 GB reference), SAM3D @ `afdf6a31`, PyTorch 2.5.1/cu121,
  **gated** HF weights `facebook/sam-3d-objects` (SAM License), and GAPS `mshalign` built from source.
  Reference cost: 82 s for one 4-object view, including model load.
- The authors' naming: `train_0000_az_055_el_073_fr_0000`, dataset `GSO_easy`.

## Hold-out list

The MessyKitchens-Real snapshot `edc5482` has 134 object-model PLYs (`mk_objects_###.ply`; the dataset
card says 130). Names and SHA-256s are saved in `docs/messykitchens_benchmark_object_models_edc5482.json`.
The generator rejects any asset whose source is `messykitchens_real` or whose file hash matches.
That dataset is CC BY-NC 4.0, so it is excluded by the licence rule anyway.
