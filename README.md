# photon-minibeam-tps

Native pyRadPlan 0.5.0 planning with a `TOPASPhoton` extension for OpenTOPAS 4.2.p3. Prepare TOPAS configuration files locally, execute them on your cluster, then collect the results locally as native `Dij` or weighted forward dose. The cluster needs TOPAS and its Geant4 environment; it does not need Python or pyRadPlan.

## Install and activate

Use Python 3.12–3.14 (validated with 3.13).

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

For later sessions, run `source .venv/bin/activate` from the repository root. All commands below start there unless a cluster directory is shown.

## Run patient_workflow.py

The script has four stages:

| Stage | What it does | Output |
|---|---|---|
| `inspect` | Native DICOM import; list original ROI names and numbers, and report omissions. | Printed CT/structure information. |
| `prepare` | Generate native photon steering and TOPAS inputs. No simulation runs. | A portable bundle and `derived/steering.mat`. |
| `collect` | Validate completed TOPAS CSV files and assemble a sparse influence matrix. | `derived/result.mat` with native planning objects and `dij`, plus variance data. |
| `forward` | Stream completed beamlet doses with source-dependent exposure weights, without building a matrix. | `derived/dose.mha` and weighting metadata. |

The sequence is **inspect → edit settings → prepare → run TOPAS on the cluster → collect and/or forward**. `collect` and `forward` are independent; neither launches TOPAS.

After `forward`, each project/setup receives one `derived/dose_beam_001.png`,
`dose_beam_002.png`, etc., in saved beam order. Each figure shows **that beam's
weighted dose only**, combining its selected beamlet jobs. CT, dose and target
masks are resampled into the saved gantry/couch frame. The dose maximum is taken
only through the target's first-to-last occupied depth slices, including dose
outside the ROI shape inside this slab. Multiple targets use their combined
depth extent; absent targets fall back to full depth with a notice.

The grayscale CT is the full perpendicular cross-section nearest the beam's
isocenter, with projected target outlines in green. Small cyan dots show selected
bixel centers at the isocenter plane from saved steering, including zero-exposure
bixels in separate and combined modes; repeated ray associations share one dot.
The larger cyan `+` marks isocenter. Forward plot metadata records each dot’s
U/V and LPS coordinates and beam/ray/bixel associations. Axes U/V are transverse
millimeter coordinates relative to isocenter (U=X, V=Z at zero gantry/couch);
depth is positive downstream. All beam figures share linear inferno colors and
90% opacity with no positive-dose cutoff. Warnings and exposure weights are
annotated. Display interpolation does not change saved dose.

`dose.mha` remains the summed dose from all beams. CSV files are read once;
only the current beam volume and smaller 2D display arrays are retained for
plotting. Plot failures preserve dose outputs and are recorded per beam in
forward metadata and `indexing.md`. The previous `dose_projections.png` is no
longer generated; existing copies are left untouched and may be older outputs.

Collection accepts TOPAS's specific invalid-voxel-index warning when its complete
unscored-step count and unscored-energy block is valid. Both stages print the job,
CSV path, warning, unscored steps, energy in MeV, history count, and normalization.
They finish with **“completed with scorer warnings; validity requires user review.”**
The details are saved in `derived/collect_metadata.json` or
`derived/forward_metadata.json` (`scorer_warnings`, `review_required`) and
`derived/indexing.md`. Dose and normalization are unchanged; statistical uncertainty
does not account for missing energy, and no lost-dose percentage is inferred.
Unknown warnings, unexpected filtering, malformed warning blocks, and all other
result-integrity failures still stop collection. No setting or fresh simulation
is required to collect an existing result with this recognized warning.


### 1. Inspect the patient

```sh
python patient_workflow.py inspect
```

`inspect` always lists imported ROIs. With `TARGET` configured, it also prints steering and setup estimates without writing run files; set `TARGET = None` to list ROIs only.

The supplied patient imports as 341 × 341 × 160 CT voxels and all 51 structures. Native pyRadPlan owns DICOM import and rasterization. Omitted ROIs are reported; original ROI numbers are retained in metadata.

### 2. Edit the settings at the top of the script

Settings are editable Python variables; for example:

```python
DICOM_DIR = "dicom_9306087_fine"
HISTORIES_PER_JOB = 10
ENABLE_OPENGL = True
SOURCE_TYPE = "point"  # or "phase_space"
TARGET = "PTV2017fw"
GANTRY_ANGLES = [45.0, 135.0, 225.0, 315.0]
COUCH_ANGLES = None
DOSE_SPACING_MM = None
ONLY_CENTRAL_BEAMLET = True  # False: all target-selected beamlets
ENABLE_COLLIMATOR = True  # False omits the entire slit collimator
COLLIMATOR_SHIFT_FRACTIONS = []  # single-run walkthrough below; no slit overrides
```

In the inspect listing `38  PTV2017fw`, **38 is the DICOM ROI number**. Select the exact name `PTV2017fw`, without the number.

With `SOURCE_TYPE = "point"`, photon energies are sampled discretely from the 28 user-supplied energy points and relative weights in `minibeam/sources/empirical.py` (0.13–6.88 MeV). Weights are normalized when TOPAS inputs are generated. There is no `ENERGY_MEV` workflow setting. The internal machine energy label is 6.0 for native pyRadPlan steering compatibility; it does not set or rescale the spectrum. Couch angles default to zero. SAD and bixel width are in mm. **LPS** means Left–Posterior–Superior: positive x points toward the patient’s left, positive y toward their back, and positive z toward their head. `ISO_CENTER_LPS_MM` is a physical point `[x, y, z]` in the DICOM coordinate system, not voxel indices or coordinates relative to the target. With `ISO_CENTER_LPS_MM = None`, native steering places the isocenter at the selected target's mask centroid. Only the selected structure remains TARGET for steering; other targets become OAR.

`DOSE_SPACING_MM = None` scores on the native CT grid. The earlier target-covering preparation (`ONLY_CENTRAL_BEAMLET = False`) estimated 278 jobs per setup, about **482 GiB of CSV output** and **38.5 GiB for a dense dose matrix**. The sparse matrix may be smaller, but the CSV files still include every voxel.

For a smaller initial test you may explicitly change:

```python
DOSE_SPACING_MM = (6.0, 6.0, 6.0)
```

This changes **scoring resolution only**; CT transport retains its original voxel geometry. Actual scoring spacing is adjusted slightly to cover the same outer bounds. Both transverse grid dimensions must match for the pinned native matRad export.

With `ONLY_CENTRAL_BEAMLET = False`, native pyRadPlan projects target voxel centers for each beam angle from the source onto the isocenter plane, rounds their positions to the `BIXEL_WIDTH_MM` grid, and keeps unique positions. The current script disables additional target margin. `BEAMLET_EXECUTION = "separate"` gives one TOPAS job per selected bixel; `"combined"` gives one job per beam containing exactly the selected bixels. No optimization filters them. Point photons originate at the beam's common focal point and are sampled uniformly over disjoint square bixels on its isocenter plane. The former circular-cone approximation is replaced in newly prepared runs.

For a small test, set `ONLY_CENTRAL_BEAMLET = True` and optionally `HISTORIES_PER_JOB = 3000`. Collection uses the values saved in that run.

### Choose the Elekta phase-space source

Edit these settings in `patient_workflow.py` for a **new** run:

```python
SOURCE_TYPE = "phase_space"
PHASE_SPACE_FILE_BASE = "~/Local/MCGPU/TOPAS/Elekta_Precise_6MV/ELEKTA_PRECISE_6mv_part1"
HISTORIES_PER_JOB = 1_000_000
ENABLE_OPENGL = False  # cluster batch execution
ONLY_CENTRAL_BEAMLET = True
```

Supply the common basename without `.header` or `.phsp`. The existing default remains `SOURCE_TYPE = "point"` and your history count is not automatically changed. Ten point-source histories are usually too few to select particles from a narrow phase-space bixel; start with one million **original accelerator histories**, then assess statistics. Empty selections stop preparation with an error. Collection and forward dose read the saved source settings.

This reader explicitly supports the supplied little-endian IAEA 33-byte layout with constant Z, signed particle/energy fields, history increments and LATCH. It validates the entire file in chunks, even when replaying a small prefix, and records original header/data hashes. Unsupported layouts, inconsistent counts, malformed records and requests exceeding the original-history count are rejected. The originals are never changed.

Recorded photons, electrons and positrons retain their positions, directions, energies and statistical weights (approximately 0.05 in this file). The plane is **272.1 mm downstream of the nominal target**, placed using each beam’s fixed native gantry/couch frame. It is not aimed separately at each bixel, and neither the empirical spectrum nor the native nominal energy overrides recorded energies. In patient PDFs, the purple plane is the replay origin; stars mark the separate nominal focal points.

Each downstream trajectory is projected to the beam’s isocenter plane and selected inside its native **square** bixel: lower edges are inclusive and upper edges exclusive. Non-overlapping squares partition trajectories deterministically. Multiple beamlets per ray and overlapping squares are rejected. Bixels outside recorded coverage fail when empty. Selection occurs before transport through downstream hardware; it is not a guarantee of transmitted field coverage.

`HISTORIES_PER_JOB` selects a prefix of complete original histories, including gaps and trailing empty histories. Selected particles from one original history remain grouped. TOPAS binary headers record the requested original-history total and the number of nonempty selected histories; `PhaseSpaceIncludeEmptyHistories` restores the difference as empty events at the end of replay. `PhaseSpaceMultipleUse = 1` forbids recycling. This preserves the dose/statistical denominator, while not preserving the original ordering of empty events. LATCH is audited as part of the original file but is not a transported particle property.

Bundles contain only selected `.header`/`.phsp` pairs under `inputs/`; identical selection regions across rotated beams share a pair. Input hashes, selection bounds, particle/species counts, exclusions, empty-history counts and normalization appear in every job’s `source` record. The cluster still needs only TOPAS. Transfer the complete bundle and execute the existing per-job parameter files normally.

Dose is divided by **original accelerator histories**, never by selected particles or the sum of particle weights. Forward coefficients describe original-history exposure per saved job (one bixel or one combined beam). Columns share upstream histories, so `physical_dose_std_error` is omitted for phase-space forward dose; sparse per-column variance remains available. There is no MU calibration or clinical validation. See [OpenTOPAS phase-space source documentation](https://opentopas.readthedocs.io/en/main/parameters/source/phasespace.html) for replay controls.

### 3. Prepare a small test

```sh
python patient_workflow.py prepare --project projects/patient-cluster-smoke
```

- `ONLY_CENTRAL_BEAMLET = True` uses native `photonSingleBixel` steering: one central beamlet per configured beam, or four jobs for your four angles. `False` uses native `photonIMRT` steering for full target coverage. `BEAMLET_EXECUTION` independently chooses separate or combined jobs. Only Boolean values are accepted; the full-coverage count depends on target shape, angles and bixel width.
- `--project` specifies the project directory path. The folder name does not change the beamlet count. For example, naming it “smoke” does not change `ONLY_CENTRAL_BEAMLET`.
- `HISTORIES_PER_JOB` at the top of the script sets total primary photons per TOPAS job in point mode, or original accelerator histories per TOPAS job in phase-space mode. Preparation prints this meaning.
- `DICOM_DIR` at the top of the script selects the input folder. These are Python settings, not command-line flags or shell environment variables.

The script prints estimated workload sizes before writing files. For the full target-covering run, set `ONLY_CENTRAL_BEAMLET = False` and choose another directory:

```sh
python patient_workflow.py prepare --project projects/patient-cluster-full
```

**Collection uses saved settings, independently of current editable planning settings.** Prepare a new bundle when changing geometry, source, materials, settings, or engine implementation. The old patient-script flags `--smoke`, `--bundle`, `--dicom`, and `--histories` are removed. Only the stage and `--project` remain. The default project directory is `projects/patient` when the collimator is disabled or the shift array is empty; otherwise it is `projects/patient-slit-study`. Choose a fresh directory when settings or implementation change.

### Optional OpenGL visualization

Set `ENABLE_OPENGL = True` in `patient_workflow.py` before preparing a **new run directory**. Generated `inputs/common.txt` enables an OpenGL viewer, the Qt interface, geometry and particle trajectories, and pauses before exit. XYZ axes of length 100 mm are displayed at the patient center (the TOPAS origin), aligned with world/LPS directions: +X Left, +Y Posterior, +Z Superior. For DICOM patients, each job displays the native CT axial (Z) slice nearest its own isocenter, using TOPAS `ShowSpecificSlicesZ`. The selected 1-based slice number and its physical Z coordinate are recorded in the job’s `visualization_slice` metadata. Halfway ties select the higher slice; isocenters outside the CT use the nearest boundary slice. This changes only voxel visibility, not CT transport or dose scoring. TOPAS renders the slice with its material colors, not a diagnostic grayscale CT image. Water phantoms retain their existing box display. The window opens when you execute a job with TOPAS—not during Python preparation.

New DICOM bundles prepared with `ENABLE_OPENGL = True` also display a **cyan wireframe CT box**, alongside the selected slice. It follows the full voxel boundaries and is centered on the CT volume, not the target/isocenter. The unsegmented `CTOutline` box lives in a separate, material-free `CTOutlineWorld` parallel world; it adds no physical material or scorer. It is omitted when OpenGL is disabled and for water phantoms. It does not bypass collision checks or resolve the current slit/CT overlap. Existing bundles are not updated automatically.

Run one job interactively with a TOPAS build supporting OpenGL/Qt and an available graphical display. For example, prepare with `--project projects/patient-view`, then run `topas jobs/bixel_000001.txt` from that directory. Use a small history count for viewing large CT geometries. Histories and dose scoring are otherwise unchanged.

Keep `ENABLE_OPENGL = False` for cluster batch runs: graphics, Qt and the exit pause are disabled. This setting is recorded in the manifest and fingerprint; collection reads its saved value. Interactive window rendering depends on your TOPAS installation and display and is not verified by the automated tests.

### 4. Transfer the bundle and run TOPAS

A new bundle contains:

```text
projects/patient-cluster-smoke/
├── manifest.json
├── inputs/      # CT, shared parameters, material table, provenance, device assets
├── jobs/        # One parameter file per beamlet
├── results/     # TOPAS writes CSV files here
└── derived/     # Local MATLAB exports and reconstructed dose
```

Patient bundles now contain **unchanged original CT slices in `inputs/dicom/`**. TOPAS uses `TsDicomPatient` to read voxel counts, spacing, slice order and pixel rescaling directly from DICOM. No `ct_hu.raw` is generated. The local preparer checks that the original CT series matches pyRadPlan's geometry and HU values, then copies the original files byte for byte. The cluster TOPAS installation must include DICOM support (GDCM).

The script passes `DICOM_DIR` as the engine's `dicom_dir` setting. This directory must contain exactly one classic, single-frame, 16-bit CT series; RTSTRUCT and non-CT files are not copied into the transport directory. pyRadPlan continues to use RTSTRUCT locally for steering. Mixed CT series, duplicate slices, nonuniform spacing, unsupported orientation or a modified planning CT are rejected instead of silently changing the transport anatomy.

The patient component remains centered at TOPAS's origin. TOPAS reads the internal voxel geometry from DICOM, while our beam coordinates retain the corresponding LPS-to-centered-TOPAS translation. DICOM input does not automatically place the beam or isocenter. The synthetic water example uses `TsBox` and needs no DICOM files. See [TOPAS patient components](https://opentopas.readthedocs.io/en/latest/parameters/geometry/patient.html) for the native geometry conventions.

Transfer the entire bundle, for example:

```sh
scp -r projects/patient-cluster-smoke user@cluster:/work/
```

On the cluster, load your OpenTOPAS **4.2.p3** environment, including matching Geant4 libraries and data. Execute each job with the **bundle root as the working directory**:

```sh
cd /work/patient-cluster-smoke
topas jobs/bixel_000001.txt
```

Submit one such command per file in `jobs/` using your scheduler. The scheduler controls concurrency, logs and retries; there is no supplied submission script. TOPAS uses one thread per job by default. To change that, set `num_threads` in `plan.prop_dose_calc` before preparation and request matching CPUs from your scheduler. All paths in the generated configuration are relative to the bundle root.

Each job is independent. Do not run the same job twice concurrently. TOPAS refuses to overwrite an existing output. To retry a failed job, first confirm it is no longer running and move its partial CSV outside `results/`, then resubmit that job only. Leave completed outputs in place. Wait for all expected jobs to finish before collecting.

### 5. Transfer results back and collect

```sh
rsync -av user@cluster:/work/patient-cluster-smoke/results/ projects/patient-cluster-smoke/results/
python patient_workflow.py collect --project projects/patient-cluster-smoke
```

Collection validates input hashes and every expected CSV's configuration-specific scorer name, units, grid, history count, statistics, completeness and duplicate bins. No completion receipt files are needed. A missing or malformed result causes an error, never a zero-dose substitution. Leave the manifest and input files unchanged.

For weighted forward dose, set `FORWARD_WEIGHT_PER_BIXEL` (a scalar or a vector in saved job order) and run:

```sh
python patient_workflow.py forward --project projects/patient-cluster-smoke
```

Weights are **primary photon counts** in point mode and **original accelerator-history exposure per bixel** in phase-space mode; neither is MU. The default vector assigns one unit to each beamlet. `collect` is useful for matRad and future optimization; `forward` avoids retaining the whole influence matrix.

## Per-job parameters in manifest.json

Each entry in `manifest.json["jobs"]` automatically records the parameters of the native steering beamlet used to generate that TOPAS job. No separate metadata settings are needed.

| Fields | Meaning |
|---|---|
| `job_id`, `parameter_file`, `output` | Job identity and relative TOPAS input/CSV paths. |
| `bixel_index` | Global 1-based beamlet index: MATLAB matrix column, or Python column `bixel_index - 1`. |
| `beam_index`, `ray_index`, `beamlet_index` | 1-based beam index, ray within that beam, and beamlet within that ray. |
| `gantry_angle_deg`, `couch_angle_deg` | Angles from the generated native beam. |
| `iso_center_lps_mm`, `sad_mm`, `bixel_width_mm` | Actual isocenter, source-to-axis distance and bixel width, in mm. |
| `source` | Point square selections, sampling seed and spectrum parameters, or phase-space plane, selection square, original-file hashes, species and excluded-particle counts, represented histories and replay normalization. |
| `seed`, `histories` | Random seed and requested histories in the manifest’s source-dependent units. |

For example, after preparation:

```python
import json
from pathlib import Path
manifest = json.loads(Path("projects/patient-example/manifest.json").read_text())
for job in manifest["jobs"]:
    print(job["job_id"], job["beam_index"],
          job["gantry_angle_deg"], job["couch_angle_deg"])
```

A `planning` section also retains automatically generated plan/beam summaries and target names. Its native prescription/fraction fields are recorded only; they do not calibrate dose or change forward weights. All recorded geometry comes from native steering, including target-derived isocenters, rather than manually entered annotations.

Job records are included in the run fingerprint. Existing manifests are not rewritten; prepare a new run directory to obtain these expanded records. Collection uses the saved records.

## MATLAB exchange

```matlab
addpath('matlab');
verify_matrad('projects/patient-cluster-smoke/derived/result.mat');
s = load('projects/patient-cluster-smoke/derived/result.mat');
A = s.dij.physicalDose{1};  % units are recorded in derived/metadata.json
w = ones(size(A,2),1);     % photons (point) or original histories (phase_space)
cube = reshape(A*w, s.dij.doseGrid.dimensions);
```

Scripts use native `to_matrad()` serializers and `scipy.io.savemat`. Native conversion handles voxel ordering and 1-based structure voxel indices. Only exported `dij.beamNum`, `rayNum`, and `bixelNum` receive `+1` for pyRadPlan 0.5.0's label issue; native objects stay unchanged. Native matRad export currently requires square transverse CT/dose grids. The TOPAS engine itself can score non-square grids.

`steering.mat` contains `ct`, `cst`, `pln`, `stf`; `result.mat` adds `dij`. `metadata.json` preserves preparation metadata. `collect_metadata.json` and `forward_metadata.json` record collection provenance, integrity checks, normalization and forward weights when applicable. Forward dose is saved as `dose.mha`.

## Native planning and extension

`patient_workflow.py` groups editable settings by purpose, followed by shared planning helpers and four clearly labeled stage functions:

- `inspect(project_dir)` lists structures and, when a target is selected, displays setup estimates without writing simulation inputs.
- `prepare(project_dir)` imports the patient, generates native steering once, and prepares one run or the configured shift setups.
- `collect(project_dir)` assembles dose matrices and reports from saved bundles.
- `forward(project_dir)` applies `FORWARD_WEIGHT_PER_BIXEL` in saved column order and writes summed dose and reports.

`main()` only parses arguments and dispatches to the selected function. Native `PhotonPlan` configuration remains visible in `configure_plan()`, and `generate_stf()` in `build_steering()`. Collection and forward reconstruction do not import original patient data or read current planning/geometry settings. Small ROI bookkeeping utilities remain in `minibeam/workflow/`; they do not replace native import or rasterization.

`patient_workflow.py` uses native `load_patient`, `PhotonLINAC`, `PhotonPlan` and `generate_stf`. Saved-run collection builds native `Dij` directly; native `calc_dose_influence` and `calc_dose_forward` remain supported by the adapter. Importing `minibeam` registers the distinct `TOPASPhoton` engine; native particle `TOPAS` remains unchanged. `prepare_jobs()` generates configurations; `collect_results()` returns native `Dij`; `collect_forward(weights)` streams beamlet contributions. Native dose methods require an already prepared matching bundle and completed results.

The point model emits photons from `beam.iso_center + beam.source_point` toward uniformly sampled positions in each native square bixel on the common isocenter plane. Squares use lower-inclusive, upper-exclusive boundaries; overlapping regions are rejected. Directions are validated after conversion to the TOPAS binary representation. The empirical spectrum remains user-supplied, not a commissioned machine model. Point particles are generated locally in streaming chunks and replayed by standard TOPAS `PhaseSpace` sources; this is distinct from selecting recorded accelerator phase-space particles. The cluster still needs only TOPAS. Replay is single-use, with one independent photon per point-source history. There is no MU calibration. Configured hardware attenuates the incident photons; scattering and cross-bixel dose are allowed.

The unchanged reference Schneider table, OpenTOPAS license and provenance live in `minibeam/materials/data/` and are copied into bundle inputs. TOPAS applies its native DICOM pixel rescaling and Schneider conversion; this package does not round or rewrite patient pixels. The reference table is an example rather than scanner-specific calibration; `material_file` can replace it. See the included provenance for the upstream source.

For separate jobs, dose is **TOPAS Sum / active histories**: Gy per primary photon for point sources, or Gy per original accelerator history for phase-space sources. Active histories must match the requested histories, including empty histories. Point-source independent beamlet variance is weighted with squared photon counts. Phase-space forward-dose standard errors are withheld because cross-column covariance is not estimated; per-column variance is retained. `derived/variance_of_mean.npz` stores sparse variance separately from native `Dij`. Matrix assembly has a configurable memory budget. With coarse scoring, viewing dose is resampled to CT while uncertainty stays on its scoring grid.

### Hardware configuration

Edit **`minibeam/geometry/config.toml`** to control the treatment head. Both patient and water workflows load it automatically; no new top-level script settings are required.

```toml
[assembly]
enabled = true

[field]
width_mm = 100.0
height_mm = 100.0

[aperture]
enabled = true
type = "slits"
```

This is an excerpt: retain the other keys in the supplied file. Field width and height are full openings **at isocenter**, independently configurable. They set the X-moving MLC and Y-moving jaws. They do not select more bixels: native target-covering or single-bixel steering still determines the incident jobs. Devices stay fixed across all rays of a beam. A beamlet can be partially blocked or strongly attenuated; forward weights still describe upstream source exposure (photons or original accelerator histories), not transmitted particles.

`[mlc]`, `[jaws]`, and `[aperture]` each have an `enabled` switch. The MLC is two continuous banks, not individual leaves. Rounded tips preserve the previous generator's 64-segment polygon approximation, source-tangent field edges, radius and axial offset; `round_tip_enabled = false` selects flat focused banks. Slit blades preserve their physical normal thickness; exact entrance spacing differs slightly from the nominal width-plus-thickness spacing. The report lists actual openings, spacings and edge air gaps.

All `source_to_center_mm` distances are measured from the nominal focal point. Local Z is distance minus native beam SAD, with isocenter at zero. Changing SAD recalculates field-edge focusing and placement. Aperture X/Y rotations rigidly rotate its already-focused assembly about its center; they do not re-focus its blades. Original phantom SSD/air-gap settings are not imported.

The imported defaults are tungsten MLC/jaws at 450/550 mm and a brass slit assembly at 707 mm. **These are reference hardware dimensions, not a placement fitted to this patient.** With the supplied patient's current four beam angles and CT box, the 707 mm slit placement overlaps the CT transport box and preparation is rejected. Do not treat CT air voxels as empty world geometry: the CT box is still a Geant4 volume. Choose physically justified hardware dimensions/positions, or disable the aperture for a jaw-only study. The writer does not move hardware or crop CT automatically.

Preflight checks cover stage order, openings, rounded-tip tangency, slit/blade containment, and conservative rotated device-envelope/CT-box collisions. They may reject a configuration whose detailed solid would clear the CT box; they are deliberately conservative. TOPAS additionally runs sampled Geant4 overlap checks with 10,000 points and quits on detected overlaps in non-Qt batch execution. Qt may continue to permit inspection; examine its overlap messages. Neither check is a mathematical proof of every solid intersection.

When patient preparation encounters a CT/head collision, it exits without a traceback and saves diagnostics in a new sibling folder named `<run-name>-diagnostics-<unique suffix>`. No runnable job bundle is written and any existing run remains untouched. `geometry.pdf` contains the patient projections and collision pages for **all** beams; `collisions.txt` summarizes the findings; `collisions.json` records component bounds and points inside confirmed intersections in DICOM LPS millimeters. The folder also contains one exact `geometry-beam-NNN.toml` snapshot per beam.

Detailed diagnostics check convex material solids (both banks, each brass frame section, and every slit blade), separately from conservative envelopes. Rounded MLC banks use the same polygon approximation as the writer. The reported intersection inradius is the radius of a ball inside both volumes, not penetration depth. A slit-envelope intersection is significant even when its brass clears: the enclosing `Collimator` box, including its daughter air cavity, is itself a TOPAS volume.

The requested **1000 mm SAD / 393 mm collimator-to-isocenter** arrangement was investigated with source distances **425/510/607 mm** for MLC/jaws/slits and a **158 mm** slit outer width. This candidate clears internal hardware checks but its frame and blades still intersect this patient's CT transport box at all four configured angles. It is **not adopted as a default**. The crop investigation retains all 51 ROIs (including `CouchSurface` and `CouchInterior`) and every voxel above -950 HU as a necessary bounding region; even that region still collides. This threshold is only a diagnostic retention rule, not authorization to remove lower-HU material. No crop is applied. Any future crop must separately validate patient/couch retention, physical coordinates, scoring and native matRad alignment.

Configured-head runs record immutable per-beam snapshots and resolved positions in `inputs/beam_geometry.json` and the manifest, with resolved device parameters in each job. Native steering enriched from TOML also retains `inputs/geometry-config.toml`. Patient runs also create **`derived/geometry.pdf`** in fixed DICOM LPS coordinates: X increases toward Left, Y toward Posterior, and Z toward Superior. Patient Z is not the beam axis. The patient remains fixed while sources and enabled hardware follow the native gantry/couch rotations.

The patient PDF starts with an axial overview and anatomy close-up for each distinct isocenter slice level, followed by a 3D overview and per-beam axial/coronal/sagittal projections. CT slices use a fixed -1000 to 1000 HU grayscale display, with BODY/EXTERNAL contours in green and selected TARGET contours in pink when available. The CT center and isocenter are separate markers. The dashed CT box denotes the complete transport boundary, not the patient surface. Hardware outlines use actual transformed vertices; hardware and beam paths are **projections**, not intersections with the displayed CT slice. For nonzero couch angles, the 3D view shows the out-of-plane direction. All panel axes use equal physical units, though overview and close-up magnifications differ. The 3D close-up shows the CT boundary and beam paths without hardware or an anatomy surface.

Water reports retain their existing beam-local schematic. Collision diagnostics retain all-beam detail pages even when patient preparation fails. Custom device fragments outside the configured treatment head are not rendered. Existing reports are not overwritten. Reports are excluded from simulation-input hashes; implementation updates may still change the bundle fingerprint. Editing the source TOML affects new preparations; CLI collection uses saved geometry. Prepare into a fresh directory when geometry changes.

For Python integrations, `TOPASPhotonEngine(plan, devices=())` explicitly disables automatic devices; a supplied device tuple replaces them. `geometry=GeometryInclude(...)` also bypasses the configured head. For native factory calls, `plan.prop_dose_calc["geometry_config"]` can specify an alternate TOML path, or `False` for a device-free calculation; omit it to load the packaged default. Never combine explicit `geometry=` and `devices=` arguments.

The current `aperture.type` is `"slits"`; unsupported names fail. Future implementations belong under `geometry/apertures/` and register in `APERTURE_TYPES`. They supply `config_keys`, `validate_config`, `resolve`, `render`, `envelope`, `drawing`, `draw_entrance`, and `summary`. The report delegates aperture-specific illustrations and text to that implementation; common numeric/material validation remains in the configuration loader. No alternative aperture placeholders are provided. The old generator's phase-space source, phantom, variance reduction and custom scoring are not imported.

### Source and device contracts

`TOPASPhotonEngine(plan, source_model=..., devices=(...))` accepts source and device implementations. `geometry=GeometryInclude(text, assets)` remains a raw TOPAS escape hatch; combining it with `devices` is rejected. The default source is `PointBeamletSource(energy_model=EmpiricalSpectrum())`.

- `JobContext` supplies native Beam, Ray and Beamlet objects, the patient-center translation, and 1-based job indices. Job identity does not select particles on its own.
- A source's `describe(context)` returns `SourceDescription`: coordinate-frame meaning, finite spatial bounds in patient-centered TOPAS world mm, TOPAS source name, supported job modes, normalization and history independence. World sizing uses these bounds, not an assumed focal spot or bixel width.
- `render(context)` returns `ParameterFragment(text, assets, record, bounds_topas_mm)`. Its text contains the source's own TOPAS parameters. The writer adds the seed, requested histories for the declared source name, and dose scoring; it does not inject analytical `Beam*` parameters. Records must contain JSON-compatible values.
- Energy models render their specification and sample energies for point-source replay. `EmpiricalSpectrum` samples the supplied discrete probabilities locally, without interpolation, independently of the native beamlet energy. Native pyRadPlan 0.5.0 defaults to a scalar energy of 6.0; the geometry-only machine supports that nominal label and the workflows omit an explicit steering-energy override. The scalar is retained in native exports but must not be interpreted as the simulated photon energy. `MonoenergeticEnergy` remains explicitly injectable for tests; it is not the workflow default.
- Assets are `("inputs/filename", Path(...))` pairs. The built-in sources declare paired TOPAS particle/header files. Assets are hashed and copied as files without loading them into memory. Destinations must be unique flat paths below `inputs/`, excluding generated inputs. A shared asset may be reused across jobs by its owner; conflicting ownership or files at the same destination are rejected.
- Each device has a unique `name` and `render(context)`. Its fragment defines `Ge/<name>/Type`, `Parent = "BeamFrame"`, and explicit `TransX`, `TransY`, `TransZ` in mm. Additional child components must also have unique names. Provide world bounds for components extending beyond the source/patient envelope. Device tuple order determines text composition; physical positions determine placement.

`BeamFrame` is fixed across all rays in a beam: its origin is the isocenter, local +Z is the central beam direction, and +X follows native beam orientation. The built-in `Source` frame uses the same fixed beam orientation; per-bixel directions are carried by the replay records. Devices should use `BeamFrame`, so they do not turn independently with each ray.

Separate bixel and combined beam jobs are supported by the square point and recorded phase-space sources. See the normalization table below; unsupported custom source job modes fail before writing a bundle. A serializable `source_config` in `pln.prop_dose_calc` reconstructs either source for native dose calls. Do not combine it with an injected `source_model`; remove `source_config` explicitly before injecting a custom source or monoenergetic point model. All package Python modules are fingerprinted recursively, together with rendered parameters and declared asset hashes. Changes invalidate cached calculations, and the fingerprint appears in each scorer name.

### Future physics implementations

The supported Elekta reader and square-bixel selector live in `sources/iaea.py` and `sources/phase_space.py`. Additional file layouts require explicit parsers and history/weight validation. Add spectral models alongside `sources/energy.py` and devices under `geometry/`. Individual-leaf MLCs, full-field replay and optimization remain future work. New normalization conventions or reuse strategies must account for original versus recorded histories and correlations before enabling dose or uncertainty reporting.

## Repository layout and existing runs

Root scripts remain the user entry points. The installable package is `minibeam`, with no `src/` directory or old package alias. Package discovery is restricted to `minibeam` and its subpackages.

| File | Responsibility |
|---|---|
| `minibeam/__init__.py` | Register `TOPASPhoton`; expose `TOPASPhotonEngine` and `ResultsPendingError`. |
| `engines/topas.py` | Adapt native pyRadPlan dose calls and engine settings to preparation and result collection. |
| `sources/beamlet.py`, `bixels.py` | Generate square point-source replay and define shared disjoint bixel selections. |
| `sources/energy.py` | Render the discrete spectrum; retain an injectable monoenergetic test utility. |
| `sources/iaea.py` | Stream and validate the supported IAEA binary layout; write TOPAS binary headers. |
| `sources/phase_space.py` | Select square bixels, preserve history grouping and particle weights, and render fixed-plane replay sources. |
| `sources/empirical.py` | User-supplied energy points and relative weights, preserved unchanged. |
| `geometry/coordinates.py` | Patient translation, scoring grid and rotation conventions. |
| `geometry/patient.py` | Validate/copy-list original CT inputs; render water or DICOM patient geometry. |
| `geometry/devices.py` | Fixed beam frame and raw geometry include contract. |
| `geometry/config.toml`, `configuration.py` | Editable hardware values and configuration snapshots. |
| `geometry/assembly.py`, `mlc_jaws.py` | Resolve and render the field-defining hardware. |
| `geometry/apertures/` | Focused slits and the aperture implementation registry. |
| `geometry/spatial.py`, `report.py`, `patient_report.py` | Bounds/collision transforms, water schematics, and patient-centered axial/3D PDFs. |
| `topas/contracts.py` | Run settings, job context, source description and parameter fragments. |
| `topas/writer.py` | Compose fragments, validate assets, create portable jobs and fingerprints. |
| `topas/manifest.py` | Hash files, record native planning parameters and validate bundle integrity. |
| `topas/scoring.py` | Render scorer parameters and strictly parse dose CSV files. |
| `topas/results.py` | Assemble sparse native `Dij` or streamed weighted dose and variance. |
| `materials/data/` | Packaged Schneider table, license and provenance. |
| `workflow/patient.py` | Small target-selection and original-ROI bookkeeping utilities. |

Paths in the table after the first row are relative to `minibeam/`. `matlab/` contains verification utilities and `tests/` contains automated checks. DICOM inputs, `.venv`, generated runs and caches are ignored by Git.

New bundles use **schema version 3**, recording execution mode, job membership, matrix-column mapping and normalization denominators. Schema-2 bundles remain collectable using their saved conventions; schema 1 is unsupported. Prepare a fresh directory for the new point model or combined execution. Never mix results between preparations. Existing bundles are left untouched.

## Water example and verification

```sh
python water_phantom.py prepare --bundle runs/water-cluster --histories 3000
# Transfer the bundle; execute both job parameter files from its root on the cluster.
# Transfer results back, then:
python water_phantom.py collect --bundle runs/water-cluster --histories 3000
python water_phantom.py forward --bundle runs/water-cluster --histories 3000
```

The water example has square transverse dimensions and an asymmetric off-center target. Tests cover native steering, target selection, matRad alignment/nonmutation, configuration mismatch, direct CSV import, normalization, relocation and incomplete/malformed output. Extension tests also check point-source geometry and spectrum sampling, exercise source-owned asset pairs and energy composition, verify fixed device frames, and reject unsupported normalization and job modes.

```sh
python -m pytest -q
PHOTON_TPS_DICOM=dicom_9306087_fine python -m pytest -q -m patient
# With the TOPAS environment loaded:
PHOTON_TPS_TOPAS=/absolute/path/to/topas python -m pytest -q -m topas
```

Hardware regression tests use `tests/fixtures/reference_geometry.toml`, independently of your editable `minibeam/geometry/config.toml`. Phase-space tests cover history gaps, particle species/weights, half-open boundaries, malformed inputs, per-column normalization, withheld combined uncertainty, relocated assets, and replay of an exhaustive synthetic bixel partition.

For the supplied Elekta patient replay test, also set `PHOTON_TPS_PHASE_SPACE` to the original basename (without either suffix). That test uses one million original histories, one 45-degree beam, a coarse scoring grid and explicitly disabled hardware. It scans the complete original file and requires approximately half a minute of local transport.

Optional transport tests invoke TOPAS directly from test code, including a synthetic heterogeneous DICOM and the original patient CT. Set both environment variables to include the coarse-grid single-patient-beamlet test. Local OpenTOPAS 4.2.p3 transport is validated; your external cluster and MATLAB execution remain unverified.

## Per-beam hardware and the shifted-slit patient study

`patient_workflow.py` uses `ENABLE_COLLIMATOR` to control whether the slit
collimator exists. Set it to `False` for one run without the brass frame, air
cavity, or blades. You can leave `COLLIMATOR_SHIFT_FRACTIONS` unchanged:

```python
ENABLE_COLLIMATOR = False
COLLIMATOR_SHIFT_FRACTIONS = [0.0, 0.25, 0.50, 0.75]  # inactive until re-enabled
```

Disabled runs write directly into `--project` (default `projects/patient`). Shift,
slit-width and nominal-CTC overrides are ignored without validation, and the
script prints that they are inactive. The MLC, jaws, source and steering remain
controlled by their existing settings. Re-enabling restores the configured shifts.

`ENABLE_COLLIMATOR` must be a Boolean and overrides TOML's `aperture.enabled`
without changing the file. Requesting a collimator when TOML's entire assembly
is disabled raises a clear conflict. The selected TOML is loaded once; the switch
is passed explicitly to a helper, applied to typed geometry snapshots and attached
to native beams. Writers, collision checks, reports and MATLAB exports use these
same authoritative snapshots. Disabled manifests record `aperture.enabled=false`
and no resolved aperture; inactive shifts are not recorded as executed setups.

When `ENABLE_COLLIMATOR = True` (the default), the array selects the layout below.
The `--study` flag has been removed.

| Array | Behavior | Default output root |
|---|---|---|
| `[]` | One run directly in the root, using TOML geometry without slit overrides | `projects/patient` |
| `[0.0]` | One explicitly unshifted collimator setup in `shift_000/` | `projects/patient-slit-study` |
| `[0.0, 0.25, 0.50, 0.75]` | Four setups in the existing shift folders | `projects/patient-slit-study` |

The four-value array remains the default. Every setup uses the shared gantry/couch
angles and **always honors `ONLY_CENTRAL_BEAMLET`**. Its default remains `True`, giving four
central-bixel jobs per setup (16 jobs across four shifts). Set it to `False` for
native target coverage. Steering is generated once and copied across shifts.

When enabled, an empty array preserves the configured slit width, CTC and lateral shift from
`GEOMETRY_CONFIG`; it does not apply the script's slit overrides. A nonempty array
applies `SLIT_ENTRANCE_WIDTH_MM`, `NOMINAL_ENTRANCE_CTC_MM`, and each translation.
Thus `[]` and `[0.0]` intentionally differ. `GEOMETRY_CONFIG` selects the base TOML
in both cases.

`inspect` always lists ROIs. When a target is configured it also generates native
steering and prints setup/output estimates; otherwise it finishes after the list.
Inspection writes no run files.

```sh
python patient_workflow.py inspect
python patient_workflow.py prepare --project projects/my-slit-study
# Run each generated jobs/bixel_*.txt externally from its setup directory.
python patient_workflow.py collect --project projects/my-slit-study
python patient_workflow.py forward --project projects/my-slit-study
```

Collection discovers single runs and shift setups from `manifest.json` or `study.json`.
It ignores the current switch, shifts, source path, bixel mode and other planning settings.
Reproducing the earlier 278-job-per-setup study for a new preparation requires
`ONLY_CENTRAL_BEAMLET = False`. Existing schema-2 runs remain collectable after implementation changes.

Edit the study constants in the COLLIMATOR SETUPS section of `patient_workflow.py` for slit entrance
width, nominal entrance CTC, and `COLLIMATOR_SHIFT_FRACTIONS`; beam angles use the shared planning settings.
`COLLIMATOR_SHIFT_FRACTIONS` translates only the slit collimator along beam-frame X,
as fractions of `NOMINAL_ENTRANCE_CTC_MM`. The MLC and jaws stay fixed. `GEOMETRY_CONFIG`
selects the starting TOML; `None` uses `minibeam/geometry/config.toml`. Other
hardware dimensions and positions come from that snapshot. These defaults use
2 mm entrance slits and retain the configured 4 mm physical blades. The 6 mm CTC
is **nominal**: constant blade thickness normal to each focused blade produces
slightly varying entrance pitches, recorded explicitly in the outputs.

| Setup directory | Fraction of nominal CTC | Rigid translation along beam X |
|---|---:|---:|
| `shift_000` | 0% | 0 mm |
| `shift_025` | 25% | 1.5 mm |
| `shift_050` | 50% | 3 mm |
| `shift_075` | 75% | 4.5 mm |

The whole brass frame, air cavity and blade pattern move together in the fixed
beam frame. Gantry/couch rotation maps this translation into patient LPS.
Blade shapes and focusing angles remain unchanged; rays, source, isocenter, MLC
and jaws stay fixed. This is a translation study, not a new focused aperture.

`inspect` imports the patient and generates steering for a read-only workload
estimate. `prepare` generates steering once and copies it for each setup, then
writes portable inputs and native MATLAB snapshots. `study.json` records settings,
resolved beam geometry, directories, and preparation status. Failed clearance
checks produce separate diagnostic PDFs/JSON/text and no runnable overlapping
bundle. `collect` builds each setup's sparse dose matrix; `forward` streams its
weighted columns. Set `FORWARD_WEIGHT_PER_BIXEL` in `patient_workflow.py`; their units are
photons for point sources or original accelerator histories for phase-space
sources. No exposure or dose is combined across the four setups automatically.

With `ONLY_CENTRAL_BEAMLET = False`, the supplied patient has 278 jobs per setup (1,112 total across four shifts).
At native CT scoring resolution, estimated CSV output is **481.697 GiB per setup**,
or **1,926.787 GiB total**. These are output estimates, not input-bundle sizes.
The corresponding dense matrices would occupy 38.536 GiB each; collection uses
sparse matrices but can still be large. Coarse scoring is an explicit change to
`DOSE_SPACING_MM` in the patient settings; it does not alter CT transport geometry.

### Typed geometry on native steering

The extension types are `MinibeamBeam(Beam)` and
`MinibeamSteeringInformation(SteeringInformation)`. Native `generate_stf()` still
determines rays and bixels; enrichment attaches independent immutable geometry
snapshots. Use validated replacements to edit a snapshot:

```python
from minibeam import enrich_stf
from minibeam.geometry.configuration import GeometryConfig
from minibeam.geometry.models import BeamGeometry
from minibeam.steering import validate_minibeam_stf, matrad_steering

geometry = BeamGeometry.from_config(GeometryConfig.load())
stf = enrich_stf(native_stf, geometry)
stf.beams[0].geometry = stf.beams[0].geometry.replace(
    aperture={"lateral_shift_mm": 1.5})
# Dictionary reconstruction must use the extension validator:
restored = validate_minibeam_stf(stf.model_dump(exclude_computed_fields=True))
```

The attached geometry is authoritative at preparation. Editing TOML afterward
cannot change an attached beam. Explicit `geometry=` or `devices=` injection
cannot be combined with attached geometry. Ordinary native steering remains
supported and is enriched internally from configured defaults. Geometry is
validated again at preparation to catch unchecked model-copy updates.

Live extended objects work through native pyRadPlan dose functions. **Do not
pass extension dictionaries into the native dictionary validator:** pyRadPlan
0.5.0 drops unknown fields. Reconstruct with `validate_minibeam_stf()` first.
MATLAB export strips extension fields before calling native `to_matrad()` and
adds a separate `beam_geometry` variable in the same file. Its 1-based
`beam_index` reflects current beam ordering. Use `matrad_steering(stf)` rather
than calling the inherited extended serializer directly. Edited MATLAB imports
and general full-steering JSON persistence are not supported.

Every new manifest and `inputs/beam_geometry.json` records resolved per-beam
settings, actual pitches, LPS translations and component centers. Jobs retain
stable bixel identities and their parent beam's resolved device parameters.
Geometry and implementation changes invalidate fingerprints: prepare a fresh
study root after changing settings or upgrading this implementation. Existing
runs are left untouched. Cluster execution must still be verified on the cluster.

### Collection from saved runs and indexing reports

`collect` and `forward` require an explicit `--project`. Pass either a complete single
bundle or the study root: every setup listed in `study.json` is processed independently.
Neither stage imports original DICOM, generates steering, reads current TOML geometry,
nor opens the original accelerator phase-space file. Keep the complete bundle, including
`derived/steering.mat`; collection builds `result.mat` from that preparation snapshot.
Forward preserves `steering.mat` and writes the summed `dose.mha` on the CT grid.

Manifest integrity, bundled input hashes, scorer identities, grids, normalization,
histories, statistics, duplicate bins and completeness remain checked. A code-version
change alone does not invalidate completed simulations: differences are recorded in
`derived/collect_metadata.json` or `derived/forward_metadata.json`. Newly prepared
planning snapshots have a SHA-256/bundle association in `derived/planning_snapshot.json`.
Legacy snapshots undergo structural checks but are explicitly reported as unhashed;
those checks cannot establish the original contents of an old planning snapshot.

Each successful stage writes `derived/indexing.md` beside its dose artifact. It includes
separate loading examples for `steering.mat` and `result.mat`, descriptions of their
variables, beam/ray/beamlet and hardware access, and column-to-TOPAS-job mapping.
Copyable MATLAB examples extract a single beamlet dose, reconstruct a weighted dose
cube, and find an ROI by name to build its CT mask. The guide explains voxel ordering,
CT versus scoring grids, source-dependent exposure units, empty planning settings,
uncertainty arrays, and MHA LPS coordinates. Original DICOM ROI identifiers in
`metadata.json` are distinguished from native `cst` structure numbers. Matrix examples
in a forward-stage report explicitly require a separately completed collect stage.
For a study, each shift receives its own report and starts numbering at one.
Derived files are replaced atomically; preparation snapshots, manifests, inputs and
simulation CSV files are preserved. The same saved-run collection supports separate and combined execution; it never reconstructs original source inputs.

## Separate bixels or one combined job per beam

Edit the patient workflow settings:

```python
ONLY_CENTRAL_BEAMLET = False           # target-selected coverage; True keeps only the central bixel
BEAMLET_EXECUTION = "combined"  # default: "separate"
```

Combined mode uses the union of selected bixels, not an unrestricted rectangular
field. It supports exactly one native beamlet per ray. Sources and devices retain
the same fixed beam frame; MLC, jaws, collimator shifts and selected rays are unchanged.
Each configured shift remains its own independent run. TOPAS job filenames are
`jobs/beam_000001.txt`, etc. Run them from the bundle root as usual. Newly prepared bundles set
`i:Ts/ShowHistoryCountAtInterval = 100000` in `inputs/common.txt` to limit
history progress messages to every 100,000 histories.

| Source / execution | Transported histories per job | Divide scored sum by | Forward coefficient |
|---|---|---|---|
| Point / separate | N photons | N | photons for that bixel |
| Point / combined | N photons total over selected bixels | N | total photons for that beam |
| Recorded phase space / either | N original accelerator histories | N | original-history exposure |

Here N is `HISTORIES_PER_JOB`: no multiplication by the number of bixels.
Combined point jobs sample a uniform-area mixture over the selected squares.
Member photon counts fluctuate statistically and sum to exactly N; small budgets
can leave some squares unsampled. They sample area, not solid angle.
Binary replay costs 34 bytes per emitted point photon (plus small headers).
Preparation prints the total and checks TOPAS's integer limit before generation.

Prepare fresh directories for this change. Existing bundles retain their saved
history counts and normalization, including older combined point runs normalized
per member bixel; collection does not reinterpret them.

For phase space, each downstream record is classified on the common isocenter
plane and included at most once per beam. Union filtering preserves particles,
weights, multi-particle histories and empty original histories. A requested member
with no selected particles remains an error. Different gantry beams can reuse
original records; cross-beam combined forward uncertainty remains unavailable.

Combined collection returns one sparse `Dij` column per beam, plus explicit
`column_mapping` records in the manifest and MATLAB snapshots. Native `stf` is
preserved. `beamNum` identifies the beam; exported `rayNum` and `bixelNum` are **0
sentinels**, not native ray/bixel indices (−1 in Python). Do not feed aggregate
columns into code that assumes one independently adjustable column per native
bixel. `derived/indexing.md` provides the aggregate mapping and MATLAB examples.

`FORWARD_WEIGHT_PER_BIXEL` remains the exposure setting: a scalar applies to all
saved columns, and a vector has one entry per saved column. For an aggregate,
that exposure applies equally to every member bixel. Native forward-dose calls
also accept a native bixel vector only when all members of each aggregate have
identical weights; unequal weights require separate simulations.

Point combined variance uses TOPAS's history variance divided by the total job
history count. Bixel membership is sampled statistically as part of the uniform-area mixture.
The source energy extension now supplies `sample(context, rng, count)` as well as
`render(context)`; the built-in empirical and monoenergetic models support both.
The water entry point retains its CLI and separate-job default; no full-field
water/SSD redesign is included.

### Patient project directory migration

The patient workflow now uses `--project` instead of `--run-dir`. Relative and
absolute paths are accepted as written; `projects/` is not automatically prepended.
Existing directories are not moved and remain usable:

```sh
python patient_workflow.py collect --project runs/patient-dose
```

New default outputs go into `projects/patient` or `projects/patient-slit-study`.
Both `projects/` and legacy `runs/` are ignored by Git. Internal bundle metadata
and shift folder names are unchanged. The water workflow retains `--bundle`
and its `runs/water-cluster` default.
