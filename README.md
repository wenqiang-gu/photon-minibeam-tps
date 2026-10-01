# photon-minibeam-tps

Native pyRadPlan 0.5.0 planning with a `TOPASPhoton` extension for OpenTOPAS 4.2.p3.
Prepare portable simulation inputs, run TOPAS externally, then collect sparse native
`Dij` matrices or reconstruct weighted forward dose. No optimizer or MU calibration
is implemented.


## Planned development

The following features are planned but not yet complete. Priorities may change
as the patient and water workflows are evaluated.

- [ ] Compare "field" with "all_forward" for the water phantom simulation
- [ ] Add water-phantom analysis of depth-dose curves, lateral profiles, and peak-to-valley dose ratios.
- [ ] Benchmark calculated dose against measurements across field sizes, slit widths, and depths.
- [ ] Split simulations into batches and combine their dose results, with explicit random-seed management and history normalization.
- [ ] Validate phase-space files generated from a tuned Monte Carlo accelerator model.
- [ ] Propagate uncertainty for shared phase-space histories, accounting for correlations between dose-matrix columns.
- [ ] Calibrate dose to measured output and monitor units under clearly defined reference conditions.
- [ ] Implement fluence-map optimization using separate beamlet dose matrices, with target and organ-at-risk objectives.
- [ ] Add MLC leaf shift for beamlet delivery (and recalcualte dose for dose validation?)
- [ ] How to convert an optimized fluence map to deliverable MLC and other weights?
- [ ] Calculate dose delivery with realistic geometry (MLC shape) and non-beamlet particles


## Install

Use Python 3.12–3.14 (tested locally with 3.13). Dependencies are declared in
`pyproject.toml` and installed by pip:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# Optional test dependencies:
python -m pip install -e '.[test]'
```

Run the following commands from the repository root. For later sessions, activate
`.venv` again; alternatively use your own compatible Python environment.

## Patient workflow

Edit the settings in `patient_workflow.py`, especially DICOM location, target, source
files, histories and geometry. Use the imported ROI name, e.g. `PTV2017fw`, without
its displayed ROI number.

```sh
python patient_workflow.py inspect
python patient_workflow.py prepare --project projects/my-patient
# Run generated TOPAS inputs externally and return the CSV outputs.
python patient_workflow.py collect --project projects/my-patient
python patient_workflow.py forward --project projects/my-patient
```

| Stage | Purpose |
|---|---|
| `inspect` | List patient structures, preview configured cropping, and estimate setups; water inspection prints phantom/field settings. No TOPAS source inputs are generated. |
| `prepare` | Generate native steering, validate geometry, create TOPAS inputs, geometry PDF and planning snapshot. Does not run TOPAS. |
| `collect` | Validate saved inputs/results and assemble a sparse dose matrix and indexing guide. |
| `forward` | Stream weighted saved dose columns, write dose images and beam-view figures. Does not require a preceding `collect`. |

`prepare`, `collect` and `forward` require `--project`. The argument is a relative
or absolute path; `projects/` is not automatically prepended. Inspection defaults
to `projects/patient` or `projects/water-phantom`. A patient crop preview is written
to the sibling `<project-name>-crop-preview` directory, including when validation
fails. Collection and forward use saved projects, independently of current planning
settings and original DICOM/phase-space files.

## Water workflow

Edit `water_phantom.py`; hardware settings are independently configured in
`minibeam/geometry/water_config.toml`.

```sh
python water_phantom.py inspect
python water_phantom.py prepare --project projects/my-water
# Run the single combined TOPAS job externally, then:
python water_phantom.py collect --project projects/my-water
python water_phantom.py forward --project projects/my-water
```

The default is a 200 mm cube, 2 mm native voxels, SSD 1000 mm, and a perpendicular
beam along +Y. Planning isocenter is the entrance-surface center. A 100 mm square
field with 5 mm bixels contains 400 bixels simulated in one combined job. `WATER`
is a whole-phantom calculation ROI, not a tumor or the beamlet-selection mask.

Both workflows support `PHASE_SPACE_SELECTION="field"` (default) or
`"all_forward"` for unrestricted downstream phase-space replay in combined mode.
See the [geometry/source guide](docs/geometry-and-sources.md) for comparison semantics.

## Execute TOPAS

Each job runs from its prepared project (or study setup) root:

```sh
cd /cluster/path/to/project
topas jobs/beam_000001.txt   # combined mode; use the filename in manifest.json
```

Direct TOPAS execution needs no Python. The supplied optional Slurm submission
helper requires Python 3.8+ and the standard library, but not pyRadPlan. It submits
jobs; it does not install TOPAS. See the [cluster guide](docs/cluster.md).

## Project files

```text
project/
├── manifest.json     # Hashed inputs, grids, jobs, normalization and provenance
├── inputs/           # DICOM or phantom parameters, devices, materials, replay files
├── jobs/             # TOPAS parameter files
├── results/          # Returned TOPAS CSV files
└── derived/          # steering.mat, geometry.pdf, collected/forward outputs and reports
```

Patient rotation/shift studies have a saved `study.json` and one such project per
setup. Collect/forward accept the study root and discover its saved setups.

## Detailed guides

- [Geometry and sources](docs/geometry-and-sources.md): selection, phase space,
  normalization, rotations/shifts, cropping, scoring and extension interfaces.
- [Cluster execution](docs/cluster.md): portable files, Slurm options, retries,
  threads, transfer and submission records.
- [Results and MATLAB](docs/results-and-matlab.md): grids, indices, combined columns,
  weights, uncertainty, plots and saved-project validation.
- [MATLAB utilities](matlab/README.md): optimization handoff and visual checks.

## Code organization

The two root scripts contain editable settings, one copied settings snapshot, short
stage functions, and CLI dispatch. Native planning lives in `minibeam/workflow/patient.py`
and `water.py`. Plain settings dataclasses do not replace native patient objects.
Shared planning helpers and CLI code have no dependency on root-script globals.

`workflow/artifacts.py` prepares inputs and writes planning snapshots;
`workflow/collection.py` handles saved-result stages. `engines/`, `sources/`,
`geometry/`, `topas/` and `materials/` retain their distinct responsibilities.
`matlab/` provides interchange tools, `tests/` holds tests, and `tools/` contains
manual diagnostics such as:

```sh
python tools/print_rtplan_info.py /path/to/RTPLAN.dcm --summary-only
```

Only root submission scripts under `projects/` are tracked. DICOM, project outputs,
legacy `runs/`, environments and caches are ignored.
