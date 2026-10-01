# Saved results, dose grids and MATLAB

[Main guide](../README.md) · [MATLAB utility guide](../matlab/README.md)

## Collect or forward independently

```sh
python patient_workflow.py collect --project projects/my-patient
python patient_workflow.py forward --project projects/my-patient
# The same stages are available in water_phantom.py.
```

Both stages read the saved manifest or study index. They do not import original
DICOM, regenerate steering, open original phase-space files, or use current geometry
settings. A study's setup discovery comes from `study.json`, not current arrays.
`forward` can run without `collect`; it reads CSV columns and sums them without
retaining the full sparse matrix. Explicit exposure weights remain active settings.

Validation covers supported schema/normalization, manifest integrity, bundled hashes,
job identities, planning snapshot associations, and CSV scorer fingerprints, grids,
units, histories, statistics, duplicate bins and completeness. Missing/malformed
results fail rather than becoming zero dose. Hash receipts protect new planning
snapshots; legacy unhashed snapshots get structural checks and an integrity notice.
Implementation-version differences are recorded rather than alone blocking collection.

Derived dose/report outputs are written atomically. Preparation snapshots, inputs,
manifest and CSVs remain unchanged. Study collection updates collection status while
preserving preparation records. MATLAB-empty optional optimization/sequencing fields
are not missing beamlets; unselected beamlets are absent from steering.

## Outputs and normalization

| File under `derived/` | Contents |
|---|---|
| `steering.mat` | Prepared native `ct`, `cst`, `pln`, `stf`, resolved `beam_geometry`, plus applicable crop/column metadata. |
| `planning_snapshot.json` | Snapshot hash and bundle association. |
| `result.mat` | Saved planning variables plus calculated native sparse `dij`. |
| `collect_metadata.json` | Collection status, normalization, integrity and warning information. |
| `variance_of_mean.npz` | Per-column dose variance information in native array order. |
| `dose_scoring_grid.mha` | Weighted forward dose on the saved scoring grid. |
| `dose.mha` | Dose reconstructed on the original CT grid. |
| `dose_coverage.mha` | CT-grid scoring coverage, distinguishing unscored space from zero dose. |
| `forward_metadata.json` | Weights, units, diagnostics and plot records. |
| `indexing.md` | Run-specific indexing guide, dimensions, column associations and examples. |
| `dose_beam_001.png`, etc. | Forward beam-view plots, one per saved beam. |

Additional uncertainty files depend on the source convention. Point-source columns
are Gy per primary photon, weighted by photon counts. Phase-space columns are Gy per
original accelerator history, weighted by original-history exposure. Empty histories
belong in the normalization denominator. Neither convention implies MU calibration.
Do not divide again by particle count or summed statistical weights.

Patient `FORWARD_WEIGHT_PER_BIXEL` accepts a scalar or vector in **saved column/job
order**, including combined mode despite the historical name. Water `FORWARD_WEIGHT`
applies to its aggregate field. A scalar is broadcast to saved columns.

Separate mode has one column per bixel. Combined mode has one aggregate column per
beam and explicit `column_mapping` membership. It cannot supply individually weighted
member-bixel doses for later FMO. Use separate jobs if individual fluence variables
are needed. Each study setup's numbering starts independently.

Per-column statistics use the scorer's sums, standard deviations and validated
represented histories. Shared phase-space histories create correlations across
columns; combined forward standard errors are withheld where covariance is unavailable.
Reported statistical uncertainty does not include missing scored energy or model error.

## MATLAB indexing and grids

Use distinct loading variables:

```matlab
planning = load('steering.mat');
result = load('result.mat');
D = result.dij.physicalDose{1};  % nominal scenario; sparse voxel-by-column matrix
w = ones(size(D, 2), 1);       % replace with the required exposure in saved units
doseVector = D * w;
```

`ct` and `cst` retain the original planning CT resolution and extent. Matrix rows use
`dij.doseGrid`, which may be cropped or finer/coarser. MATLAB dose cubes are `[Y,X,Z]`
with 1-based column-major voxel indexing:

```matlab
dimsXYZ = double(result.dij.doseGrid.dimensions(:)');
doseCube = reshape(full(doseVector), dimsXYZ([2 1 3]));
j = 1;
beamletCube = reshape(full(D(:,j)), dimsXYZ([2 1 3]));
row = sub2ind(dimsXYZ([2 1 3]), iy, ix, iz);
```

Consult the generated `indexing.md` for actual saved fields, grids and associations.
Native serializers perform voxel permutations; exported separate-mode `beamNum`,
`rayNum` and `bixelNum` labels are corrected to 1-based values without modifying native
objects. `rayNum` is local to its beam and `bixelNum` local to its ray, not the global
matrix column. Combined columns use aggregate labels and `column_mapping`; do not
interpret their sentinel ray/bixel values as an individual ray.

`stf` is standard native MATLAB steering; hardware lives separately in `beam_geometry`
and is associated by current 1-based beam order. No edited-MATLAB steering importer is
provided. `crop_metadata` records original retained X/Y/Z ranges as zero-based,
stop-exclusive indices and the planning/transport/dose grids. Cropped-away dose bins
have no matrix rows; original CT/structure voxels are still present.

To map a named ROI to the dose grid, use `matlab/roi_on_dose_grid.m` and the example
`matlab/dose_optimization_example.m`. They map original structure indices through
physical coordinates with nearest-neighbor masks and report coverage. `find(roiMask)`
returns MATLAB linear dose-grid indices; original `cst` indices cannot directly index
a different scoring grid. Intentionally clipped ROIs have incomplete coverage, which
is reported rather than silently assigned full coverage.

`matlab/visualize_dose_check.m` overlays CT, dose, original ROI and mapped ROI on three
reference **slices**, not maximum projections. It supports separate and combined
results with appropriate column weights; see its help and the MATLAB guide. The
native pyRadPlan 0.5.0 square-transverse export restriction remains enforced.

## Images, coordinates and warnings

MHA is a medical image format readable by SimpleITK or image viewers. Its origin,
spacing and direction encode physical LPS placement. SimpleITK's NumPy view is
`[Z,Y,X]`, zero-based, different from MATLAB cube ordering. Preserve image metadata
when viewing or resampling; a NumPy index alone is not a physical coordinate.

Forward figures use each beam's own weighted dose, resampled into its transverse
U/V and depth frame. They show maximum dose through the combined selected-target
depth extent, including dose outside the target shape within that slab. Water uses
the whole WATER depth. Background CT is the reference slice nearest the saved
isocenter; contours are projected target outlines. Cyan dots mark all selected bixel
centers, including zero-weight members; a larger cyan plus marks isocenter.

Plots share a linear inferno scale, no positive-dose threshold and no gamma correction.
Physical scales are equal. Coarse-grid coverage is respected and saved dose arrays are
not modified by display resampling. Missing targets, differing source units and scorer
warnings are annotated. Plot failures are reported while preserving completed dose.
The old three-plane `dose_projections.png` renderer is retired; existing image files
are left in place and are not refreshed.

The recognized TOPAS invalid-voxel/unscored-step warning may pass collection only
when its warning block is complete and consistent. For each affected job the program
prints the CSV path, original warning, unscored-step count, unscored MeV, histories
and normalization, and saves structured diagnostics and a review-required notice.
The outcome is “completed with scorer warnings; validity requires user review.”
No lost-dose percentage or validity claim is inferred. Unknown warnings, unexpected
filtering and malformed results still fail.

## Unrestricted phase-space results

`phase_space_selection` in MATLAB and collection/forward metadata records `field`,
`all_forward`, or `not_applicable` for point sources. Legacy phase-space projects
without this field retain their original field-selection meaning.

In `all_forward`, one matrix column represents a whole-source beam dose, including
particles outside the planning bixels. `column_mapping.members` preserves steering
associations for reference; it does not partition that column's incident fluence.
Bixel dots in beam plots are labeled as planning reference positions. The same
selected-target depth slab is used for projection, so these figures can omit dose
outside that slab; inspect the full scoring-grid dose for peripheral contributions.

## Collection speed and progress

Both collect and forward parse CSVs in bounded chunks of about 100,000 rows and
validate the numeric columns with array operations. Every bin, including zero-dose
bins and zero-weight jobs, is still checked for completeness, duplicates, valid
coordinates, histories and consistent statistics. Existing CSV files need no
conversion or new simulation.

Jobs are read sequentially. The terminal shows each job's index, ID, path and file
size, throttled bin progress for long files, and elapsed time at completion.
Collect builds sparse columns; forward accumulates dose without retaining the full
matrix. Chunk buffers are bounded, but dense per-job dose/variance arrays and a
per-grid duplicate-detection mask still scale with grid size. Collection additionally
needs memory for the accumulated sparse matrices.

`TOPAS_THREADS_PER_JOB` controls particle transport, not collection. Parsing speed,
shared-storage throughput, grid size, and sparse MATLAB export can all affect total
runtime; faster CSV parsing does not eliminate those other costs.
