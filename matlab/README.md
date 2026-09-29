# Minimal dose-optimization handoff

Share these MATLAB files together with the collected project's
`derived/result.mat`, `derived/metadata.json`, and `derived/indexing.md`.
Keep `manifest.json` for full job provenance. Each study setup is independent.
Original DICOM files are not needed for this example: `result.mat` already
contains the original `ct`, `cst`, planning information, and calculated `dij`.
Patient CT and structures are sensitive data; use your lab's approved transfer route.

```matlab
addpath('matlab');
test_roi_on_dose_grid;  % Small synthetic mapping checks.
e = dose_optimization_example('derived/result.mat', 'PTV2017fw');

% Supply one source exposure per saved matrix column.
w = ones(size(e.D,2),1); % Demonstration values, not a treatment prescription.
doseInROI = e.D(e.roiRows,:) * w;
doseCube = reshape(full(e.D*w), size(e.roiMask));
```

To list names or map another structure directly:

```matlab
s = load('derived/result.mat');
disp(s.cst(:,2));
[mask, coverage] = roi_on_dose_grid(s.ct,s.cst,s.dij,'PTV2017fw');
rows = find(mask);
Aroi = s.dij.physicalDose{1}(rows,:);
```

`cst{r,2}` is the exact ROI name; `cst{r,4}{1}` contains 1-based indices on the
**original CT grid** for the first nominal scenario. These are not dose-row
indices when cropping or resolution differs. The helper reconstructs the CT mask
and samples it at dose-bin centers using nearest-neighbor interpolation in physical
LPS coordinates. It uses the origins (`cubeCoordOffset`), spacing and directions in `dij.ctGrid` and
`dij.doseGrid`, so it does not manually add crop offsets or scale linear indices.
Grid dimensions are XYZ; returned MATLAB masks/cubes are `[Y,X,Z]`. `find(mask)`
therefore selects the corresponding rows of `physicalDose{1}`.

The matrix stays on its original scoring grid and stays sparse. Converting only
`D*w` to a full vector avoids densifying the influence matrix. Finer mask sampling
does not create more anatomical detail than the original CT-grid segmentation;
this example does not calculate partial-volume fractions.

Check `coverage` before defining objectives. It reports original ROI voxel centers
inside the scoring grid, their fraction, and the mapped dose-bin count. This is a
discrete coverage measure, not an exact geometric volume fraction. An empty original
ROI has a NaN coverage fraction. A small ROI may disappear under coarse sampling.
Clipped anatomy has no dose rows: it is **unscored**, not zero dose. Statistics on
`Aroi*w` describe only the represented part of the ROI. Do not silently interpret
them as a whole-ROI DVH or objective when coverage is incomplete.

Matrix units and weight units are in `metadata.json` and `indexing.md`:

- Point source: Gy per primary photon; weights are photon counts.
- Phase space: Gy per original accelerator history; weights are original-history
  exposures, including empty histories. No MU calibration is implied.
- Separate execution: independently weighted beamlet columns.
- Combined execution: aggregate beam columns; individual bixel fluences cannot
  be optimized independently. Prepare and collect separate jobs for that purpose.

This is a data-access example, not an optimizer. It does not change simulation,
normalization, saved CT/structures, or dose matrices. Review any scorer warnings
in the handed-over report before using results. Shared phase-space histories may
correlate columns; this example does not propagate their covariance.

## Visual alignment check

Run the viewer in MATLAB R2020a or newer (no matRad or image-processing toolbox):

```matlab
addpath('matlab');
file = 'derived/result.mat';
visualize_dose_check(file, 'PTV2017fw');

% Isolate column 1: one beamlet in separate mode, one beam in combined mode.
s = load(file, 'dij');
w = zeros(size(s.dij.physicalDose{1},2),1); w(1)=1;
visualize_dose_check(file, 'PTV2017fw', w, [], 'dose_check.png');

% Optional explicit slice intersection in LPS millimeters:
% visualize_dose_check(file, 'PTV2017fw', w, [-5.6 93.2 169.5]);
```

The three panels are XY, XZ and YZ **slices**, not maximum projections. The default
reference is the original ROI centroid (CT center for an empty ROI). Green contours
come directly from the original CT-grid ROI; magenta dashed contours come from the
mapped dose-grid ROI. They should agree up to sampling differences. CT uses a fixed
-1000 to 1000 HU window; dose uses one shared linear hot color scale. Cyan dotted
lines mark scoring boundaries. Outside coverage, only CT is displayed, not zero
dose. Full CT display extents are retained. Physical axes increase toward
Left, Posterior and Superior. The cyan cross is the chosen reference point,
not necessarily the treatment isocenter.

Try an asymmetric OAR as well as the target to detect flips. Inspect multiple
slice positions: a centroid slice need not show every clipped region. Partial
coverage prints a warning. All-ones weights are only an illustration; use the
saved normalization units and review scorer warnings in `indexing.md`. This viewer
checks spatial alignment and reconstruction, not absolute dose calibration or
transport physics. It explicitly rejects non-identity grid directions, matching
the current axial patient workflow, instead of silently mislabeling oblique axes.

`test_visualize_dose_check` creates synthetic cropped/finer-grid examples in both
execution modes and an all-zero-dose case. MATLAB execution and visual inspection
of these figures must be performed on a machine with MATLAB installed.
