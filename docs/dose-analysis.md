# Water dose analysis from MHA

Edit `tools/dose_analysis.toml`, then run from the repository root in the installed Python environment:

```sh
python tools/dose_analysis.py pdd --field 10
python tools/dose_analysis.py lateral --field 10 --combine-depths
python tools/dose_analysis.py lateral --field 10 --mc-only --combine-depths
python tools/dose_analysis.py mc-fields-ratio --field 10
python tools/dose_analysis.py data-fields-ratio --field 10
python tools/dose_analysis.py output-factor
python tools/dose_analysis.py pvdr-depth --field 10
python tools/dose_analysis.py report
```

Use `--help` after a command for its options. `--config FILE` selects another TOML; `--output-dir DIR` redirects artifacts. `report --output-dir DIR` rebuilds the Markdown gallery without reading dose or measurement inputs. Each successful analysis refreshes it automatically.

The shell launcher accepts the same arguments:

```sh
./tools/dose_analysis.sh lateral --field 10
./tools/dose_analysis.sh pdd --field 10
```

It uses the active virtual environment or Conda environment, otherwise the repository's `.venv`, then `python3`. Set `PYTHON_BIN=/path/to/python` to override this choice. The launcher works from any directory; relative command-line paths remain relative to your current directory.

## Inputs and coordinates

The supplied configuration references the original measurements in `~/Local/vroc-photon-minibeam/datasets/` without copying them. Paths expand `~`; other relative paths are relative to the TOML file. Measurement MU settings (400 open, 2000 slits), measurement depths, centering options and profile presets are retained. Update these paths on other computers.

The 10 cm open-field MC input is `projects/openfield-10x10/derived/dose_scoring_grid.mha`. Other MC fields and collimator inputs are deliberately unset: add explicit `mc_open`/`mc_collimator` paths as results become available. Commands needing them stop with a configuration error. `data-fields-ratio` needs only measurements, even when no MC files exist. Output factors require both open MC and measurement inputs for every configured field, including the 10 cm reference; a separate TOML can select the fields available for comparison.

Supported images are scalar, identity-direction scoring-grid MHA files from a single water beam along physical +Y (zero gantry and couch). The reader checks the adjacent `forward_metadata.json`, associated setup `manifest.json`, grid, saved water entrance and source placement, units, and exposure weights. Keep these files together when transferring results. Individual batch outputs find the shared setup manifest. Patient images, rotated beams and mismatched/resampled images are rejected. This is a consistency check, not a cryptographic verification of dose-file contents.

| Analysis coordinate | Physical coordinate, relative to saved entrance center |
|---|---|
| X | MHA X |
| Y | MHA Z |
| Depth in water (mm) | MHA Y |

SimpleITK's NumPy order `[Z,Y,X]` becomes `[analysis X, analysis Y, depth]`. Voxel-center positions and dose samples are preserved. A finer scoring grid is supported independently of the original phantom CT resolution. There is no axis-swap option. Measurement X maps to analysis X; measurement Y maps to analysis Y. The packaged slit orientation modulates along analysis Y; `--mc-axis` and `--data-axis` choose profiles, not coordinate swaps.

`--mc-shift-x-mm` and `--mc-shift-y-mm` translate lateral MC coordinates before ROI extraction and normalization, without resampling dose. Slit measurement maps use coordinate-midpoint centering unless explicit `slit_map_x_offset_mm`/`slit_map_y_offset_mm` values override it.

Lateral comparisons include a lower MC/measurement ratio panel fixed to 0.95–1.05. It uses the same normalized MC profile as the upper panel, interpolated onto measurement coordinates without extrapolation. Nonpositive measurement values are omitted. Ratios outside the displayed range remain in `profiles.csv`; the plot limits do not change the data. Both separate-depth and combined-depth figures include this panel; MC-only figures do not.

## Methods and normalization

- **PDD:** MC-only shows saved Gy. Open-field comparison independently peak-normalizes MC and AGL sampled curves; Sphinx comparison separately matches the reference-depth ROI. `--agl-ssd-correction` displays an approximate inverse-square SSD shape correction; it does not correct scatter, field size or electron contamination. No collimator PDD measurement is substituted; use `--source collimator --mc-only`.
- **Lateral:** perpendicular ROI averaging at the nearest scoring-bin depth; CSV/JSON record requested and actual depths. Measurement maps must exist at the requested depth. `--normalization-mode 2d|x|y` controls reference matching; one scale applies to all requested profiles/depths. `--combine-depths` combines figures. These are depth slices, not maximum projections.
- **Field ratios:** collimator/open profiles with interpolation only within supported finite coordinate intervals and a configurable denominator cutoff. MC removes each saved common exposure weight; measurements divide by their configured MU values.
- **Output factors:** reference-depth ROI dose per exposure (MC) or per MU (measurement), each normalized to its own 10 cm field. Reference interpolation never extrapolates. All fields must use compatible source-normalization units and common reference/ROI settings.
- **PVDR:** sampled peak/valley detection, optional expected CTC, prominence and edge-peak controls. Saves diagnostic profiles, retained/excluded features, mean peaks/valleys and reasons for unavailable PVDR. `--fields 2 4 10` or `--all-fields` adds an aggregate plot.

Saved MHA values are already dose in **Gy** after exposure weighting. They must **not** be divided by simulated history counts again. All input images require finite positive common exposure weights (per dose column). Removing this common weight recovers Gy/original accelerator history or Gy/primary photon; those conventions cannot be mixed in quantitative comparisons. Batch averages remain averages: no batch-count scaling is applied. This does not provide absolute MU calibration.

Optional historical reference curves are commented, attributed configuration entries. They are not universal standards or automatically applied corrections. The historical output-factor curve applies only at 100 mm depth; the historical point-source PVDR series is a 10 cm aggregate overlay.

## Outputs and limitations

Artifacts go under `outputs/dose_analysis/field_<N>cm/<command>/<source>/` (aggregate commands omit the field folder): PNG, numerical CSV and `metadata.json`. `report.md` links comparison figures, values and metadata. The original gallery grouping is retained: open lateral/PDD/output factor, then collimator lateral/PVDR. MC-only, ratio and peak/valley diagnostic figures remain in their command folders rather than the main gallery.

Metadata records explicit inputs, coordinate mapping, normalization, exposure, source selection, saved batches, scorer warnings and analysis settings. Review-required scorer warnings are printed and carried into the Markdown report; analysis does not declare those results valid. Input dose images and simulations are never rewritten.

MHA alone supplies no MC uncertainty, and voxel covariance needed for ROI-averaged uncertainty is unavailable. No MC uncertainty bands are invented. Measurement bands show **spatial spread within the ROI**, not measurement uncertainty. PVDR bands/error bars show **retained-feature variation**, not Monte Carlo or measurement uncertainty. Repeated transport batches share their replay sample; their statistical variation does not represent independent accelerator phase-space sampling.

Open-field measurement maps support `open_map_x_offset_mm` and `open_map_y_offset_mm` in `[defaults]` or a field section. Corrected coordinates are CSV coordinates **minus** these offsets, so a positive offset moves the profile toward negative coordinates. Omitted open offsets mean zero (no automatic centering). The offsets apply before ROI extraction and normalization in all commands using open measurement maps, including PDD and lateral profiles. They do not change AGL depth coordinates, MC coordinates, or stored dose values. Slit offsets remain independent. Applied offsets are recorded in metadata and printed with the appropriate setting names.
