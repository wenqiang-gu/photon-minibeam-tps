# Geometry, steering and sources

[Main guide](../README.md) · [Results](results-and-matlab.md)

## Patient steering

The patient workflow imports native pyRadPlan CT and structures, reports omitted
ROIs, and preserves original ROI identifiers in metadata. Select an exact nonempty
`TARGET` name. Only that structure drives target-covering steering. An explicit
`ISO_CENTER_LPS_MM` overrides its centroid; otherwise the resolved centroid is printed.
LPS means Left, Posterior, Superior, with positions in millimeters.

`ONLY_CENTRAL_BEAMLET=True` selects native `photonSingleBixel`; `False` selects
native `photonIMRT`. The latter projects target voxel centers onto the isocenter
plane, rounds to the bixel-width lattice and keeps unique rays. Additional target
margin is disabled by this workflow. This is candidate selection, not optimization.
Nominal machine energy 6.0 and SCD=SAD/2 are native compatibility parameters;
source energies and physical device positions are configured separately.

`BEAMLET_EXECUTION` is independent of selection:

- `separate`: one job and dose column per selected bixel.
- `combined`: one job and aggregate dose column per beam, containing exactly the
  selected square bixels. It is not necessarily a full square field.

The disjoint squares use lower-inclusive, upper-exclusive boundaries on the common
isocenter plane. Combined selection uses their union without counting boundary
particles twice. Native beam/ray/beamlet identities remain recorded.

## Water field

The water script uses registered `photonSquareField`, derived from native photon
IMRT steering. Only ray-position selection is replaced inside `generate_stf()`;
native transformations and ray construction remain intact.

`FIELD_SIZE_MM` in `water_phantom.py` sets both square-field dimensions at the
water entrance, overriding the selected TOML width/height in the resolved snapshot
without editing that file. It must be positive, finite, and an integer multiple of
`BIXEL_WIDTH_MM`. The square is tiled exactly at the entrance plane.
100 mm / 5 mm gives 20 × 20 centers from −47.5 through +47.5 mm. The whole-water
ROI does not select rays.

This setting controls both steering and MLC/jaw openings. It does not resize
hardware bodies, slit geometry, the phantom or scoring voxels; existing containment
checks remain enabled. `field` and `outside_field` source selection use this square;
`all_forward` remains unrestricted while hardware openings change. Prepare a fresh
project after changing field size. Existing package callers can leave
`WaterSettings.field_size_mm=None` to retain equal TOML width/height.

Water extends from Y=0 to Y=`PHANTOM_SIZE_MM`, centered about X=Z=0, with native
voxel-center origins chosen to preserve these boundaries. At gantry/couch zero the
source is (0, −SSD, 0), and native SAD=SSD because isocenter lies at the surface.
Combined execution is fixed in this workflow. Hardware may block the selected
incident fluence. The water TOML is independent of the patient TOML.

## Source models and history budgets

`SOURCE_TYPE` selects `point` or `phase_space`. Both use the same selected squares.
`HISTORIES_PER_JOB` is a **total job budget**, not automatically multiplied by the
number of combined bixels. TOPAS recycling remains disabled.

Point mode uses the empirical discrete spectrum in `sources/empirical.py`. Photons
originate at the nominal focal point with uniform fluence over the selected square
area at isocenter. The recorded spectrum is independent of nominal steering energy.
Normalization is Gy per primary photon. `MonoenergeticEnergy` remains available
through the source-model injection interface for explicit use and tests.

Phase-space mode takes an ordered `PHASE_SPACE_FILE_BASES` list, each basename with
`.header` and `.phsp` files. The implemented IAEA reader supports the explicitly
validated little-endian 33-byte layout with constant Z, history increments and LATCH.
The supplied source plane is 272.1 mm downstream of the nominal focal point. Replay
uses the fixed beam frame; it is not separately aimed at each bixel.

Files must represent distinct original-history batches with compatible geometry,
source/transport descriptions and coordinate conventions. Duplicate paths/content
are rejected, but hashes cannot prove statistical independence. Shared-history
particle splits are unsupported. The supplied four parts declare 170 million
original histories each: 680 million total, not 680 billion.

All listed pairs are validated even if the prefix ends in an earlier file. Consume
histories sequentially in list order; 200 million uses 170 million from part1 and
30 million from part2. Requests beyond the total fail. Gaps, trailing empty histories,
particle grouping, species, energies, directions and statistical weights are retained.
Empty-history counts are retained in normalization metadata, not transported as empty events.
Downstream particle trajectories are projected onto the selection plane. A selected
bixel with no particles is rejected; increase histories or check coverage.

Only selected portable TOPAS replay pairs are bundled; identical selections are shared.
`PhaseSpaceMultipleUse = 1` is required by the current normalization assumptions.
New projects use guarded replay (`nonempty_guard_v1`) to avoid locally reproduced
OpenTOPAS 4.2.p3 end-of-file/empty-history replay errors. The replay contains selected
physical particles plus one skipped empty guard record, with
`PhaseSpaceIncludeEmptyHistories = False`. TOPAS transports **N nonempty histories**;
`HISTORIES_PER_JOB` still requests **H original histories**, including those that
became empty after filtering. The guard is not an accelerator history or a dose
contribution. Its header declares N+1 histories (N reached the plane), while the
TOPAS run requests N events. The point-source replay uses the same guard with N=H.
Do not remove the guard or independently change the run/header history counts.
Dose is Gy per **represented original accelerator history**, including empty histories,
not per selected particle or sum of particle weights. A history may contribute to
several bixels through different particles. Cross-column covariance is not estimated.
Start small validation runs with 1,000,000 original histories; narrow fields can still
need more. Do not edit recycling/history parameters in a prepared project.

## Hardware and studies

`GEOMETRY_CONFIG` selects TOML; `None` uses the packaged patient or water configuration.
Native steering is enriched with immutable per-beam geometry. Attached snapshots are
authoritative for parameters, collisions, reports, manifests and MATLAB exports;
preparation does not overwrite them by rereading TOML.

The packaged reference has MLC **body center 356.8 mm**, jaws 470.5 mm and slit
center 707 mm from the source. Elekta's *Agility BLD* tutorial (2021), PDF pages
9 and 17, places the MLC tip-curvature center at **349.3 mm**. With 90 mm
thickness and the retained -7.5 mm offset, the entrance and exit are 311.8 and
401.8 mm. The body-center setting applies to both simple banks and individual
leaves; projected widths and tip tangency are recalculated at this placement.
Custom configurations and saved snapshots retain their explicit distances; fresh
preparation is required to apply the corrected packaged reference. Inspect the
selected TOML before changing hardware. No patient clearance is assumed.
All conservative transport-volume and hardware checks remain enabled. Moving a
focused collimator's source-to-center distance currently regenerates focused blade
angles; it is not a rigid relocation of a fixed manufactured collimator.

`ENABLE_COLLIMATOR=False` omits the full slit assembly, leaves MLC/jaws unchanged,
and produces one patient run directly in the chosen project. Rotation, shift and
slit overrides are inactive and are not validated. Enabling an aperture while the
whole assembly is disabled is a configuration conflict.

For enabled patient collimators:

- `COLLIMATOR_ROTATION_DEG` contains additional rotations relative to TOML Z.
  The packaged baseline is −90°, so offsets 0°, 45°, 90° resolve to −90°, −45°, 0°.
  An empty rotation list preserves TOML. Custom TOML missing Z defaults to 0°.
- With zero X/Y tilt, zero additional rotation under the packaged baseline puts
  slit length along beam X between the banks. Positive shift is then beam +Y;
  a 90° additional rotation makes positive shift beam +X.
- Rotate the entire frame/cavity/blades first, then translate along rotated
  aperture-local X. X/Y tilts participate in that full 3D transform.
- `COLLIMATOR_SHIFT_FRACTIONS` multiplies `NOMINAL_ENTRANCE_CTC_MM`. With nominal
  CTC 6 mm, fractions 0, .25, .5, .75 translate 0, 1.5, 3, 4.5 mm. Actual entrance
  pitches vary with blade focusing; physical blade thickness is retained.
- Nonempty shift lists apply the editable slit width/nominal CTC; empty shifts
  retain TOML dimensions and translation. Rotation and shift arrays form a
  Cartesian product, rotation first. Every setup has all configured beams.

Folders are `rotation_045_shift_025`, `rotation_045`, or `shift_025` according to
active overrides; no overrides use the root. Negative and decimal names use `m`
and `p`, e.g. `rotation_m045` and `rotation_022p5`. Duplicate/name-colliding settings
are rejected. Empty rotations and `[0.0]` have identical physical orientation but
different folder labels. The water workflow has one configured setup, no study arrays.

## Transport crop and scoring grids

Patient `CT_CROP_VOXELS` is `None` or original CT X/Y/Z ranges, zero-based and
stop-exclusive. `x=(40,300)` retains 40..299, corresponding to MATLAB 41..300 and
inclusive TOPAS limits 41..300. These are not DICOM InstanceNumber values.
Original DICOM, native CT and structures are unchanged.

Three grids are saved: original planning CT, retained transport CT and dose scoring.
Transport uses TOPAS voxel restrictions without moving retained physical voxels.
`DOSE_SPACING_MM=None` uses native spacing; `(dx,dy,dz)` requests independent finer
or coarser scoring. Whole-bin fitting preserves outer boundaries and reports actual
spacing. Native MATLAB export under pyRadPlan 0.5.0 still requires square transverse
CT and dose grids. Axial CT is the supported transport orientation.

`ENFORCE_CT_CROP_PROTECTION=True` rejects any ROI clipping and material flags.
`False` allows non-target clipping but reports every clipped ROI and material finding.
Selected targets remain protected. Invalid grids and geometry collisions always fail.
Material flags include HU > −950 and low-HU enclosure/connectivity findings; these
are conservative air heuristics, not an exact material classifier. The removed ROI
allow-list and material-only switch produce migration errors in new preparation;
legacy saved metadata retains its original policy.

Inspection/preparation writes a crop preview and clipping summary to a sibling
`<project>-crop-preview` directory before anatomical rejection. Full CT reference
slices and whole-volume diagnostic projections distinguish target conflicts,
clipped ROIs and material flags. Anatomy outside the retained scoring region remains
in `cst` but has no dose rows; coverage is explicitly reported.

## Reports, visualization and extensions

Patient geometry PDFs use fixed LPS axial/3D views and transformed hardware.
Water overviews use beam-local longitudinal projections of complete frame/blade
solids; transparency reveals overlap rather than artificial slit gaps. The entrance
aperture detail is in its own local coordinates before assembly rotation.

Dimension arrows label the distance from the nominal focal point to the MLC and
jaw **entrance surfaces** (`source_to_center_mm - thickness_mm / 2`). Water
reports show them in both longitudinal panels; patient detail pages use the
projection with the longest visible beam axis. Labels retain true axial distances
even when the drawing is foreshortened. These distances do not start at the
phase-space replay plane. Disabled devices have no arrows; body and curvature
centers remain separately identified in the geometry records.

`ENABLE_OPENGL` adds axes, the isocenter CT slice and material-free cyan CT boundary
for DICOM geometry; cropped boundaries are also shown. Disable interactive viewing
for cluster batch runs. Display components do not change transport material or scoring.

Source models own their parameter fragments, declared assets, coordinate frame,
spatial extent, job-mode capability and normalization. Unsupported modes/conventions
fail explicitly. Assets are copied/hashed in chunks. Device components are ordered,
uniquely named and positioned; `geometry=` remains an escape hatch, mutually
exclusive with `devices=`. Attached geometry cannot compete with injected devices.

Future spectra, phase spaces and MLC/apertures belong in their respective source or
geometry modules. New phase-space normalization must address weights, original versus
recorded histories, empty histories, reuse and inter-job correlations. No FMO,
sequencing, clinical machine calibration or additional aperture type is implemented.

## Phase-space field selection versus unrestricted replay

Both root workflows expose `PHASE_SPACE_SELECTION = "field"`. This setting is
inactive for point sources. `field` retains the existing projected-square filter:
combined jobs use the union of selected bixels, while separate jobs use each bixel.
For patients, selected bixels may follow the target rather than fill a rectangle.

Set `PHASE_SPACE_SELECTION = "all_forward"` to replay every downstream-going
photon, electron and positron in the requested original-history prefix. Patient
`BEAMLET_EXECUTION` must be `"combined"`; water already uses combined execution.
The MLC, jaws and collimator now physically define irradiation. Native bixels are
planning references only and do not restrict incident fluence. Without suitable
hardware, this setting can irradiate a much larger region than the intended field.

Both modes include scatter generated during TOPAS transport. The difference is
which incoming particles enter transport. Backward and tangent records are excluded
in both modes and reported. No absent upstream radiation is reconstructed. Particle
properties and empty histories are preserved; dose remains Gy per original
accelerator history, never per selected particle. Replay uses each file once.

For a paired comparison, keep geometry, source files, original-history prefix,
exposure weights and seed unchanged. Prepare `projects/water-field` with `field`,
then `projects/water-all-forward` with `all_forward`; execute and reconstruct each
project separately. The difference includes direct and scattered contributions
from previously excluded particles. Shared histories correlate the two estimates;
a quantitative uncertainty for their difference is not currently provided.

### Outside-field phase-space diagnostic

Both workflows accept `PHASE_SPACE_SELECTION = "outside_field"` with combined
execution. It retains downstream particles whose straight-line projections lie
outside the union of selected bixels. Water uses its square entrance field;
patients use their selected native bixels, not the physical target volume.
The half-open square boundaries exactly complement `"field"`. Backward/tangent
particles remain excluded. Point sources ignore this setting.

Use a fresh project with the same source files, history prefix, hardware and dose
grid as the corresponding `field` and `all_forward` projects. For example:

```sh
# Set PHASE_SPACE_SELECTION = "outside_field" in water_phantom.py first.
python water_phantom.py prepare --project projects/openfield-10x10-outside
```

No source recycling is added. Particle weights and original histories (including
empty ones) retain the same normalization. Each beam has one aggregate dose column,
representing its outside-field incident-particle contribution. Listed bixels and
plot dots describe the excluded selection region. Changing the setting requires
fresh preparation; saved projects retain their existing behavior.

After TOPAS execution, collect or reconstruct forward dose normally. Compare
`all_forward ≈ field + outside_field` using equal original-history exposure weights,
without independently fitting each result to measurements. Batches are averaged,
not summed. Transport scatter exists in all three modes; this decomposition
classifies incident particles by projection, not by whether their dose arose from
scatter. Target-depth maximum-projection plots can omit dose elsewhere; use the
full scoring-grid dose for quantitative additivity checks. No paired-difference
uncertainty is currently supplied.

## Effective source and spectrum modifications

The standalone modifier creates spectrum-reweighted and effective focal-spot
variants before source selection. See [phase-space modifications](phase-space-modifications.md)
for configuration, illustrations, normalization and use of the generated IAEA
files in either workflow.


### Native Y-shaped diaphragms

Both packaged TOMLs select `[jaws].model = "y_shaped"`. The rounded front band
spans the **full body width**. A thin upstream base remains beside the taper and
rear spine. This is an Agility-inspired approximation, not a verified EVO
manufacturing model. Eight non-overlapping native `G4ExtrudedSolid` pieces (four
per jaw at the default thickness) represent the base, front, shoulders and spine;
no STL files or custom TOPAS extension are required.

```toml
[jaws]
model = "y_shaped"
outline_reference_distance_mm = 1000.0
outline_points_mm = [
    [0.0,   205.0], # full-width front
    [30.0,  205.0], # taper begins
    [210.0,  30.0], # narrow spine begins
    [350.0,  30.0], # rear endpoint
]
thin_region_thickness_mm = 35.0
# Retain enabled, source_to_center_mm, thickness_mm and tip settings.
```

Each point is `[distance behind nominal radiation edge, thick-region half-width]`
at the **fixed drawing reference distance**, not the current beam SAD/SSD.
The front half-width defines the full body width; the final distance defines the
nominal length of one jaw. The thin base retains that full width to the rear.
No separate outer width, height or length is accepted for `y_shaped`.

Physical drawing dimensions equal projected dimensions multiplied by
`source_to_center_mm / outline_reference_distance_mm`. At 470.5/1000:
`2 × 205 = 410 mm` projected width becomes **192.905 mm** physical width;
350 mm projected nominal length becomes **164.675 mm**. Treatment SAD/SSD changes
tip tangency and the nominal opening, but does not resize these hardware dimensions.
Changing the body-center distance or the drawing reference explicitly changes the
outline's physical magnification. The rear boundary follows the nominal field edge,
not a fixed pair envelope; curved-front material lengths vary through the thickness.

The total physical thickness is 77 mm in both packaged configurations. At the
470.5 mm body midpoint the entrance/exit distances are **432/509 mm**. The
35 mm upstream base ends at **467 mm**. Radius 135 mm and offset -3.5 mm place
the tip-curvature center at 467 mm. A common 64-segment front profile is split at
the base interface without changing its surface. TOPAS, collision diagnostics and
reports consume the same convex-piece vertices. Reports include a dedicated jaw
page showing projections and a rear-spine section.

Sources and assumptions:

- Elekta *Agility BLD* tutorial (2021), PDF pages 11, 16, 66 and 70: rounded-tip
  dimensions and projected diaphragm outline. The four coordinates above are
  approximate graph readings. The 350 mm graph endpoint does **not** establish
  the manufactured jaw length; we adopt it as a configurable assumption.
- The 35 mm thin thickness and upstream placement are interpretations of page 16,
  not a confirmed side-region specification. They are independently configurable.
- [Elekta patent US20100166150A1, Figures 5–7](https://patents.google.com/patent/US20100166150A1/en)
  explicitly describes a thick front edge across the full diaphragm width. Its
  preferred 20 mm thin-region example is not a confirmed production Agility dimension.
- Karl N. Sachse (2019), *Development and validation of an electron Monte Carlo
  model for an Elekta Synergy linear accelerator*, §3.1.1.2.3, Table 3-15,
  [PDF page 83](https://scholar.ufs.ac.za/bitstreams/ecf5466d-0410-4e41-8194-c11ac02b5f56/download#page=83):
  radius 135 mm; offset derived as `467 - (432 + 77/2) = -3.5 mm`.

The material remains `G4_W`. This does not implement the alloy, Agility radiation-edge
calibration or automatic MLC constraints behind thin regions. MLC positions are
unchanged: a thin region without leaf shielding can transmit radiation. Compare
transport before treating this approximation as a model of the measured machine.

### MLC and jaw parameter conventions

Beam X is MLC travel; beam Y is jaw travel; beam Z points downstream. Physical
lengths below are millimeters. The workflow field reference is the patient
isocenter or the water entrance plane. It is distinct from the fixed jaw drawing
reference.

| Setting | Meaning and applicability |
|---|---|
| `[field].width_mm`, `height_mm` | Full **projected** X/Y opening at workflow SAD/SSD; not hardware body dimensions. Water `FIELD_SIZE_MM` overrides both. |
| `[field].material` | TOPAS material name for MLC and jaws. |
| `[field].side_margin_mm` | **Physical** minimum travel-axis material margin beyond the opening; no automatic body enlargement. |
| `enabled` | Include this device independently of other devices; the assembly switch must also be enabled. |
| `source_to_center_mm` | **Physical axial source distance** to the body midpoint, not tip-curvature center. |
| `thickness_mm` | **Physical** axial body thickness; for Y jaws, total thickness of thick regions. |
| `round_tip_enabled` | Rounded-tip tangency versus the existing flat-tip construction. |
| `tip_radius_mm` | **Physical** radius in the travel/Z plane; active with rounded tips. |
| `tip_center_z_offset_mm` | **Physical signed** curvature-center offset from body center; negative is upstream. |
| MLC `model` | `simple`: two continuous banks; `leaves`: rigid individual leaves. Legacy `banks` aliases `simple`. |
| MLC `outer_width_mm` | **Physical full pair-envelope X width**, including opening, for `simple`; does not truncate rigid translating leaves. |
| MLC `outer_height_mm` | **Physical full Y span** for `simple`; also the transverse containment limit for `leaves`. |
| `leaf_length_mm` | **Physical** rigid length of one individual leaf along X; `leaves` only. |
| `minimum_leaf_gap_mm` | **Physical** minimum opposing-tip separation at the bank, not projected and not an interleaf gap; `leaves` only. |
| `shape_source`, `shape_file` | Static CSV/RTPLAN selection and path for `leaves`; relative paths resolve against the TOML. |
| `shape_beam_number` | RTPLAN DICOM BeamNumber identifier, not workflow beam order. |
| `shape_control_point_index` | Zero-based RTPLAN control-point index; positions may be inherited. |
| Jaw `model` | `y_shaped`: sculpted native pieces; `uniform`: legacy fixed-envelope banks. |
| Jaw `outer_width_mm`, `outer_height_mm` | **Physical full pair-envelope X/Y dimensions**, including opening; `uniform` only. |
| `outline_reference_distance_mm` | **Fixed source-to-drawing-plane distance**, independent of workflow SAD/SSD; `y_shaped` only. |
| `outline_points_mm` | **Projected** `[rearward distance, X half-width]` at that fixed plane. Full width is twice the first half-width; last distance sets nominal per-jaw length. |
| `thin_region_thickness_mm` | **Physical** upstream axial thickness, not projected; `y_shaped` only. |

CSV offsets are nominal **reference-plane mm**, while its bank positions and field
sizes are **reference-plane cm**. RTPLAN MLCX boundaries/positions are nominal
**reference-plane mm**. Neither supplies physical tungsten-tip coordinates or a
verified radiation-edge calibration. Leaf widths are projected to the MLC plane;
rigid leaf length, thickness, radius and enforced tip gap remain physical.

### Uniform and legacy geometry

For an explicit uniform jaw, remove the three Y-shape settings and provide:

```toml
model = "uniform"
outer_width_mm = 230.0
outer_height_mm = 310.0
# Retain enabled, placement, total thickness and tip settings.
```

These are fixed **pair envelopes including the opening**, not one jaw's length.
Their outer edges stay fixed as the opening changes. Packaged MLC simple banks
retain their approximate 310 × 160 mm envelope. These envelope estimates are not
verified manufacturer dimensions.

Older configurations missing the jaw model retain uniform behavior; missing
rounded-tip options retain flat jaws. Shared legacy `[field].outer_*` dimensions
still migrate to both devices; mixed shared/device dimensions are rejected.
Do not combine Y-outline settings with legacy jaw outer dimensions. Saved projects,
inputs and hashes are not rewritten. Use fresh projects after changing geometry;
collection and forward reconstruction continue to use saved data independently.

## Static individual MLC leaves

In the selected geometry TOML, set `[mlc].model = "leaves"` to replace the
continuous banks. `model = "simple"` is the default (two continuous blocks).
The older `"banks"` name remains accepted as a compatibility alias. With
`shape_source = "csv"` and `shape_file = "mlc_shape_example.csv"`, the supplied
`mlc_shape_example.csv` is used. A custom path is relative to the TOML file.
CSV leaf offsets are millimeters; bank positions and field sizes are centimeters
at the planning reference plane. They are nominal edges, not curvature centers.

For DICOM use `shape_source = "rtplan"`, set `shape_file` to an RTPLAN, and
select `shape_beam_number` (DICOM BeamNumber) and `shape_control_point_index`
(zero-based). Only MLCX is supported. Earlier control-point positions are inherited.
This reads a static shape only, not delivery sequencing, gantry or head rotations.
All workflow beams use this same shape in their native fixed hardware frames.

`leaf_length_mm = 155.0` is an approximate rigid length based on the supplied
Elekta slide. Thickness, tip radius, material and placement retain their existing
settings. Leaf side widths are projected at the MLC midpoint and extruded; leaf
side focusing, tongue-and-groove and calibrated dosimetric offsets are not modeled.
The fixed continuous-bank X envelope does not limit translating rigid leaves;
actual leaf bounds determine the collision envelope. Transverse height is checked.
Rounded tips use source-ray tangency; a deficient physical gap is corrected by
symmetric translation (see below). Requested edges, adjustments, polygons and input hash are recorded
in TOPAS device metadata. Original leaf files are not required for saved collection.
No leaf-position arrays are added to native steering, and bixel selection is unchanged.
Use a fresh project and inspect geometry before transport. The screenshot-derived
CSV is an example, not a verified machine calibration.

MLC shape files are read only during preparation. Generated TOPAS solids need no
CSV or RTPLAN on the cluster. Paths are relative to the selected TOML, not the
working directory; an absolute path works regardless of the TOML location.
There is no special `packaged` filename keyword. Package callers constructing
geometry directly from dictionaries should supply an absolute shape-file path.

### Minimum physical leaf-tip gap

The patient and water TOMLs set `[mlc].minimum_leaf_gap_mm = 1.0`, following
Elekta's *Agility BLD* tutorial (2021), PDF pages 75–76, “Closed Leaf Constraint.”
This is the minimum transverse separation of opposing tips **at the physical MLC
bank**, not an isocenter-plane opening and not the 90 μm gap between adjacent
leaves in one bank. It applies only to `model = "leaves"`.

Every opposing pair is checked through the leaf thickness, including the
curvature-center depth. If its physical separation is too small, both complete
leaves move outward equally to reach the minimum. This can widen a small requested
opening; pairs already satisfying the limit are unchanged. The midpoint, rigid
leaf length and tip shape are preserved. This does not implement Elekta's
radiation-edge calibration or controller-specific closed-position conversion.

Preparation and geometry PDFs report the minimum, adjusted-pair count and largest
per-leaf translation. TOPAS device records retain each pair's original/resolved
physical gap and each leaf's signed translation. TOPAS and collision geometry use
the same resolved solids. The setting changes the preparation fingerprint.

Zero restores overlap-only correction. Missing settings in older configurations
or geometry snapshots default to zero. Existing projects are unchanged and
collection/forward continue to use saved information. Prepare a fresh project to
apply a different gap; all other hardware and source settings remain independent.

## TOPAS replay PreCheck

Both patient and water scripts expose `TOPAS_PHASE_SPACE_PRECHECK = True`.
The Boolean is passed through planning settings into the native plan's
`prop_dose_calc["phase_space_precheck"]`; direct engine callers use
`phase_space_precheck=True` by default. It controls the generated
`b:So/Beam/PhaseSpacePreCheck` parameter for phase-space sources and for the
binary replay generated by the point-source model, in separate and combined modes.

Keep it enabled for the first TOPAS use of a newly generated or modified replay
pair. Disabling it for an unchanged, previously checked pair avoids the extra
startup scan; verify transfer integrity if moving files to a cluster. This does
not disable Python source validation, input hashes, replay guard checks, CSV
validation or geometry overlap checks. PreCheck is not physics validation.

TOPAS **requires** PreCheck when `PhaseSpaceIncludeEmptyHistories=True`, even if
the pair passed a previous check: the scan determines whether empty histories
are in the file or need appending. Our generated guarded replay keeps that
setting `False`, transports only nonempty histories, and preserves original-history
normalization during collection. There is no new empty-history toggle.
See the [OpenTOPAS source](https://github.com/OpenTOPAS/OpenTOPAS/blob/main/primary/TsSourcePhaseSpace.cc)
and [phase-space documentation](https://opentopas.readthedocs.io/en/main/parameters/source/phasespace.html).

The resolved PreCheck setting is saved in new manifests and planning snapshots
and participates in preparation fingerprints. Use fresh preparation to change
it; do not edit existing job files. Appended batches inherit the saved value.
Legacy projects retain their original enabled setting, and saved-only collection
and forward reconstruction ignore the current workflow switch.
