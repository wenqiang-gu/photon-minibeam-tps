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

Field width and height in the water TOML must agree, and width must be an integer
multiple of `BIXEL_WIDTH_MM`. The square is tiled exactly at the entrance plane.
100 mm / 5 mm gives 20 × 20 centers from −47.5 through +47.5 mm. The whole-water
ROI does not select rays.

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
Downstream particle trajectories are projected onto the selection plane. A selected
bixel with no particles is rejected; increase histories or check coverage.

Only selected portable TOPAS replay pairs are bundled; identical selections are shared.
`PhaseSpaceMultipleUse = 1` is required by the current normalization assumptions.
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

The packaged reference has MLC center 350 mm, jaws 470 mm and slit center 707 mm
from the source. Inspect the actual selected TOML before changing hardware. MLC is
a simplified two-bank model, not individual leaves. No patient clearance is assumed.
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
