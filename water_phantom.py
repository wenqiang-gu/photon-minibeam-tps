"""Native pyRadPlan water phantom and portable TOPAS smoke test."""
import argparse
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.io import savemat
from pyRadPlan import PhotonPlan, generate_stf, calc_dose_influence, calc_dose_forward
from pyRadPlan.machines import PhotonLINAC
from minibeam import TOPASPhotonEngine
from minibeam.geometry.coordinates import scoring_grid
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet, create_voi

TARGET = "TARGET"
GANTRY_ANGLES = [0., 90.]
COUCH_ANGLES = [0., 15.]  # zero for each gantry angle
SAD_MM = 1000.0
BIXEL_WIDTH_MM = 5.0
ISO_CENTER_LPS_MM = None  # selected target mask centroid
DOSE_SPACING_MM = None  # preserve the native CT scoring grid
WATER = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "collect", "forward"])
    parser.add_argument("--bundle", default="runs/water-cluster")
    parser.add_argument("--histories", type=int, default=10000)
    args = parser.parse_args()
    if args.stage in {"collect", "forward"} and not (Path(args.bundle) / "manifest.json").is_file():
        raise SystemExit("Prepare a bundle and run its TOPAS jobs before collecting dose")
    image = sitk.GetImageFromArray(np.zeros((12, 20, 20), dtype=np.float32))
    image.SetSpacing((3., 3., 3.))
    image.SetOrigin((-28.5, -28.5, -16.5))
    ct = CT(cube_hu=image)
    body = np.ones((12, 20, 20), dtype=np.uint8)
    target = np.zeros_like(body)
    target[5:8, 8:12, 7:10] = 1  # asymmetric, off-center target
    cst = StructureSet(ct_image=ct, vois=[
        create_voi(name="BODY", grid=ct.grid, mask=body, voi_type="EXTERNAL"),
        create_voi(name="TARGET", grid=ct.grid, mask=target, voi_type="TARGET")])
    metadata = {"synthetic": "water"}
    matches = [v for v in cst.vois if v.name == TARGET]
    if len(matches) != 1 or not np.any(sitk.GetArrayViewFromImage(matches[0].mask)):
        raise ValueError("Selected target must exist uniquely and be nonempty")
    # The native importer may classify several ROIs as TARGET. Only this one
    # should drive target-covering steering and the default centroid.
    for index, voi in enumerate(cst.vois):
        kind = "TARGET" if voi.name == TARGET else ("OAR" if voi.voi_type == "TARGET" else voi.voi_type)
        if kind != voi.voi_type:
            data = voi.model_dump(exclude_computed_fields=True)
            data.update(voi_type=kind, overlap_priority=0 if kind == "TARGET" else 5)
            cst.vois[index] = create_voi(data)
    gantry = np.asarray(GANTRY_ANGLES, dtype=float)
    couch = np.zeros_like(gantry) if COUCH_ANGLES is None else np.asarray(COUCH_ANGLES, dtype=float)
    if gantry.ndim != 1 or not gantry.size or couch.shape != gantry.shape or not np.isfinite([gantry, couch]).all():
        raise ValueError("Provide matching, nonempty finite gantry/couch angle lists")
    if not all(np.isfinite(x) and x > 0 for x in (SAD_MM, BIXEL_WIDTH_MM)):
        raise ValueError("SAD and bixel width must be positive and finite")
    # Native steering defaults to 6.0; this label does not set TOPAS photon energies.
    machine = PhotonLINAC(version=2, name="IdealSpectrumPhoton", energies=np.array([6.0]),
                          sad=SAD_MM, scd=SAD_MM / 2)
    machine_data = machine.model_dump(exclude_none=True, exclude_computed_fields=True)
    machine_data["meta"] = {"radiation_mode": "photons"}
    plan = PhotonPlan(machine=machine_data, num_of_fractions=1)
    plan.prop_stf = {"generator": "photonSingleBixel",
                     "gantry_angles": gantry.tolist(),
                     "couch_angles": couch.tolist(), "bixel_width": BIXEL_WIDTH_MM, "add_margin": False}
    if ISO_CENTER_LPS_MM is not None:
        iso = np.asarray(ISO_CENTER_LPS_MM, dtype=float)
        if iso.shape != (3,) or not np.isfinite(iso).all():
            raise ValueError("Isocenter must be three finite LPS coordinates in mm")
        plan.prop_stf["iso_center"] = iso.reshape(1, 3)
    # Without an override generate_stf uses native cst.target_center_of_mass().
    plan.prop_dose_calc = {"engine": "TOPASPhoton", "bundle_dir": args.bundle,
                          "histories": args.histories, "water": WATER}
    if DOSE_SPACING_MM is not None:
        plan.prop_dose_calc["dose_spacing_mm"] = DOSE_SPACING_MM
    grid = scoring_grid(ct, DOSE_SPACING_MM)
    if ct.size[0] != ct.size[1] or grid.dimensions[0] != grid.dimensions[1]:
        raise ValueError("matRad export requires square transverse CT and dose grids with pyRadPlan 0.5.0")
    stf = generate_stf(ct, cst, plan)
    engine = TOPASPhotonEngine(plan)
    root = engine.prepare_jobs(ct, cst, stf, provenance=metadata)
    derived = root / "derived"
    derived.mkdir(exist_ok=True)
    # Native serializers own all voxel permutations and structure indices.
    exported = {"ct": ct.to_matrad(), "cst": cst.to_matrad(),
                "pln": plan.to_matrad(), "stf": stf.to_matrad()}
    metadata.update(units="Gy/primary photon", weight_units="primary photons",
                    normalization="TOPAS Sum / histories with scorer active",
                    lps_to_topas_translation_mm=json.loads((root / "manifest.json").read_text())["lps_to_topas_translation_mm"])
    if args.stage == "collect":
        dij = calc_dose_influence(ct, cst, stf, plan)
        exported["dij"] = dij.to_matrad()
        # Upstream 0.5.0 leaves these three labels zero based.
        for key in ("beamNum", "rayNum", "bixelNum"):
            exported["dij"][key] = exported["dij"][key] + 1
    elif args.stage == "forward":
        weights = np.ones(stf.total_number_of_bixels)  # primary photons, not MU
        result = calc_dose_forward(ct, cst, stf, plan, weights=weights)
        sitk.WriteImage(result["physical_dose"], str(derived / "dose.mha"))
        metadata["weights_primary_photons"] = weights.tolist()
    savemat(derived / ("result.mat" if args.stage == "collect" else "steering.mat"), exported)
    (derived / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Bundle: {root}; artifacts: {derived}")


if __name__ == "__main__":
    main()
