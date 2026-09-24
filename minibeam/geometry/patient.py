"""Validate and inventory original CT files for TOPAS; never rewrite DICOM."""
from pathlib import Path
import numpy as np
import pydicom
from pydicom.uid import CTImageStorage
import SimpleITK as sitk

from ..topas.manifest import sha256


def dicom_inputs(directory, ct):
    """Return original slice paths and provenance for one matching axial CT series.

    SimpleITK (the native pyRadPlan reader) verifies pixel values. Header checks
    prevent mixing series or passing nonuniform/oblique stacks to TOPAS.
    """
    if not directory or not Path(directory).is_dir():
        raise ValueError("Set dicom_dir to the original CT directory for TsDicomPatient")
    entries = []
    for path in sorted(Path(directory).rglob('*')):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            continue
        if getattr(ds, 'Modality', '') == 'CT':
            if str(ds.SOPClassUID) != str(CTImageStorage) or int(getattr(ds, 'NumberOfFrames', 1)) != 1:
                raise ValueError("Only classic single-frame CT DICOM is supported")
            entries.append((path, ds))
    if not entries:
        raise ValueError("No CT DICOM slices found")
    if len({str(ds.SeriesInstanceUID) for _, ds in entries}) != 1:
        raise ValueError("dicom_dir must contain exactly one CT series; isolate the planning CT series")
    if len({str(ds.SOPInstanceUID) for _, ds in entries}) != len(entries):
        raise ValueError("Duplicate CT SOPInstanceUID")
    if len({str(ds.FrameOfReferenceUID) for _, ds in entries}) != 1:
        raise ValueError("CT slices have mismatched frames of reference")
    if len(entries) != ct.size[2]:
        raise ValueError("DICOM slice count differs from planning CT")
    entries.sort(key=lambda item: float(item[1].ImagePositionPatient[2]))
    spacing = ct.grid.resolution_vector
    for z, (_, ds) in enumerate(entries):
        if int(ds.BitsAllocated) != 16:
            raise ValueError("OpenTOPAS 4.2.p3 CT input requires 16-bit pixel storage")
        if not np.allclose(ds.ImageOrientationPatient, [1,0,0,0,1,0], atol=1e-6, rtol=0):
            raise ValueError("Only axial identity LPS DICOM orientation is supported")
        if (int(ds.Columns), int(ds.Rows)) != tuple(ct.size[:2]):
            raise ValueError("DICOM dimensions differ from planning CT")
        if not np.allclose(np.asarray(ds.PixelSpacing, float)[::-1], spacing[:2], atol=1e-5, rtol=0):
            raise ValueError("DICOM spacing differs from planning CT")
        expected = np.asarray(ct.origin) + [0, 0, z * spacing[2]]
        if not np.allclose(ds.ImagePositionPatient, expected, atol=1e-4, rtol=0):
            raise ValueError("DICOM slice positions differ from planning CT or spacing is nonuniform")
        if len(entries) == 1 and not np.isclose(float(ds.SliceThickness), spacing[2]):
            raise ValueError("DICOM slice thickness differs from planning CT")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path, _ in entries])
    image = reader.Execute()
    if not np.array_equal(sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkFloat32)),
                          sitk.GetArrayFromImage(ct.cube_hu)):
        raise ValueError("DICOM HU values differ from planning CT; use the unmodified native import")
    files = {f'inputs/dicom/ct_{index:06d}.dcm': path
             for index, (path, _) in enumerate(entries, 1)}
    info = {'series_instance_uid': str(entries[0][1].SeriesInstanceUID),
            'frame_of_reference_uid': str(entries[0][1].FrameOfReferenceUID),
            'sop_instance_uids': [str(ds.SOPInstanceUID) for _, ds in entries],
            'files': {name: sha256(path) for name, path in files.items()}}
    return files, info


def patient_parameters(ct, *, water, num_threads, world_half, enable_opengl=False):
    extent = np.asarray(ct.size) * ct.grid.resolution_vector
    lines = ['includeFile = inputs/materials.txt', 's:Ge/World/Material = "G4_AIR"',
             's:Ge/Patient/Parent = "World"', 's:Ge/Patient/Material = "G4_WATER"',
             f'i:Ts/NumberOfThreads = {num_threads}', 'b:Ts/ShowCPUTime = "True"',
             'i:Ts/ShowHistoryCountAtInterval = 100000',
             f'b:Ts/PauseBeforeQuit = "{enable_opengl}"', f'b:Ts/UseQt = "{enable_opengl}"',
             f'b:Gr/Enable = "{enable_opengl}"',
             'sv:Ph/Default/Modules = 1 "g4em-standard_opt4"',
             'd:Ph/Default/CutForAllParticles = 0.1 mm']
    if enable_opengl:
        lines += ['s:Gr/PatientView/Type = "OpenGL"',
                  'i:Gr/PatientView/WindowSizeX = 1000',
                  'i:Gr/PatientView/WindowSizeY = 800',
                  'b:Gr/PatientView/IncludeGeometry = "True"',
                  'b:Gr/PatientView/IncludeAxes = "True"',
                  's:Gr/PatientView/AxesComponent = "Patient"',
                  'd:Gr/PatientView/AxesSize = 100 mm',
                  'b:Gr/PatientView/IncludeTrajectories = "True"']
        if not water:
            # A material-free parallel world displays the full CT envelope while
            # TsDicomPatient hides its own envelope to show the selected slice.
            # Never add this box to the physical patient/world geometry.
            lines += ['s:Ge/CTOutline/Type = "TsBox"',
                      's:Ge/CTOutline/Parent = "World"',
                      'b:Ge/CTOutline/IsParallel = "True"',
                      's:Ge/CTOutline/ParallelWorldName = "CTOutlineWorld"',
                      's:Ge/CTOutline/Color = "Aqua"',
                      's:Ge/CTOutline/DrawingStyle = "Wireframe"']
            for axis, length in zip('XYZ', extent):
                lines += [f'd:Ge/CTOutline/HL{axis} = {length/2:.12g} mm',
                          f'd:Ge/CTOutline/Trans{axis} = 0 mm',
                          f'd:Ge/CTOutline/Rot{axis} = 0 deg']
    if water:
        lines.append('s:Ge/Patient/Type = "TsBox"')
    else:
        lines += ['s:Ge/Patient/Type = "TsDicomPatient"',
                  's:Ge/Patient/DicomDirectory = "inputs/dicom"',
                  'sv:Ge/Patient/DicomModalityTags = 1 "CT"']
    for axis, n, spacing, length in zip("XYZ", ct.size, ct.grid.resolution_vector, extent):
        lines += [f'd:Ge/World/HL{axis} = {world_half:.12g} mm',
                  f'd:Ge/Patient/Trans{axis} = 0 mm', f'd:Ge/Patient/Rot{axis} = 0 deg']
        if water:
            lines += [f'd:Ge/Patient/HL{axis} = {length/2:.12g} mm',
                      f'i:Ge/Patient/{axis}Bins = {n}']
    return "\n".join(lines) + "\n"



def axial_slice_for_isocenter(ct, iso_center_lps_mm):
    """Nearest native CT slice (TOPAS uses 1-based original slice numbers)."""
    z = float(iso_center_lps_mm[2])
    if not np.isfinite(z):
        raise ValueError("Isocenter Z must be finite for slice visualization")
    spacing = float(ct.grid.resolution_vector[2])
    continuous = (z - float(ct.origin[2])) / spacing
    index = int(np.clip(np.floor(continuous + 0.5), 0, ct.size[2] - 1))
    return {"axis": "Z", "slice_index_1based": index + 1,
            "slice_center_lps_mm": float(ct.origin[2]) + index * spacing,
            "isocenter_z_lps_mm": z}
