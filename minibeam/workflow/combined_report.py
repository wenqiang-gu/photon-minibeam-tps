"""Aggregate-column supplement to the saved MATLAB guide."""

def combined_report(manifest, stage, image=None):
    from .collection import indexing_report
    from ..topas.manifest import phase_space_selection
    unrestricted = phase_space_selection(manifest) == 'all_forward'
    base = indexing_report(dict(manifest,beamlet_execution='separate'),stage,image)
    base = base.replace('j = 1;  % choose a global beamlet column','j = 1;  % choose an aggregate beam column')
    base = base.replace('beamletCube =', 'beamCube =')
    base = base.replace(' per beamlet', ' per beam aggregate')
    start = base.index('## Beamlet columns and TOPAS jobs')
    stop = base.index('## Structures and CT voxels')
    lines = ['## Aggregate beam columns and TOPAS jobs', '',
        ('Execution: combined; selection: all_forward. Each column is a whole-source beam dose, not a sum limited to the listed planning bixels. Members are planning references only.' if unrestricted else 'Execution: combined. Each column sums all selected bixels of one beam; it is not an individual native bixel.'),
        '`beamNum` is a 1-based native beam index. `rayNum` and `bixelNum` are 0 sentinels (−1 in native Python Dij), meaning not applicable. Never use them to index stf.',
        '`stf` remains the full native steering. `column_mapping` is a MATLAB cell array identifying each aggregate and its native members.', '',
        '| Column | Beam | Gantry / couch (deg) | Native bixels | TOPAS job |', '|---|---|---|---|---|']
    for j in manifest['jobs']:
        members=j['members']
        lines.append(f"| {j['column_index']} | {j['beam_index']} | {j['gantry_angle_deg']:g} / {j['couch_angle_deg']:g} | {members[0]['bixel_index']}–{members[-1]['bixel_index']} | {j['job_id']} |")
    lines += ['', '```matlab', 'j = 1;', 'mapping = result.column_mapping{j};',
              'b = mapping.beam_index;', 'beam = result.stf(b);', 'hardware = result.beam_geometry{b};',
              "manifest = jsondecode(fileread('../manifest.json'));", 'job = manifest.jobs(j);',
              'members = job.members;', 'disp(members);  % native bixel_index, beam_index, ray_index, beamlet_index', '```', '',
              'Weights have one entry per aggregate column. One exposure applies to the entire replayed beam with its saved fluence distribution. Independent member weights cannot be reconstructed from an aggregate dose; use separate jobs.',
              'Each shift has independent numbering starting at 1.', '']
    if manifest['normalization']=='photons_per_member_bixel':
        lines += ['Point source: each member emits N independent photons. TOPAS transports N × M histories for M members, but the scored sum is divided by N. A weight of 1 means one photon per member bixel, not one total photon for the beam. Variance scales by M² relative to TOPAS variance of its mean.', '']
    elif manifest['normalization']=='independent_primary_photon':
        lines += ['Point source: N photons total per job are sampled uniformly over the selected square union. Divide scored sum by N. A forward weight is the total number of photons for this beam, distributed across its member bixels by area.', '']
    else:
        lines += [('Phase space: all downstream records appear once, without a projected-field cut,' if unrestricted else 'Phase space: each selected record appears once in the square union,') + ' with original histories and particle weights preserved. Divide by represented original histories, not by particles or number of members.', '']
    return base[:start]+'\n'.join(lines)+'\n'+base[stop:]
