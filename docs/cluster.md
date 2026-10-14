# Portable TOPAS execution and Slurm

[Main guide](../README.md)

## Transfer and execute

Prepare locally with the patient or water script. Transfer the **complete project**,
including `inputs`, `jobs`, `manifest.json` and `derived/steering.mat` plus its hash
receipt. For studies, transfer the whole root with `study.json` and setup folders.
Input paths are relative to each setup's root.

Example transfer from macOS, avoiding Apple extended attributes:

```sh
COPYFILE_DISABLE=1 tar --no-xattrs -czf patient.tar.gz -C projects my-patient
scp patient.tar.gz user@cluster:/path/to/work/
# On the cluster:
cd /path/to/work
tar -xzf patient.tar.gz
```

Use your installation's OpenTOPAS environment script, then run from the bundle root:

```sh
source /path/to/opentopas-env.sh
cd /path/to/work/my-patient
topas jobs/bixel_000001.txt
```

The name is illustrative: combined jobs use `beam_000001.txt`; inspect `manifest.json`
for exact selected jobs. For a study, enter the corresponding setup folder first.
TOPAS execution requires OpenTOPAS 4.2.p3 and its Geant4 data/environment, not Python
or pyRadPlan. Configuration preparation/collection requires the full Python package.

Return the `results/` files to the corresponding unchanged local project. Collect
only after all relevant jobs finish. Preserve input files, source histories and scorer
names: returned CSVs must match the saved manifest. Do not edit threads, geometry or
histories in an already prepared project and expect its old fingerprint to match.

## Optional Slurm submission

Copy the root helper files under `projects/` along with your projects:
`submit_topas.sh`, `submit_topas.py`, `topas_worker.sh`, `add_batches.sh`, and
`add_batches.py`. The submission machine
needs Python 3.8+ standard library; the worker needs Bash and TOPAS. Python 3.12 also
works. A separate pyRadPlan installation is unnecessary for submission.

```sh
bash projects/submit_topas.sh projects/my-patient --dry-run \
  --topas-env /path/to/opentopas-env.sh
bash projects/submit_topas.sh projects/my-patient/shift_000 \
  --throttle 10 --partition defq --topas-env /path/to/opentopas-env.sh
```

Use a single project, one study subdirectory, or a saved study root. Defaults are
editable at the beginning of `submit_topas.sh`; check its environment-script default
for your cluster. Command-line options override ordinary script defaults.

| Option | Meaning |
|---|---|
| `--throttle N` | Maximum simultaneous array tasks; default 10. |
| `--time LIMIT` | Wall-clock limit; default 48:00:00. |
| `--mem SIZE` | Optional per-task memory request, e.g. 12G or 32000M; default empty. |
| `--partition NAME` | Optional Slurm partition. |
| `--exclude NODES` | Optional node list, e.g. `rohpc[9003-9005]`. |
| `--job-name NAME` | Default `topas_general`. |
| `--topas-env FILE` | Environment script; CLI first, then `TOPAS_ENV`, then shell default. |
| `--cpus-per-task N` | Must match inferred TOPAS threads when present; fallback when absent. |
| `--dry-run` | Validate and print the submission; no manifest writes or `sbatch`. |
| `-h`, `--help` | Full current usage. |

CPU precedence is inferred `Ts/NumberOfThreads`, then explicit CLI fallback, then
`DEFAULT_CPUS_PER_TASK`. An explicit CLI value conflicting with inferred threads
is rejected. Mixed incompatible thread counts are rejected. The scripts do not
silently override configured TOPAS threads; fallback wrappers handle missing values.

Both root planning workflows default to `TOPAS_THREADS_PER_JOB=4`; the underlying
engine default remains 1. This is threads per job, independent of array throttle.
Match CPU requests to threads. More threads do not guarantee faster execution;
measure wall-clock time and memory on your actual nodes. The node's physical memory
and Slurm's configured/allocatable memory can differ. Omission of `--mem` follows
cluster policy and is not a guarantee of unlimited available RAM.

## Submission records and retries

The helper creates a timestamped `slurm/` submission directory containing the task
list, worker snapshot, submission metadata, logs and scheduler response. Successful
submissions append a per-bundle `slurm/submissions.jsonl` history with job identity.
The helper reports earlier submissions and queries Slurm when available; a history
record does not prove completion and does not automatically prevent resubmission.

The worker selects its array row, loads the TOPAS environment, changes directory to
that row's project root and invokes its TOPAS input. The scheduler controls allocation,
concurrency, logs and retry policy. Avoid resubmitting an entire study unintentionally.
For a failed direct job, move aside its partial CSV first, then rerun its exact input.
Do not treat missing, duplicate or truncated results as zero dose.

Useful scheduler checks:

```sh
squeue -u "$USER"
sacct -u "$USER" -S today --format=JobID,JobIDRaw,State,AllocCPUS,Elapsed,MaxRSS
# Use a running task's numeric JobIDRaw for older Slurm versions:
sstat -j 123456.batch --format=JobID,AveCPU,AveRSS,MaxRSS
```

Slurm accounting availability depends on cluster configuration. TOPAS `Real`/shell
elapsed time measures wall time; user and system CPU totals are not added to wall time.

## Repeated transport batches

Set `NUMBER_OF_BATCHES` (initial total) and `RANDOM_SEED` in either workflow before
preparation. To add batches later, use saved inputs only:

```sh
# From the repository root; adds three batches, not a total of three.
./projects/add_batches.sh projects/my-patient --count 3 --dry-run
./projects/add_batches.sh projects/my-patient --count 3
# Or, from projects/:
./add_batches.sh my-patient/shift_000 --count 2
```

`DEFAULT_ADDITIONAL_BATCHES=3` at the top of `add_batches.sh` is editable; `--count`
overrides it. A study adds the requested number to every setup. This command does
not submit jobs. Copy `add_batches.sh` and `add_batches.py` alongside the submission
scripts on the cluster; only Python 3.8+ standard library is needed. No pyRadPlan,
original DICOM or original accelerator phase-space files are opened by this helper.

Batch `000` is the original `jobs/` and `results/`; after adding batches,
`batches/batch_000/` is also a selectable reference to that original batch. Additional batches are under
`batches/batch_001/`, with a signed `batches.json` index at the setup root. All batches
share `inputs/` and the planning snapshot. **Run TOPAS from the setup root**, including
when manually running an additional batch:

```sh
cd projects/my-patient/shift_000
topas batches/batch_001/jobs/bixel_000001.txt
```

Submit a setup/study to submit all its prepared batches, or select just one:

```sh
./projects/submit_topas.sh projects/my-patient/shift_000/batches/batch_001
```

Submission prints its selected batches; history records include batch IDs. It does
not automatically skip previously submitted jobs. Each physical job has its own
seed, scorer identity and output path. `PhaseSpaceMultipleUse` stays at 1. Point
sources also reuse their saved sampled replay particles: changing the seed repeats
transport, not source sampling. Transfer the entire setup with its batch index and
batch directories; a batch folder alone is not self-contained.

Batch appending validates saved input hashes and uses an exclusive lock plus atomic
index publication. A leftover lock or an unindexed batch folder after an interrupted
process requires inspection before retrying; do not remove it while another writer
is active. Study additions are atomic per setup, not a transaction across the study.


### Replay safety checks

Copy `projects/replay_audit.py` alongside `add_batches.py` and `submit_topas.py` when
updating the cluster helpers. It uses only the Python standard library. Submission
and batch addition reject known-affected legacy replay configurations and invalid
new guarded replay. Dry runs do not write audit files. A rejection requires fresh
local preparation and a new simulation; do not edit old CSVs or rescale old dose.

New guarded jobs request only selected nonempty histories from TOPAS. Their original
history normalization remains saved separately. A lower TOPAS event count than
`HISTORIES_PER_JOB` is expected and does not reduce the represented exposure.
No OpenTOPAS rebuild is required for this workaround. The replay issue was reproduced
locally with 4.2.p3; this is not an upstream-confirmed bug advisory.
