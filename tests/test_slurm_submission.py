"""Slurm contracts tested without a scheduler or transport executable."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'projects'
spec = importlib.util.spec_from_file_location('submit_topas', SCRIPTS/'submit_topas.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def bundle(root, threads=4):
    (root/'jobs').mkdir(parents=True)
    (root/'inputs').mkdir()
    (root/'inputs/common.txt').write_text('' if threads is None else f'i:Ts/NumberOfThreads = {threads}\n')
    (root/'jobs/bixel_000001.txt').write_text('includeFile = inputs/common.txt\n')
    (root/'manifest.json').write_text(json.dumps(dict(bundle_id='fixture',jobs=[dict(
        job_id='bixel_000001',parameter_file='jobs/bixel_000001.txt')])))
    return root


@pytest.fixture
def tools(tmp_path,monkeypatch):
    binpath=tmp_path/'bin';binpath.mkdir()
    monkeypatch.setenv('PATH',str(binpath)+os.pathsep+os.environ['PATH'])
    monkeypatch.setattr(submit,'scheduler_status',lambda _: 'status unavailable')
    env=tmp_path/'env with spaces.sh';env.write_text('export MOCK_TOPAS_ENV=loaded\n')
    sbatch=binpath/'sbatch';sbatch.write_text('#!/bin/sh\nprintf "12345;mockcluster\\n"\n');sbatch.chmod(0o755)
    return binpath,env


def invoke(root,env,*args):
    submit.main([str(root),'--topas-env',str(env),*args])


def test_dry_run_and_cpu_errors(tmp_path,tools,capsys):
    _,env=tools;root=bundle(tmp_path/'bundle with spaces')
    before=set(tmp_path.rglob('*'))
    invoke(root,env,'--dry-run','--mem','12G','--exclude','rohpc[9003-9005]')
    assert set(tmp_path.rglob('*'))==before
    out=capsys.readouterr().out
    assert '--cpus-per-task=4' in out and '--array=0-0%10' in out
    with pytest.raises(ValueError,match='CPU mismatch'):invoke(root,env,'--cpus-per-task','8','--dry-run')
    with pytest.raises(SystemExit):invoke(root,env,'--throttle','0')
    assert not (root/'slurm').exists()


def test_study_submission_worker_and_history(tmp_path,tools,capsys):
    bins,env=tools;root=tmp_path/'study'
    a=bundle(root/'shift_000');bundle(root/'shift_025')
    (root/'study.json').write_text(json.dumps(dict(setups=[dict(directory='shift_000'),dict(directory='shift_025')])))
    invoke(root,env,'--time','01:00:00','--mem','32000M','--partition','cpu','--job-name','test')
    history=json.loads((a/'slurm/submissions.jsonl').read_text())
    assert history['jobs']==[dict(array_index=0,job_id='bixel_000001')]
    dest=Path(history['submission_directory'])
    settings=json.loads((dest/'submission.json').read_text())['settings']
    assert '--mem=32000M' in settings['command']
    topas=bins/'topas';topas.write_text('#!/bin/sh\nprintf "%s|%s|%s\\n" "$PWD" "$MOCK_TOPAS_ENV" "$1"\nexit 7\n');topas.chmod(0o755)
    run=subprocess.run(['bash',str(dest/'worker.sh'),str(dest/'tasks.tsv'),str(env)],
        env=dict(os.environ,SLURM_ARRAY_TASK_ID='1'),text=True,capture_output=True)
    assert run.returncode==7
    assert str(root/'shift_025')+'|loaded|' in run.stdout
    assert 'exit=7 elapsed=' in run.stdout
    invoke(root,env,'--dry-run')
    assert 'Previously submitted' in capsys.readouterr().out
    assert len(list((root/'slurm').iterdir()))==1


def test_missing_thread_fallback(tmp_path,tools):
    _,env=tools;root=bundle(tmp_path/'bundle',None)
    invoke(root,env,'--cpus-per-task','3')
    history=json.loads((root/'slurm/submissions.jsonl').read_text())
    dest=Path(history['submission_directory'])
    assert 'NumberOfThreads = 3' in (dest/'input_0.txt').read_text()
    assert (root/'inputs/common.txt').read_text()==''


def test_default_fallback_and_mixed_threads(tmp_path,tools,capsys):
    _,env=tools;root=tmp_path/'study'
    bundle(root/'a',None);bundle(root/'b',4)
    (root/'study.json').write_text(json.dumps(dict(setups=[dict(directory='a'),dict(directory='b')])))
    with pytest.raises(ValueError,match='Mixed'):invoke(root,env,'--dry-run')
    invoke(root,env,'--default-cpus','4','--dry-run')
    assert '4 CPUs/task' in capsys.readouterr().out


def test_include_validation(tmp_path):
    root=bundle(tmp_path/'bundle')
    job=root/'jobs/bixel_000001.txt';common=root/'inputs/common.txt'
    job.write_text('includeFile = inputs/common.txt\ni:Ts/NumberOfThreads = 2\n')
    assert submit.thread_setting(job,root)==2
    common.write_text('includeFile = jobs/bixel_000001.txt\n')
    with pytest.raises(ValueError,match='cycle'):submit.thread_setting(job,root)
    common.write_text('i:Ts/NumberOfThreads = -1\n')
    with pytest.raises(ValueError,match='Unsupported'):submit.thread_setting(job,root)
    common.unlink()
    with pytest.raises(ValueError,match='Missing'):submit.thread_setting(job,root)


def test_failure_and_recording_failure(tmp_path,tools,monkeypatch):
    bins,env=tools;root=bundle(tmp_path/'bundle')
    (bins/'sbatch').write_text('#!/bin/sh\necho unavailable >&2\nexit 1\n')
    with pytest.raises(ValueError,match='sbatch failed'):invoke(root,env)
    assert not (root/'slurm/submissions.jsonl').exists()
    (bins/'sbatch').write_text('#!/bin/sh\necho 12345\n')
    original=Path.open
    def fail_record(self,*args,**kwargs):
        if self.name=='submissions.jsonl':raise OSError('simulated disk failure')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',fail_record)
    with pytest.raises(ValueError,match='JOB 12345 WAS SUBMITTED'):invoke(root,env)
    assert list((root/'slurm').glob('*/receipt.json'))


def test_shell_help_and_git_ignore():
    run=subprocess.run(['bash',str(SCRIPTS/'submit_topas.sh'),'--help'],capture_output=True,text=True)
    assert run.returncode==0 and '--topas-env' in run.stdout
    for path,ignored in [('projects/submit_topas.sh',False),('projects/submit_topas.py',False),
                         ('projects/example/jobs/job.txt',True),('projects/example.json',True)]:
        r=subprocess.run(['git','check-ignore','--no-index','-q',path],cwd=SCRIPTS.parent)
        assert (r.returncode==0)==ignored


def test_incomplete_study_and_ambiguous_includes(tmp_path):
    root=tmp_path/'study';a=bundle(root/'a')
    (root/'study.json').write_text(json.dumps(dict(setups=[dict(directory='a'),dict(directory='missing')])))
    with pytest.raises(FileNotFoundError):submit.discover(root)
    second=a/'inputs/second.txt';second.write_text('i:Ts/NumberOfThreads = 4\n')
    job=a/'jobs/bixel_000001.txt'
    job.write_text('includeFile = inputs/common.txt inputs/second.txt\n')
    with pytest.raises(ValueError,match='Ambiguous'):submit.thread_setting(job,a)
    # A root literal is the explicit override of two independent include branches.
    job.write_text(job.read_text()+'i:Ts/NumberOfThreads = 2\n')
    assert submit.thread_setting(job,a)==2


def test_status_fallback(tmp_path,monkeypatch):
    bins=tmp_path/'bin';bins.mkdir();monkeypatch.setenv('PATH',str(bins))
    assert submit.scheduler_status({'slurm_job_id':'123'})=='status unavailable'
    sacct=bins/'sacct';sacct.write_text('#!/bin/sh\necho "123_0|COMPLETED"\n');sacct.chmod(0o755)
    assert 'COMPLETED' in submit.scheduler_status({'slurm_job_id':'123'})
