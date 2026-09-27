"""Exercise validation command routing and cleanup without a Docker daemon."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('failure', ['', 'up', 'check-env'])
def test_validate_selected_image_and_cleanup(tmp_path, failure):
    project = tmp_path / 'release'
    (project / 'devops').mkdir(parents=True)
    shutil.copy(ROOT / 'devops/validate.sh', project / 'devops/validate.sh')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    log = tmp_path / 'commands.jsonl'
    fake = '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
name = Path(sys.argv[0]).name
with open(os.environ['COMMAND_LOG'], 'a') as out:
    out.write(json.dumps([name, *args]) + '\\n')
if name == 'docker':
    failure = os.environ.get('FAIL_STEP')
    if failure and (failure in args or '--' + failure in args): sys.exit(17)
elif name == 'curl':
    if '-w' in args: print('200', end='')
    elif any('/v1/models' in a for a in args): print('{"data":[{"id":"naive-n0.5"}]}')
    else: print(json.dumps({'content':[{'type':'text','text':'hello'}], 'naivert':{'timings':{'prefill_path':'native','prompt_ms':1,'decode_tok_s_device':100}}}))
'''
    for name in ('docker', 'curl'):
        path = bindir / name
        path.write_text(f'#!{sys.executable}\n' + fake)
        path.chmod(0o755)
    (bindir / 'python3').symlink_to(sys.executable)
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
               COMMAND_LOG=str(log), FAIL_STEP=failure, NAIVERT_IMAGE='test/image:chosen',
               NAIVERT_PORT='18093')
    result = subprocess.run(['bash', str(project / 'devops/validate.sh'), '--no-build'],
                            env=env, capture_output=True, text=True)
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    run = next(c for c in commands if c[:2] == ['docker', 'run'])
    assert run[run.index('--entrypoint') + 1:][:2] == ['python3', 'test/image:chosen']
    assert not any(c[:2] == ['docker', 'build'] for c in commands)
    downs = [c for c in commands if c[-1] == 'down']
    assert len(downs) == (0 if failure == 'check-env' else 1)
    if failure:
        assert result.returncode == 17
        assert 'VALIDATE_OK' not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert 'VALIDATE_OK' in result.stdout
        assert any('http://127.0.0.1:18093/health' in c for c in commands)
        assert any(c[-3:] == ['up', '-d', '--no-build'] for c in commands)


@pytest.mark.parametrize('overrides,arguments,expected', [
    ({}, [], (1048576, 524288, 524288)),
    ({'NAIVERT_CAPACITY': '65536', 'NAIVERT_OUTPUT_CAPACITY': '32768',
      'NAIVERT_MAX_DECODE_TOKENS': '16384'}, [], (65536, 32768, 16384)),
    ({'NAIVERT_EXTRA_ARGS': '--max-decode-tokens 16384'},
     ['--max-decode-tokens', '8192'], (1048576, 524288, 8192)),
])
def test_entrypoint_capacity_defaults_and_overrides(overrides, arguments, expected):
    import shlex
    env = {key: value for key, value in os.environ.items() if not key.startswith('NAIVERT_')}
    env.update(NAIVERT_DRY_RUN='1', **overrides)
    result = subprocess.run(['bash', str(ROOT / 'devops/entrypoint.sh'), *arguments],
                            env=env, capture_output=True, text=True, check=True)
    command = shlex.split(result.stdout.splitlines()[-1])
    def last_value(flag):
        return int(command[max(i for i, value in enumerate(command) if value == flag) + 1])
    assert tuple(last_value(flag) for flag in
                 ('--capacity', '--output-capacity', '--max-decode-tokens')) == expected


@pytest.mark.parametrize('build_rc,nogpu_rc,success', [
    (0, 1, True), (17, 1, False), (0, 0, False), (0, 124, False), (0, 125, False),
])
def test_tier1_propagates_failures(tmp_path, build_rc, nogpu_rc, success):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    log = tmp_path / 'commands.jsonl'
    fake = '''import json, os, sys
with open(os.environ['COMMAND_LOG'], 'a') as out:
    out.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == 'build': sys.exit(int(os.environ['BUILD_RC']))
if sys.argv[1] == 'run' and '--name' in sys.argv: sys.exit(int(os.environ['NOGPU_RC']))
'''
    docker = bindir / 'docker'
    docker.write_text(f'#!{sys.executable}\n' + fake)
    docker.chmod(0o755)
    timeout = bindir / 'timeout'
    timeout.write_text('#!/bin/sh\nshift\nexec "$@"\n')
    timeout.chmod(0o755)
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
               COMMAND_LOG=str(log), BUILD_RC=str(build_rc), NOGPU_RC=str(nogpu_rc),
               NAIVERT_VALIDATION_DIR=str(tmp_path / 'results'))
    result = subprocess.run(['bash', str(ROOT / 'devops/tier1.sh'), 'test/image:tier1'],
                            env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == success
    assert ('TIER1_OK' in result.stdout) == success
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    if build_rc:
        assert len(commands) == 1  # Never test a stale tag after a failed build.
    else:
        assert any(command[:2] == ['rm', '-f'] for command in commands)
