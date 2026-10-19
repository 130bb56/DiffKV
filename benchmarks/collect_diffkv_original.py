"""Collect each original DiffKV workload exactly once with GPU0 monitoring."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.cache/baselines/fp16-env/bin/python'
NSYS = '/opt/nvidia/nsight-systems-cli/2024.4.1/target-linux-x64/nsys'
QUERY = (
    'timestamp,index,pstate,temperature.gpu,power.draw,power.limit,'
    'clocks.sm,clocks.mem,clocks_throttle_reasons.active,'
    'clocks_throttle_reasons.sw_power_cap,'
    'clocks_throttle_reasons.hw_thermal_slowdown,'
    'clocks_throttle_reasons.sw_thermal_slowdown,'
    'utilization.gpu,utilization.memory,memory.used'
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def preflight(folder, plan):
    samples = []
    deadline = time.monotonic() + 600
    while True:
        raw = subprocess.check_output([
            'nvidia-smi', '-i', '0', '--query-gpu=' + QUERY, '--format=csv',
        ], text=True)
        row = {
            key.strip(): value.strip()
            for key, value in next(csv.DictReader(raw.splitlines())).items()
        }
        apps = subprocess.check_output([
            'nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name',
            '--format=csv,noheader',
        ], text=True).strip()
        row['compute_processes'] = apps
        samples.append(row)
        write_json(folder / 'preflight.json', samples)
        if apps:
            raise RuntimeError('GPU0 is occupied: ' + apps)
        if int(row['clocks.current.sm [MHz]'].split()[0]) != 1395:
            raise RuntimeError('GPU0 SM clock must remain locked to 1395MHz')
        if float(row['power.limit [W]'].split()[0]) != plan['power_limit_W']:
            raise RuntimeError('GPU0 power limit differs from the previous measurements')
        if (
            int(row['temperature.gpu']) <= 50
            and row['clocks_throttle_reasons.sw_thermal_slowdown'] == 'Not Active'
            and row['clocks_throttle_reasons.hw_thermal_slowdown'] == 'Not Active'
        ):
            return
        if time.monotonic() > deadline:
            raise RuntimeError('GPU0 cooling criterion not reached')
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    plan = json.loads((run / 'plan.json').read_text())
    if (
        plan['diffkv_variant'] != 'original'
        or plan['repeats_per_case_backend'] != 1
        or len(plan['cases']) != 14
        or plan['physical_gpu'] != 0
    ):
        raise ValueError('Expected exactly 14 original DiffKV measurements on GPU0')

    state = {
        'status': 'running', 'total_measurements': 14,
        'captured': [], 'current': None, 'stage': 'preflight',
    }
    active = monitor = None

    def save():
        state['updated_at'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
        write_json(run / 'progress.json', state)

    def stop(signum, frame):
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
        if monitor is not None and monitor.poll() is None:
            monitor.terminate()
        state['status'] = 'interrupted'
        save()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    try:
        base = subprocess.check_output(['nvidia-smi', 'base-clocks', '-i', '0'], text=True)
        (run / 'base-clock-query.txt').write_text(base)
        if '1395MHz' not in base.replace(' ', ''):
            raise RuntimeError('Unexpected GPU0 base clock')
        for index, case in enumerate(plan['cases'], 1):
            for source, expected in plan['source_sha256'].items():
                if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
                    raise RuntimeError('Measurement source changed: ' + source)
            job = case['id'] + '-diffkv'
            folder = run / job
            state.update(current=job, stage='preflight')
            save()
            if folder.exists():
                metadata = folder / 'worker/profile.json'
                if metadata.exists() and json.loads(metadata.read_text()).get('status') == 'completed':
                    state['captured'].append(job)
                    save()
                    print('PRESENT ' + job, flush=True)
                    continue
                raise RuntimeError('Existing incomplete case will not be automatically repeated: ' + job)
            folder.mkdir()
            preflight(folder, plan)
            command = [
                NSYS, 'profile', '--trace=cuda,nvtx,osrt',
                '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                '--sample=none', '--cpuctxsw=none', '--output=' + str(folder / 'trace'),
                str(PYTHON), str(ROOT / 'benchmarks/profile_diffkv_original.py'),
                '--backend', 'diffkv', '--gpu', '0',
                '--batch', str(case['batch']),
                '--input-length', str(case['input_length']),
                '--output-length', str(case['output_length']),
                '--warmup-tokens', '16', '--output-dir', str(folder / 'worker'),
            ]
            write_json(folder / 'command.json', command)
            state['stage'] = 'capturing'
            save()
            print(f'START {index}/14 {job} variant=original n=1', flush=True)
            with (folder / 'gpu.csv').open('w') as gpu, (folder / 'capture.log').open('w') as log:
                monitor = subprocess.Popen([
                    'nvidia-smi', '-i', '0', '--query-gpu=' + QUERY,
                    '--format=csv', '--loop-ms=200',
                ], stdout=gpu, stderr=subprocess.STDOUT)
                try:
                    active = subprocess.Popen(
                        command, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)),
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    code = active.wait()
                    active = None
                finally:
                    monitor.terminate()
                    monitor.wait()
                    monitor = None
            metadata = json.loads((folder / 'worker/profile.json').read_text())
            worker = json.loads((folder / 'worker' / (job + '.json')).read_text())
            if (
                code != 0 or metadata.get('status') != 'completed'
                or worker.get('status') != 'completed'
                or worker.get('diffkv_variant') != 'original'
                or worker.get('python_variant') is not None
                or len(worker.get('samples', [])) != 1
                or not (folder / 'trace.nsys-rep').exists()
            ):
                raise RuntimeError('Single original capture did not complete: ' + job)
            state['captured'].append(job)
            save()
            print(f'CAPTURED {len(state["captured"])}/14 {job}', flush=True)
        state.update(status='completed', current=None, stage='done')
        save()
        print('ALL_CAPTURES_DONE', flush=True)
    except Exception as error:
        state.update(status='stopped', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
