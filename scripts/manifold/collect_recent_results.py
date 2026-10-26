"""Read-only H20 collection; distinguish final results from partial validation."""

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT = Path('/home/u3013198/projects/SSO_test-three-geometries-w256-momentum-20261005')
CLUSTER = Path('/home/u3013198/projects/cluster-selected-3b-20261025')
JOBS = [str(i) for i in range(4320101, 4320107)] + [str(i) for i in range(4329081, 4329087)]
TRAIN = re.compile(
    r'iteration\s+(\d+)/\s*(\d+).*?consumed samples:\s*(\d+)'
    r'.*?elapsed time per iteration \(ms\):\s*([\d.]+).*?\| lm loss:\s*([\d.Ee+-]+)'
)
VAL = re.compile(
    r'validation loss at iteration\s+(\d+)( on validation set)?'
    r'\s*\| lm loss value:\s*([\d.Ee+-]+)'
)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def parse_log(path, target, batch, seq):
    if not path.exists():
        return {'last_train': None, 'last_val': None, 'val': [], 'health': None}
    raw = path.read_bytes()
    text = raw.decode('utf-8', errors='replace')
    train, val = [], []
    for line in text.splitlines():
        m = TRAIN.search(line)
        if m:
            step, total, samples, ms, loss = m.groups()
            assert int(total) == target and int(samples) == int(step) * batch
            train.append(dict(step=int(step), target=target, samples=int(samples),
                              seconds=float(ms)/1000, loss=float(loss)))
        m = VAL.search(line)
        if m:
            step, final, loss = m.groups()
            val.append(dict(step=int(step), loss=float(loss), final=bool(final),
                            tokens=int(step)*batch*seq))
    assert all(math.isfinite(p['loss']) for p in train + val)
    assert [p['step'] for p in val] == sorted(set(p['step'] for p in val))
    health = {}
    for key in ('skipped', 'nan'):
        values = re.findall(r'number of '+key+r' iterations:\s*(\d+)', text)
        health[key] = max(map(int, values)) if values else None
    args = dict(re.findall(r'^  (\w+)\s+\.{2,}\s*(.*?)\s*$', text, re.M))
    keys = ('optimizer', 'hidden_size', 'num_layers', 'seq_length', 'seed', 'lr',
            'train_iters', 'eval_interval', 'eval_iters', 'manifold_geometry',
            'manifold_momentum_beta', 'manifold_mcsd_nesterov',
            'manifold_baseline_nesterov', 'manifold_radius_policy',
            'manifold_radius_factor', 'global_batch_size', 'micro_batch_size')
    return dict(log=str(path), log_bytes=len(raw), log_sha256=hashlib.sha256(raw).hexdigest(),
                last_train=train[-1] if train else None, last_val=val[-1] if val else None,
                val=val, health=health, arguments={k:args[k] for k in keys if k in args},
                tail_train_mean=sum(p['loss'] for p in train[-10:])/min(10,len(train)) if train else None,
                recent_seconds_per_step=sum(p['seconds'] for p in train[-30:])/min(30,len(train)) if train else None)


def collect():
    fields = ['JobID', 'JobName', 'State', 'ExitCode', 'Elapsed', 'Start', 'End', 'NodeList']
    accounting = subprocess.check_output([
        'sacct', '-n', '-P', '-X', '-j', ','.join(JOBS+['4329039']),
        '-o', 'JobID,JobName%80,State,ExitCode,Elapsed,Start,End,NodeList'
    ], text=True)
    states = {}
    for line in accounting.splitlines():
        row = dict(zip(fields, line.split('|')))
        if row.get('JobID'):
            states[row['JobID']] = row
    result = dict(checked_at_hkt=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
                  runs=[], separate_spectral_study=[], accounting=accounting)
    for job in JOBS:
        status = states[job]
        run_dir = PROJECT/'results/manifold'/status['JobName']
        config_path = run_dir/'resolved_config.json'
        cfg = read_json(config_path)
        assert cfg is not None, config_path
        assert (run_dir/'job_id.txt').read_text().strip() == job
        t = cfg['training']
        model = cfg['model']
        record = dict(job_id=job, run_dir=str(run_dir), state=status['State'],
                      exit_code=status['ExitCode'], elapsed=status['Elapsed'],
                      start=status['Start'], end=status['End'], node=status['NodeList'],
                      method=cfg['method'], lr=t['lr'], beta=t['momentum_beta'],
                      nesterov=t.get('nesterov', False), seed=t['seed'],
                      tokens=cfg['target_tokens'], width=model['hidden_size'],
                      final_step=cfg['train_iters'], source_commit=cfg['source_commit'],
                      config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                      completion=read_json(run_dir/'completion.json'),
                      config={k:v for k,v in cfg.items() if k != 'source_hashes'},
                      optimizer_source_hashes={k:v for k,v in cfg.get('source_hashes',{}).items()
                                               if '/orthogonalized_optimizers/manifold_' in k})
        record.update(parse_log(PROJECT/'logs'/f"{status['JobName']}_{job}.out",
                                cfg['train_iters'], t['global_batch'], model['seq_length']))
        record['final_val'] = None
        if status['State'] == 'COMPLETED':
            assert status['ExitCode'] == '0:0'
            assert record['completion']['exit_code'] == 0
            assert record['last_val']['step'] == cfg['train_iters'] and record['last_val']['final']
            record['final_val'] = record['last_val']['loss']
        result['runs'].append(record)
    for i in range(6):
        job = f'4329039_{i}'
        status = states[job]
        run_dir = CLUSTER/'results'/job
        cfg = read_json(run_dir/'launch_config.json')
        record = dict(job_id=job, state=status['State'], exit_code=status['ExitCode'],
                      elapsed=status['Elapsed'], start=status['Start'], end=status['End'],
                      config=cfg, training_completed=(run_dir/'TRAINING_COMPLETED').exists(),
                      all_checks_completed=(run_dir/'COMPLETED').exists())
        assert cfg is not None
        record.update(parse_log(run_dir/'train.log', int(cfg['TRAIN_ITER']),
                                int(cfg['GLOBAL_BATCH']), int(cfg['SEQ_LENGTH'])))
        record['final_val'] = None
        if status['State'] == 'COMPLETED':
            assert status['ExitCode'] == '0:0' and record['all_checks_completed']
            assert record['last_val']['final'] and record['last_val']['step']==int(cfg['TRAIN_ITER'])
            record['final_val'] = record['last_val']['loss']
        result['separate_spectral_study'].append(record)
    result['account_queue'] = subprocess.check_output(
        ['squeue','-h','-u','u3013198','-o','%i|%j|%T|%M|%R'], text=True)
    result['disk'] = subprocess.check_output(['df','-h',str(PROJECT)], text=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remote', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--host', default='hpc2021')
    args = p.parse_args()
    if args.remote:
        print(json.dumps(collect(), ensure_ascii=False, separators=(',',':')))
        return
    if args.output is None:
        p.error('--output is required')
    command = shlex.join(['/home/u3013198/envs/sso_h20/bin/python','-B','-','--remote'])
    proc = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',args.host,command],
                          input=Path(__file__).read_text(), text=True, capture_output=True, timeout=180)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    result = json.loads(proc.stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(result['checked_at_hkt'])
    for r in result['runs'] + result['separate_spectral_study']:
        print(r['job_id'],r['state'],r['final_val'],r['last_val'],r['elapsed'])
    print('Saved',args.output)


if __name__ == '__main__':
    main()
