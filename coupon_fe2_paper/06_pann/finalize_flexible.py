"""Lock candidates by validation score, then evaluate their untouched test/probe.

All flex_* run directories must be complete. Existing checkpoints are never
overwritten; the output is a selection manifest and a separate audit report.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

from audit_flexible import main as audit_main

HERE=Path(__file__).resolve().parent


def main():
    folder=HERE/'enrichment_results'
    candidates={'icnn':[],'ickan':[]}
    for run in sorted(folder.glob('flex_*')):
        if not run.is_dir():continue
        if not (run/'results.json').exists() or not (run/'model.pt').exists():
            raise RuntimeError(f'Run not complete: {run}')
        result=json.loads((run/'results.json').read_text())
        config=json.loads((run/'manifest.json').read_text())['configuration']
        candidates[config['core']].append(dict(run=run.name,checkpoint=str(run/'model.pt'),
                  validation_stress=result['metrics']['validation']['stress'],
                  validation_energy=result['metrics']['validation']['energy'],
                  reference_tangent_relative_error=result['reference_tangent_relative_error']))
    chosen={core:min(rows,key=lambda row:row['validation_stress']) for core,rows in candidates.items()}
    for row in chosen.values():row['checkpoint_sha256']=hashlib.sha256(Path(row['checkpoint']).read_bytes()).hexdigest()
    selection=dict(criterion='minimum validation stress relative L2; no test/probe metric in selection',
                   candidates=candidates,selected=chosen)
    path=folder/'selected_models.json'
    path.write_text(json.dumps(selection,indent=2,allow_nan=False)+'\n')
    old_argv=sys.argv
    try:
        sys.argv=['audit_flexible.py',*[row['checkpoint'] for row in chosen.values()],
                  '--evaluate','--output',str(folder/'selected_models_audit.json')]
        audit_main()
    finally:sys.argv=old_argv
    audit=json.loads((folder/'selected_models_audit.json').read_text())
    for core,row in chosen.items():row['audit']=audit[row['checkpoint']]
    selection['saved_baselines']={}
    for core in ('icnn','ickan','free','regression'):
        ck=torch.load(HERE/f'pann_{core}.pt',map_location='cpu',weights_only=False)
        selection['saved_baselines'][core]=dict(test_stress=ck['err_test'],probe_stress=ck['err_probe'])
    path.write_text(json.dumps(selection,indent=2,allow_nan=False)+'\n')
    print('Selected checkpoint manifest:',path)


if __name__=='__main__':main()
