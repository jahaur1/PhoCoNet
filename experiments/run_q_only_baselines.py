from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import argparse
import re

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description="Train Q-only MLP and LSTM on three stations with one seed.")
parser.add_argument('--run-id',default='q_only_baselines_20261027')
parser.add_argument('--reference-run-id',default='phoconet_20261027')
parser.add_argument('--seed',type=int,default=2021)
args=parser.parse_args()
for name in (args.run_id,args.reference_run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name):
        parser.error('run identifiers must be safe directory names')
output=root/'result'/args.run_id
base=output
output.mkdir(parents=True,exist_ok=True)
os.environ.update(PYTHONHASHSEED=str(args.seed), PYTHONIOENCODING='utf-8',
                  PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                  CUDA_VISIBLE_DEVICES='0')
os.chdir(root)
sys.path.insert(0,str(root))
import numpy as np
import torch
from experiments.information_value_core import AUXILIARIES, load_frame
from ts_benchmark.baselines.synchronous_soft_sensor.synchronous_soft_sensor import SynchronousMLP
from ts_benchmark.baselines.synchronous_soft_sensor.q_only_lstm import QOnlyLSTM

def save(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

output.mkdir(parents=True,exist_ok=True)
definitions={'MLP_Q':'synchronous_soft_sensor.SynchronousMLP',
             'LSTM_Q':'synchronous_soft_sensor.q_only_lstm.QOnlyLSTM'}
params=dict(seq_len=96,horizon=24,hidden_dim=64,dropout=.1,num_epochs=100,
            patience=10,batch_size=64,lr=.001,lradj='type1',loss='MSE')
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
          'models':definitions,'seed':args.seed,'params':params,
          'input':'Q only: 24 positions x 8 auxiliary variables; no T, H, calendar or position inputs',
          'MLP':'Pointwise 8->64->64->1, GELU and dropout 0.1',
          'LSTM':'Unidirectional single-layer LSTM 8->64, output dropout 0.1 and linear 64->1; hidden state reset for each window',
          'split':'chronological tv_end=int(2286*0.8)=1828, train_end=int(1828*0.875)=1599; train 1599, validation 229, test 458 timestamps; 435 test windows',
          'normalization':'training-only target and auxiliary z-score, no RevIN',
          'reference':str(root/'result'/args.reference_run_id/'phoconet_main'),
          'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
          'data':{st:sha(root/'dataset/forecasting'/f'{st}.csv') for st in ('juzizhou','sanjiaozhou','laodaohe')}}
manifest['source_sha256']={str(p.relative_to(root)):sha(p) for p in root.joinpath('ts_benchmark').rglob('*.py') if '__pycache__' not in p.parts}
if (output/'manifest.json').exists():
    previous=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    for key in ('models','seed','params','data','source_sha256','reference'):
        if previous[key]!=manifest[key]:
            raise RuntimeError('Run inputs changed; choose a new run-id')
else:
    save(output/'manifest.json',manifest)
for station in ('juzizhou','sanjiaozhou','laodaohe'):
    path=root/'result'/args.reference_run_id/'phoconet_main'/station/'h24'/f'seed_{args.seed}'/'predictions.npz'
    if not path.exists():
        raise FileNotFoundError('Run the matching PhoCoNet main experiment first: '+str(path))
with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for folder in ('ts_benchmark','config','experiments'):
        for p in (root/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                archive.write(p,str(p.relative_to(root)))
    archive.write(Path(__file__),'comparison_run.py')

audit=[]
for cls in (SynchronousMLP,QOnlyLSTM):
    wrapper=cls(**params)
    wrapper.config.input_dim=9
    wrapper.config.output_dim=1
    wrapper.model=wrapper._init_model().eval()
    history=torch.randn(2,96,9)
    query=torch.randn(2,24,8,requires_grad=True)
    marks=torch.randn(2,96,4)
    a=wrapper._process(history,None,marks,None,query)['output']
    b=wrapper._process(torch.randn_like(history)*100,None,torch.randn_like(marks),None,query)['output']
    assert torch.equal(a,b)
    assert a.shape==(2,24,1) and torch.isfinite(a).all()
    a.sum().backward()
    assert query.grad is not None and query.grad.abs().sum()>0
    audit.append({'model':cls.__name__,'history_and_mark_invariance':True,
                  'query_gradient_nonzero':True,'parameters':sum(p.numel() for p in wrapper.model.parameters())})
save(output/'input_audit.json',audit)
rows=[]
status={}
for name, model in definitions.items():
    for station in ('juzizhou','sanjiaozhou','laodaohe'):
        directory=output/name/station/f'seed_{args.seed}'
        directory.mkdir(parents=True,exist_ok=True)
        actual_params={**params,'robustness_checkpoint_cache':str(directory/'checkpoint.pt')}
        command=[sys.executable,'-B','-u','scripts/run_benchmark.py',
                 '--config-path','rolling_forecast_config.json',
                 '--data-name-list',station+'.csv',
                 '--strategy-args',json.dumps(dict(horizon=24,target_channel=[7])),
                 '--model-name',model,'--model-hyper-params',json.dumps(actual_params),
                 '--seed',str(args.seed),'--deterministic','full','--save-true-pred','true',
                 '--gpus','0','--num-workers','1','--timeout','60000',
                 '--save-path',str(directory.relative_to(root/'result'))]
        save(directory/'request.json',{'model':model,'params':actual_params,'argv':command})
        if not (directory/'completed.json').exists():
            status[name+'/'+station]='running';save(base/'status.json',status)
            print('START '+name+'/'+station,flush=True)
            env={**os.environ,'PHOCONET_PREDICTIONS_PATH':str(directory/'predictions.npz')}
            with (directory/'training.log').open('w',encoding='utf-8') as f:
                result=subprocess.run(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:
                print((directory/'training.log').read_text(encoding='utf-8')[-6000:],flush=True)
                raise SystemExit(result.returncode)
        report=sorted(directory.glob('test_report*.csv'))[-1]
        with report.open(encoding='utf-8-sig',newline='') as f:
            metrics={row[1]:float(row[2]) for row in list(csv.reader(f))[1:] if len(row)>=3 and row[1] in ('mse_norm','mae_norm','rmse_norm')}
        reference=root/'result'/args.reference_run_id/'phoconet_main'/station/'h24'/f'seed_{args.seed}'/'predictions.npz'
        data=np.load(directory/'predictions.npz'); ref=np.load(reference)
        np.testing.assert_array_equal(data['origins'],ref['origins'])
        np.testing.assert_array_equal(data['actual'],ref['actual'])
        assert data['predicted'].shape==(435,24,1) and np.isfinite(data['predicted']).all()
        frame=load_frame(root/'dataset/forecasting'/f'{station}.csv','full8')
        train_end=int(int(len(frame)*.8)*.875)
        scale=frame.TP.iloc[:train_end].std(ddof=0)
        err=(data['predicted']-data['actual'])/scale
        recomputed={'mae_norm':float(np.abs(err).mean()),'mse_norm':float((err**2).mean()),
                    'rmse_norm':float(np.sqrt((err**2).mean(axis=(1,2))).mean())}
        for key in recomputed:
            np.testing.assert_allclose(recomputed[key],metrics[key],rtol=1e-5,atol=1e-7)
        row=dict(model=name,station=station,seed=args.seed,**metrics,
                 checkpoint_sha256=sha(directory/'checkpoint.pt'),prediction_sha256=sha(directory/'predictions.npz'))
        save(directory/'completed.json',row)
        rows.append(row)
        status[name+'/'+station]='complete';save(base/'status.json',status)
        print('DONE '+name+'/'+station+' '+str(metrics),flush=True)
save(output/'metrics.json',rows)
with (output/'metrics_by_run.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('ALL SIX RUNS COMPLETE',flush=True)
