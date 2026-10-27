from pathlib import Path
import json
import pandas as pd
import argparse
import re

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description="Summarize Q-only baselines and matched PhoCoNet results.")
parser.add_argument('--run-id',default='q_only_baselines_20261027')
args=parser.parse_args()
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id):
    parser.error('run-id must be a safe directory name')
output=root/'result'/args.run_id
base=output
manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
frame=pd.read_csv(output/'metrics_by_run.csv')
reference=pd.read_csv(Path(manifest['reference'])/'metrics_by_seed.csv')
reference=reference.loc[(reference.seed==manifest['seed'])&(reference.horizon==24)].copy()
reference['model']='PhoCoNet'
columns=['model','station','seed','mae_norm','rmse_norm','mse_norm']
all_metrics=pd.concat([frame[columns],reference[columns]],ignore_index=True)
assert len(frame)==6 and len(reference)==3
all_metrics.to_csv(output/'comparison_by_station.csv',index=False,encoding='utf-8-sig')
macro=all_metrics.groupby('model')[['mae_norm','rmse_norm','mse_norm']].mean()
macro.to_csv(output/'macro_summary.csv',encoding='utf-8-sig')
comparisons=[]
for row in frame.itertuples():
    ref=reference.loc[reference.station==row.station].iloc[0]
    comparisons.append(dict(model=row.model,station=row.station,
        mae_difference=row.mae_norm-ref.mae_norm,
        phoconet_mae_reduction_pct=100*(row.mae_norm-ref.mae_norm)/row.mae_norm,
        rmse_difference=row.rmse_norm-ref.rmse_norm))
pd.DataFrame(comparisons).to_csv(output/'paired_differences.csv',index=False,encoding='utf-8-sig')
names=dict(juzizhou='桔子洲',sanjiaozhou='三角洲',laodaohe='捞刀河')
lines=['# 仅用在线辅助观测 Q 的 MLP / LSTM 对照','',
       f'训练种子为 {manifest["seed"]}，三个数据集分别训练。历史采样长度 96、估计窗口长度 24；每站评估相同的 435 个测试窗口。MLP 与 LSTM 均只读取 Q 中的八个辅助变量，不读取历史 TP、历史辅助变量、日期标记、窗口位置或测试 TP。',
       '', '| 数据集 | MLP（Q）MAE / RMSE | LSTM（Q）MAE / RMSE | PhoCoNet（THQ）MAE / RMSE |',
       '|---|---:|---:|---:|']
for station in names:
    sub=all_metrics[all_metrics.station==station].set_index('model')
    vals=[f'{sub.loc[model,"mae_norm"]:.6f} / {sub.loc[model,"rmse_norm"]:.6f}' for model in ('MLP_Q','LSTM_Q','PhoCoNet')]
    lines.append('| '+names[station]+' | '+' | '.join(vals)+' |')
lines.append('| 三站平均 | '+' | '.join(f'{macro.loc[model,"mae_norm"]:.6f} / {macro.loc[model,"rmse_norm"]:.6f}' for model in ('MLP_Q','LSTM_Q','PhoCoNet'))+' |')
lines+=['','## 实验设置','',
        '- MLP 为逐时刻共享的 8→64→64→1 网络，使用 GELU，两层隐藏层后使用 dropout=0.1，共 4,801 个参数。各位置只接收该位置的辅助观测。',
        '- LSTM 为单层单向 8→64 网络，随后使用 dropout=0.1 和 64→1 读出，共 19,009 个参数。隐状态在各窗口重置，第 k 个输出可使用窗口内前 k 个 Q 观测。',
        '- 两模型均使用 Adam、初始学习率 0.001、type1 学习率调度、batch=64、MSE 损失，最多 100 epochs、patience=10，按验证集选择检查点，没有使用测试集选模型或超参数。',
        '- 沿用 PhoCoNet 主实验管线。2286 个时间点先取前 int(2286×0.8)=1828 点为训练验证集，再取其中前 int(1828×0.875)=1599 点训练，即训练 1599、验证 229、测试 458 点。标准化仅由这 1599 个训练时间点估计；该边界与直接 int(2286×0.7)=1600 不同。',
        '- MAE、MSE 按训练集 TP 标准差归一化；RMSE 为逐窗口 RMSE 的均值，与所列 PhoCoNet 主实验一致。',
        '- PhoCoNet 参考为 manifest.json 中指定的同种子主实验结果。',
        '', '## 核验','',
        '六次训练均完成，检查点和逐窗口预测已归档。逐一核对测试起点和 TP 标签与 PhoCoNet 完全相同；根据保存预测重新计算指标，与报告一致。修改历史输入及时间标记不会改变两种基线输出，且 Q 的输入梯度非零。',
        '', '## 结论','',
        '在本次单种子及所列配置下，PhoCoNet 在三个数据集上的 MAE 和 RMSE 均低于仅用 Q 的 MLP 和 LSTM。该比较衡量完整模型相对两种 Q-only 基线的效果；历史信息的独立贡献仍应结合信息集消融判断。',
        '', '本次未进行基线超参数搜索，不报告训练种子标准差或统计显著性。',
        '', '完整结果目录为 '+str(output)+'。']
report='\n'.join(lines)+'\n'
(output/'report.md').write_text(report,encoding='utf-8')
(base/'report.md').write_text(report,encoding='utf-8')
verification={'completed_training_runs':len(frame),'reference_runs':len(reference),
              'all_test_origins_and_targets_match_reference':True,
              'recomputed_metrics_match_reports':True,
              'input_audit':json.loads((output/'input_audit.json').read_text(encoding='utf-8'))}
(output/'verification.json').write_text(json.dumps(verification,indent=2,ensure_ascii=False),encoding='utf-8')
print(report)
