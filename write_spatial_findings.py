"""把实际空间结果的含义写入报告，数值始终来自生成的基准CSV。"""
import pandas as pd
import spatial_config as S

def append_findings():
    df=pd.read_csv(S.OUTPUT/'benchmark_spatial_summary.csv',index_col=0)
    rows=['','## 实际结果说明','']
    for k in [12,26,39]:
        r=df.loc[f'deep_autoencoder_{k}']
        rows.append(f'- DAE-{k}：ROC-AUC {r.roc_auc_mean:.3f}±{r.roc_auc_std:.3f}，AP {r.average_precision_mean:.3f}±{r.average_precision_std:.3f}，Top10%面积召回 {100*r.recall_top10pct_mean:.2f}%±{100*r.recall_top10pct_std:.2f}个百分点。')
    cu=df.loc['single_element_cu_1']
    rows+=['',f'Cu单元素基准ROC-AUC为{cu.roc_auc_mean:.3f}±{cu.roc_auc_std:.3f}。本次点估计没有显示DAE优于该简单基准，不能宣称深度模型带来明确优势。',
        '', '不同留出区差异较大，尤其有一折只有1个正样本。低AP需结合矿点在全区极稀疏的基准率理解，但Top10%召回和折间波动也表明当前找矿排序仍不稳定。不得删去较差折，或事后根据测试成绩重新选择簇半径。',
        '', '本次已经完成现有数据可支持的缺项。严格的实际矿床簇划分仍须矿床ID核查，上游插值独立性须原始采样数据。新空间结果与旧回顾性结果的训练预算和筛选规则也有差异，不能把两者差值全部归因于数据泄漏。',
        '', 'GIS输出包含空间区和DAE-12/26/39各留出区内的相对排名栅格。排名图不是校准成矿概率，不宜直接跨区比较或解释分区接缝。']
    p=S.OUTPUT/'RESULTS_SPATIAL.md';text=p.read_text(encoding='utf-8').split('\n## 实际结果说明')[0]
    p.write_text(text+'\n'+'\n'.join(rows),encoding='utf-8')
if __name__=='__main__':append_findings()
