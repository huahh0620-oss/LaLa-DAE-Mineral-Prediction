"""校验结果完整性，并从每折每配置已保存权重复现预测。直接运行。"""
import pickle,json
import numpy as np
import pandas as pd
import torch
import config as C
import spatial_config as S
from src.autoencoder import DeepAutoencoder
from src.spatial_validation import load,metrics,training_indices,fit_transform

def main():
    d=load();df=pd.read_csv(S.OUTPUT/'metrics_per_fold_seed.csv')
    assert len(df)==S.OUTER_FOLDS*len(S.SEEDS)*len(S.VARIANTS)
    assert not df[['outer_fold','variant','seed']].duplicated().any()
    assert np.isfinite(df[['roc_auc','average_precision','recall_top10pct']]).all().all()
    checked=0
    for f in range(S.OUTER_FOLDS):
        test=np.flatnonzero(d['fold']==f)
        train=training_indices(np.flatnonzero(d['fold']!=f),test,d['rc'],d['valid'])
        for method,k in S.VARIANTS:
            path=S.OUTPUT/'runs'/f'fold{f}'/f'{method}_{k}'
            for seed in S.SEEDS:
                with (path/f'model_seed{seed}.pkl').open('rb') as h:m=pickle.load(h)
                z=np.load(path/f'predictions_seed{seed}.npz');prep=m['prep']
                np.testing.assert_array_equal(z['indices'],test);np.testing.assert_array_equal(z['y'],d['y'][test])
                fresh=fit_transform(d['X'][train],d['y'][train],d['names'],k)
                np.testing.assert_array_equal(fresh['indices'],prep['indices'])
                np.testing.assert_allclose(fresh['scaler'].data_min_,prep['scaler'].data_min_)
                np.testing.assert_allclose(fresh['weights'],prep['weights'])
                T=prep['scaler'].transform(d['X'][test][:,prep['indices']]).astype(np.float32)
                if method in ['deep_autoencoder','shallow_ae']:
                    dims=m['dims']
                    net=DeepAutoencoder(dims,C.ACTIVATION,C.OUTPUT_ACTIVATION) if method=='deep_autoencoder' else torch.nn.Sequential(torch.nn.Linear(k,dims[-1]),torch.nn.Sigmoid(),torch.nn.Linear(dims[-1],k),torch.nn.Sigmoid())
                    net.load_state_dict(m['state_dict']);net.eval()
                    with torch.no_grad(): pred=np.concatenate([net(b).numpy() for b in torch.from_numpy(T).split(2048)])
                    e=(T-pred)**2;score=e@prep['weights'] if method=='deep_autoencoder' else e.mean(1)
                elif method=='pca':score=((T-m['model'].inverse_transform(m['model'].transform(T)))**2).mean(1)
                elif method=='isolation_forest':score=-m['model'].score_samples(T)
                elif method=='ocsvm':score=-m['model'].decision_function(T)
                elif method=='mahalanobis':score=m['model'].mahalanobis(T)
                else:score=T[:,0]
                np.testing.assert_allclose(score,z['score'],rtol=1e-6,atol=1e-8)
                row=df[(df.outer_fold==f)&(df.variant==f'{method}_{k}')&(df.seed==seed)].iloc[0]
                for name,value in metrics(z['y'],z['score']).items():np.testing.assert_allclose(value,row[name],rtol=1e-10,atol=1e-12)
                checked+=1
    result={'status':'passed','saved_models_reproduced':checked,'saved_metrics_recomputed':checked,'all_preprocessing_refit_train_only':True}
    (S.OUTPUT/'result_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)
if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
