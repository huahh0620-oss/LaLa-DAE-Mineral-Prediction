"""按矿点簇空间嵌套验证。禁止调用旧的全区归一化/特征缓存。"""
from __future__ import annotations
import hashlib, json, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits
import spatial_config as S
import config as C


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def load():
    d=np.load(S.OUTPUT/'data.npz',allow_pickle=False)
    return {k:d[k] for k in d.files}


def prepare():
    import rasterio
    from src.common import locate_geochem_dir,list_geochem_rasters,feature_name_from_path,read_aligned_stack,locate_raster
    if C.RAW_COMPOSITION: raise ValueError('本验证适用于上传的CLR栅格。原始组分需另行定义变换与变量选择。')
    S.OUTPUT.mkdir(parents=True,exist_ok=True)
    files=list_geochem_rasters(locate_geochem_dir(S.DATA_PATH,C.GEOCHEM_SUBDIR,C.GEOCHEM_GLOB,C.AUTO_LOCATE_GEOCHEM),C.GEOCHEM_GLOB)
    cube,meta=read_aligned_stack(files); valid=np.isfinite(cube).all(axis=0)
    with rasterio.open(files[0]) as ref:
        transform,crs=ref.transform,ref.crs
    labelpath=locate_raster(S.DATA_PATH,C.POSITIVE_RASTER,C.POSITIVE_SEARCH_PARENTS)
    with rasterio.open(labelpath) as r:
        if (r.height,r.width)!=valid.shape or r.transform!=transform or r.crs!=crs:
            raise ValueError('标签与地化栅格网格/变换/CRS不一致，禁止隐式重采样')
        labels=r.read(1,masked=True).filled(0)
    if C.MASK_RASTER:
        with rasterio.open(locate_raster(S.DATA_PATH,C.MASK_RASTER,C.POSITIVE_SEARCH_PARENTS)) as r:
            if r.shape!=valid.shape or r.transform!=transform or r.crs!=crs: raise ValueError('mask未对齐')
            valid &= r.read(1,masked=True).filled(0)!=0
    y=(labels[valid]==1).astype(np.uint8); rc=np.argwhere(valid)
    X=cube[:,valid].T.astype(np.float32)
    if X.shape[1]!=39: raise ValueError(f'当前协议预期39变量，实际{X.shape[1]}')
    pos=rc[y==1]
    cluster=DBSCAN(eps=S.CLUSTER_RADIUS_PIXELS,min_samples=1).fit_predict(pos)
    centroids=np.array([pos[cluster==i].mean(0) for i in np.unique(cluster)])
    if len(centroids)<S.OUTER_FOLDS: raise ValueError('独立距离簇不足5个，需修改预先登记的划分设置')
    groups=KMeans(n_clusters=S.OUTER_FOLDS,random_state=S.SPLIT_SEED,n_init=20).fit_predict(centroids)
    # 所有像元按最近的正样本点划入该点所在簇的区域，完整簇不会拆分。
    nearest=cKDTree(pos).query(rc)[1]; fold=groups[cluster[nearest]]
    np.savez_compressed(S.OUTPUT/'data.npz',X=X,y=y,rc=rc,valid=valid,fold=fold,
        names=np.asarray([feature_name_from_path(f) for f in files],dtype=str),
        transform=np.asarray(tuple(transform)),crs=str(crs),cluster_positive=cluster)
    pd.DataFrame({'row':pos[:,0],'col':pos[:,1],'cluster':cluster,'outer_fold':groups[cluster]}).to_csv(S.OUTPUT/'deposit_clusters.csv',index=False)
    np.save(S.OUTPUT/'fold_grid.npy',grid(fold,valid,-1))
    audits=[]
    for f in range(S.OUTER_FOLDS):
        te=np.flatnonzero(fold==f);tr=training_indices(np.flatnonzero(fold!=f),te,rc,valid)
        check_split(tr,te,rc,y,cluster)
        audits.append(audit(tr,te,rc,y,f))
    pd.DataFrame(audits).to_csv(S.OUTPUT/'split_audit.csv',index=False)
    sources=[{'file':str(f.relative_to(S.DATA_PATH)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files+[labelpath]]
    write_json(S.OUTPUT/'protocol.json',{'valid_pixels':len(y),'positive_pixels':int(y.sum()),
        'distance_proxy_clusters':len(centroids),'pixel_size_map_units':[abs(transform.a),abs(transform.e)],
        'crs':str(crs),'cluster_radius_pixels':S.CLUSTER_RADIUS_PIXELS,'buffer_pixels':S.BUFFER_PIXELS,
        'outer_folds':S.OUTER_FOLDS,'inner_folds':S.INNER_FOLDS,'split_seed':S.SPLIT_SEED,'seeds':S.SEEDS,
        'tuning_seed':S.TUNING_SEED,'crbm_epochs':S.CRBM_EPOCHS,'finetune_epochs':S.FINETUNE_EPOCHS,
        'max_fit_pixels':S.MAX_FIT_PIXELS,'batch_size':S.BATCH_SIZE,'ocsvm_max_fit':S.OCSVM_MAX_FIT,'grids':S.GRIDS,'variants':S.VARIANTS,
        'selection':'train-only descending signed element AUC; exactly top12/top26; all39; no AUC or correlation cutoff',
        'weights':'train-only max(AUC-0.5,0), normalized; uniform fallback',
        'scaling':'train-only minmax, validation/test NOT clipped',
        'background':'unlabelled pixels, not verified negatives','sources':sources,
        'scope':'same supplied interpolated CLR raster product; not raw-sample interpolation-independent validation',
        'area_ranking':'within each held-out area; ties receive fractional credit at exact area budget'})
    print('空间划分完成',audits,flush=True)


def grid(values,valid,nodata):
    a=np.full(valid.shape,nodata,dtype=np.asarray(values).dtype); a[valid]=values; return a


def training_indices(candidate,held,rc,valid):
    h=np.zeros(valid.shape,dtype=bool);h[tuple(rc[held].T)]=True
    dist=distance_transform_edt(~h)
    return candidate[dist[tuple(rc[candidate].T)]>S.BUFFER_PIXELS]


def check_split(tr,te,rc,y,cluster):
    assert not np.intersect1d(tr,te).size
    assert set(np.unique(y[tr]))=={0,1} and set(np.unique(y[te]))=={0,1},'每个训练/留出区必须有双类别'
    assert cKDTree(rc[te]).query(rc[tr])[0].min()>S.BUFFER_PIXELS
    mapping=np.full(len(y),-1);mapping[np.flatnonzero(y)]=cluster
    assert not np.intersect1d(mapping[tr[y[tr]==1]],mapping[te[y[te]==1]]).size,'距离簇泄漏'


def audit(tr,te,rc,y,f):
    return {'fold':f,'train_pixels':len(tr),'test_pixels':len(te),'train_positive':int(y[tr].sum()),
            'test_positive':int(y[te].sum()),'minimum_gap_pixels':float(cKDTree(rc[te]).query(rc[tr])[0].min())}


def fit_transform(X,y,names,k):
    """只接收本次训练数组。评估数据无法进入归一化、AUC、权重计算。"""
    auc=np.array([roc_auc_score(y,X[:,j]) for j in range(X.shape[1])])
    idx=np.array([list(names).index('Cu')]) if k==1 else (np.arange(len(names)) if k==39 else np.argsort(-auc,kind='stable')[:k])
    scaler=MinMaxScaler().fit(X[:,idx]);w=np.maximum(auc[idx]-.5,0)
    w=w/w.sum() if w.sum()>0 else np.full(len(idx),1/len(idx))
    return {'indices':idx,'scaler':scaler,'weights':w,'auc':auc,'names':names[idx].tolist()}


def train_score(method,raw_train,ytrain,raw_test,names,k,param,seed,save=None):
    import torch
    from src.autoencoder import DeepAutoencoder
    from src.common import resolve_hidden_sizes
    from src.crbm import CRBM,CRBMConfig
    prep=fit_transform(raw_train,ytrain,names,k)
    X=prep['scaler'].transform(raw_train[:,prep['indices']]).astype(np.float32)
    T=prep['scaler'].transform(raw_test[:,prep['indices']]).astype(np.float32)
    rng=np.random.default_rng(seed);idx=np.sort(rng.choice(len(X),min(S.MAX_FIT_PIXELS,len(X)),replace=False)); A=X[idx]
    torch.manual_seed(seed);np.random.seed(seed)
    history=[];state={};net=None
    if method in ('deep_autoencoder','shallow_ae'):
        dims=[k]+(resolve_hidden_sizes(k,'auto') if method=='deep_autoencoder' else [min(5,max(2,k//2))])
        net=(DeepAutoencoder(dims,C.ACTIVATION,C.OUTPUT_ACTIVATION) if method=='deep_autoencoder' else torch.nn.Sequential(torch.nn.Linear(k,dims[-1]),torch.nn.Sigmoid(),torch.nn.Linear(dims[-1],k),torch.nn.Sigmoid()))
        if method=='deep_autoencoder':
            reps=A.astype(np.float64);crbms=[]
            for li,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
                rbm=CRBM(CRBMConfig(n_visible=a,n_hidden=b,seed=seed+li+1,lr=C.CRBM_LR,
                    sigma=C.CRBM_SIGMA,momentum=C.CRBM_MOMENTUM,weight_decay=C.CRBM_WEIGHT_DECAY,learn_a=False))
                rbm.fit(reps,epochs=S.CRBM_EPOCHS,batch_size=S.BATCH_SIZE,verbose=False)
                crbms.append(rbm);reps=rbm.transform(reps)
            net.load_pretrained_crbms(crbms)
        opt=torch.optim.SGD(net.parameters(),lr=param['lr'],momentum=C.FINETUNE_MOMENTUM,weight_decay=C.FINETUNE_WEIGHT_DECAY)
        a=torch.from_numpy(A)
        for ep in range(S.FINETUNE_EPOCHS):
            total=0.
            for ix in torch.randperm(len(a)).split(S.BATCH_SIZE):
                b=a[ix];opt.zero_grad();loss=(net(b)-b).square().mean();loss.backward();opt.step();total+=float(loss.detach())*len(ix)
            history.append(total/len(a))
        net.eval()
        with torch.no_grad(): pred=np.concatenate([net(t).numpy() for t in torch.from_numpy(T).split(2048)])
        err=(T-pred)**2
        score=err@prep['weights'] if method=='deep_autoencoder' else err.mean(1)
        state={'dims':dims,'state_dict':net.state_dict(),'history':history}
    elif method=='pca':
        from sklearn.decomposition import PCA
        model=PCA(n_components=min(param['components'],k-1),svd_solver='full').fit(A)
        score=((T-model.inverse_transform(model.transform(T)))**2).mean(1);state={'model':model}
    elif method=='isolation_forest':
        from sklearn.ensemble import IsolationForest
        model=IsolationForest(n_estimators=200,max_samples=min(param['max_samples'],len(A)),random_state=seed,n_jobs=1).fit(A)
        score=-model.score_samples(T);state={'model':model}
    elif method=='ocsvm':
        from sklearn.svm import OneClassSVM
        model=OneClassSVM(gamma='scale',nu=param['nu']).fit(A[rng.choice(len(A),min(len(A),S.OCSVM_MAX_FIT),replace=False)])
        score=-model.decision_function(T);state={'model':model}
    elif method=='mahalanobis':
        from sklearn.covariance import ShrunkCovariance
        model=ShrunkCovariance(shrinkage=param['shrinkage']).fit(A)
        score=model.mahalanobis(T);state={'model':model}
    elif method=='single_element_cu': score=T[:,0];state={}
    else: raise ValueError(method)
    if not np.isfinite(score).all(): raise ValueError('非有限预测得分')
    if save:
        save.parent.mkdir(parents=True,exist_ok=True)
        with save.open('wb') as f: pickle.dump({'method':method,'parameters':param,'seed':seed,'prep':prep,**state},f)
    return np.asarray(score),prep


def metrics(y,score):
    m={'roc_auc':float(roc_auc_score(y,score)),'average_precision':float(average_precision_score(y,score))}
    # 等分数跨过面积边界时按比例计入，避免像元索引决定Top-K结果。
    for pct in [1,5,10,20]:
        budget=len(y)*pct/100; cut=np.sort(score)[::-1][min(len(y)-1,int(np.ceil(budget))-1)]
        above=score>cut;equal=score==cut;fraction=(budget-above.sum())/equal.sum()
        m[f'recall_top{pct}pct']=float((y[above].sum()+fraction*y[equal].sum())/y.sum())
    return m


def inner_splits(outer_train,d):
    # 仅按位置组织内层区域，不按元素/测试分数优化边界。每个原外层簇仍完整。
    groups=np.unique(d['fold'][outer_train]);centers=np.array([d['rc'][outer_train[d['fold'][outer_train]==f]].mean(0) for f in groups])
    order=groups[np.argsort(centers[:,0],kind='stable')];splits=[]
    for q,vg in enumerate(np.array_split(order,S.INNER_FOLDS)):
        va=outer_train[np.isin(d['fold'][outer_train],vg)]
        tr=training_indices(outer_train[~np.isin(d['fold'][outer_train],vg)],va,d['rc'],d['valid'])
        check_split(tr,va,d['rc'],d['y'],d['cluster_positive']);splits.append((tr,va))
    return splits


def run_experiments():
    import torch
    torch.set_num_threads(1)
    d=load();X,y,rc,valid,fold,names=[d[k] for k in ['X','y','rc','valid','fold','names']]
    rows=[];tuning=[];audits=[];start=time.time()
    fingerprint=hashlib.sha256((S.OUTPUT/'protocol.json').read_bytes()+Path(__file__).read_bytes()).hexdigest()
    for f in range(S.OUTER_FOLDS):
        test=np.flatnonzero(fold==f);train=training_indices(np.flatnonzero(fold!=f),test,rc,valid)
        inner=inner_splits(train,d)
        for i,(tr,va) in enumerate(inner):audits.append({'outer_fold':f,'inner_fold':i,**audit(tr,va,rc,y,i)})
        for method,k in S.VARIANTS:
            tag=f'{method}_{k}';work=S.OUTPUT/'runs'/f'fold{f}'/tag;work.mkdir(parents=True,exist_ok=True)
            checkpoint=work/'completed.json'
            if checkpoint.exists():
                c=json.loads(checkpoint.read_text(encoding='utf-8'))
                if c['fingerprint']!=fingerprint: raise RuntimeError('源码/协议变更，请删除outputs_spatial/runs后重跑')
                rows.extend(c['rows']);tuning.extend(c['tuning']);print('复用',f,tag,flush=True);continue
            trials=[]
            for pi,param in enumerate(S.GRIDS[method]):
                for i,(tr,va) in enumerate(inner):
                    score,_=train_score(method,X[tr],y[tr],X[va],names,k,param,S.TUNING_SEED)
                    trials.append({'outer_fold':f,'variant':tag,'candidate':pi,'inner_fold':i,'parameters':json.dumps(param),'validation_ap':float(average_precision_score(y[va],score))})
            means=[np.mean([r['validation_ap'] for r in trials if r['candidate']==pi]) for pi in range(len(S.GRIDS[method]))]
            best=int(np.argmax(means));param=S.GRIDS[method][best];new=[]
            for seed in S.SEEDS:
                score,prep=train_score(method,X[train],y[train],X[test],names,k,param,seed,work/f'model_seed{seed}.pkl')
                row={'outer_fold':f,'variant':tag,'method':method,'n_features':k,'seed':seed,'test_pixels':len(test),'test_positive':int(y[test].sum()),'parameters':json.dumps(param),**metrics(y[test],score)}
                np.savez_compressed(work/f'predictions_seed{seed}.npz',indices=test,y=y[test],score=score)
                write_json(work/f'selection_seed{seed}.json',{'features':prep['names'],'indices':prep['indices'].tolist(),'all_train_auc':prep['auc'].tolist(),'weights':prep['weights'].tolist(),'train_indices_sha256':hashlib.sha256(train.tobytes()).hexdigest(),'scaler_min':prep['scaler'].data_min_.tolist(),'scaler_max':prep['scaler'].data_max_.tolist()})
                new.append(row);print(f'外折{f+1}/5 {tag} seed={seed} AUC={row["roc_auc"]:.3f} AP={row["average_precision"]:.4f} 用时{time.time()-start:.0f}s',flush=True)
            write_json(checkpoint,{'fingerprint':fingerprint,'rows':new,'tuning':trials});rows.extend(new);tuning.extend(trials)
            pd.DataFrame(rows).to_csv(S.OUTPUT/'metrics_per_fold_seed.csv',index=False)
            pd.DataFrame(tuning).to_csv(S.OUTPUT/'inner_tuning.csv',index=False)
    pd.DataFrame(audits).to_csv(S.OUTPUT/'inner_split_audit.csv',index=False)
    pd.DataFrame(rows).to_csv(S.OUTPUT/'metrics_per_fold_seed.csv',index=False)
    pd.DataFrame(tuning).to_csv(S.OUTPUT/'inner_tuning.csv',index=False)
    write_json(S.OUTPUT/'run_status.json',{'status':'completed','expected':S.OUTER_FOLDS*len(S.VARIANTS)*len(S.SEEDS),'completed':len(rows),'elapsed_sec':time.time()-start,'fingerprint':fingerprint})


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=load();df=pd.read_csv(S.OUTPUT/'metrics_per_fold_seed.csv')
    expected=S.OUTER_FOLDS*len(S.SEEDS)*len(S.VARIANTS)
    assert len(df)==expected and not df[['outer_fold','variant','seed']].duplicated().any()
    cols=['roc_auc','average_precision','recall_top1pct','recall_top5pct','recall_top10pct','recall_top20pct']
    # 先对同折训练种子平均，再统计5折，不把15条相关记录当独立重复。
    folds=df.groupby(['variant','outer_fold'])[cols].mean().reset_index();folds.to_csv(S.OUTPUT/'metrics_per_fold_seed_mean.csv',index=False)
    summary=folds.groupby('variant')[cols].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(S.OUTPUT/'benchmark_spatial_summary.csv')
    seeds=df.groupby(['variant','seed'])[cols].mean().reset_index();seeds.to_csv(S.OUTPUT/'metrics_per_seed_fold_mean.csv',index=False)
    variability=seeds.groupby('variant')[cols].agg(['mean','std']);variability.columns=['_'.join(c) for c in variability.columns];variability.to_csv(S.OUTPUT/'seed_stability_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5),dpi=160)
    fg=grid(d['fold'],d['valid'],-1); axes[0].imshow(np.ma.masked_less(fg,0),cmap='tab10',vmin=0,vmax=9)
    pos=d['rc'][d['y']==1];axes[0].scatter(pos[:,1],pos[:,0],c='black',s=15);axes[0].set_title('Five held-out spatial regions (black: known pixels)')
    for variant,sub in seeds.groupby('variant'):axes[1].plot(sub.seed,sub.roc_auc,marker='o',label=variant)
    axes[1].set_ylim(0,1);axes[1].set_xticks(S.SEEDS);axes[1].set_ylabel('Macro spatial-fold ROC-AUC');axes[1].legend(fontsize=6);axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(S.OUTPUT/'spatial_regions_and_seeds.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,7),dpi=160)
    for ax,metric,title in zip(axes,['roc_auc','recall_top10pct'],['ROC-AUC','Recall at top 10% of each test region']):
        ax.barh(summary.index,summary[metric+'_mean'],xerr=summary[metric+'_std'],capsize=3)
        ax.set_xlim(0,1);ax.set_title(title+' (five-fold mean ± SD)');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(S.OUTPUT/'benchmark_spatial.png');plt.close(fig)
    # 每个分区单独画曲线，不跨模型池化原始异常得分。
    fig,axes=plt.subplots(1,2,figsize=(12,5),dpi=160)
    for f in range(S.OUTER_FOLDS):
        scores=[]
        for seed in S.SEEDS:
            z=np.load(S.OUTPUT/'runs'/f'fold{f}'/'deep_autoencoder_12'/f'predictions_seed{seed}.npz')
            from scipy.stats import rankdata
            scores.append(rankdata(z['score'],method='average')/len(z['score']))
        score=np.mean(scores,axis=0);y=z['y'];a,b,_=roc_curve(y,score);pr,re,_=precision_recall_curve(y,score)
        axes[0].plot(a,b,label=f'Fold {f+1}');axes[1].plot(re,pr,label=f'Fold {f+1}')
    axes[0].plot([0,1],[0,1],'k--');axes[0].set(xlabel='False positive rate',ylabel='Recall',title='DAE-12, per-region seed-rank ensemble ROC')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='DAE-12, per-region seed-rank ensemble PR')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(S.OUTPUT/'dae12_spatial_roc_pr.png');plt.close(fig)
    lines=['# 空间独立验证与多随机种子实际结果','',f'已完成 {len(df)}/{expected} 次外层训练评价，5折空间区×3训练种子×9配置。',
    '', '## 协议与解释','', '- DBSCAN半径6像元（12km）形成距离代理簇，整簇留出。完整测试区域外排除3像元（6km）训练缓冲。','- 外层5折，每次外层训练区内2折空间调参，目标为平均AP。筛选、权重和归一化只接触当前训练区。','- 12/26变量按训练单元素原方向AUC精确取前N个，不沿用旧流程的AUC下限和相关去冗余。39变量全部输入。','- DAE采用原工程CRBM预训练和对称网络，重构误差权重来自训练AUC。预算为每层40轮CRBM、80轮微调，最多8000训练像元。','- 外层测试数据从未用于拟合或参数选择。固定空间分区的三个训练种子反映初始化/抽样波动，未重新寻找有利分区。','- 背景是未标注区域，ROC/AP不是已确认无矿样本上的精确分类结论。AP对应稀疏矿点基准率。','- Top-K在各留出区内排序，面积按像元数计算，同分边界采用比例分配。结果为像元召回率。','- 下表先在同折内平均三个种子，再报告五折均值±样本标准差。不能把15条相关结果视为15个独立地质样本。','- 原始CLR/IDW栅格已在上游生成，缺少原始采样记录，无法折内重做插值。因而空间独立性只覆盖本次模型流程。','- 距离簇不等于实际独立矿床，尚需用矿床ID、地质边界复核。半径/分区固定为本次预设，未做尺度敏感性实验。','', '## 完整基准表','', '| 配置 | ROC-AUC | AP | Top1%召回 | Top5%召回 | Top10%召回 | Top20%召回 |','|---|---:|---:|---:|---:|---:|---:|']
    for v,r in summary.iterrows():lines.append('| '+v+' | '+' | '.join(f'{r[c+"_mean"]:.3f} ± {r[c+"_std"]:.3f}' for c in cols)+' |')
    lines+=['','## 文件与复现','','- `metrics_per_fold_seed.csv`：135条外层原始指标。','- `inner_tuning.csv`：全部内层候选结果，记录选参依据。','- `split_audit.csv`、`inner_split_audit.csv`：双类别数、训练/测试最小距离。','- `runs/`：每折每配置每种子的模型、元素列表、标准化参数、测试索引和预测分数。','- `seed_stability_summary.csv`：先平均五折，再报告三种子波动，区别于折间标准差。','- `protocol.json`：源数据SHA256和预设协议。','- IDE依次运行step08、step09、step10，或直接运行run_spatial_all.py。旧outputs保留为回顾性实验，不能与空间留出表混用。']
    (S.OUTPUT/'RESULTS_SPATIAL.md').write_text('\n'.join(lines),encoding='utf-8')
    print(summary[['roc_auc_mean','roc_auc_std','average_precision_mean','recall_top10pct_mean']].to_string(),flush=True)


def run_step(step):
    with threadpool_limits(limits=1):
        return {'prepare':prepare,'train':run_experiments,'report':report}[step]()
