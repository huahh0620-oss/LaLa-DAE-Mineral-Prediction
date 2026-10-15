"""导出GIS空间区及DAE留出区相对排名图。排名不代表真实成矿概率。"""
import numpy as np
import rasterio
from affine import Affine
from scipy.stats import rankdata
import spatial_config as S
from src.spatial_validation import load,grid

def export():
    d=load();meta=dict(driver='GTiff',height=d['valid'].shape[0],width=d['valid'].shape[1],count=1,
        transform=Affine(*d['transform'][:6]),crs=str(d['crs']),compress='deflate')
    with rasterio.open(S.OUTPUT/'spatial_holdout_fold.tif','w',dtype='int16',nodata=-1,**meta) as r:
        r.write(grid(d['fold'].astype('int16'),d['valid'],-1),1)
    for k in [12,26,39]:
        scores=np.zeros(len(d['y']),dtype=np.float32)
        for f in range(S.OUTER_FOLDS):
            ranks=[]
            for seed in S.SEEDS:
                z=np.load(S.OUTPUT/'runs'/f'fold{f}'/f'deep_autoencoder_{k}'/f'predictions_seed{seed}.npz')
                ranks.append(rankdata(z['score'],method='average')/len(z['score']))
            scores[z['indices']]=np.mean(ranks,axis=0)
        with rasterio.open(S.OUTPUT/f'dae{k}_oof_within_region_rank.tif','w',dtype='float32',nodata=-9999,**meta) as r:
            r.write(grid(scores,d['valid'],-9999),1)
            r.update_tags(meaning='Mean of seed-wise held-out WITHIN-REGION score percentile; NOT calibrated probability; NOT globally comparable ranking')
    print('已导出5折区域图与3套DAE空间留出区内相对排名图')
if __name__=='__main__':export()
