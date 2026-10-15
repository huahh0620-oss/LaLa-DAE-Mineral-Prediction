"""有意义的泄漏、空间隔离和Top-K并列值检查。直接运行即可。"""
import unittest
import numpy as np
from src.spatial_validation import load,training_indices,check_split,inner_splits,fit_transform,metrics

class SpatialTests(unittest.TestCase):
    def test_all_outer_and_inner_cluster_disjoint_with_buffer(self):
        d=load()
        for f in np.unique(d['fold']):
            te=np.flatnonzero(d['fold']==f)
            tr=training_indices(np.flatnonzero(d['fold']!=f),te,d['rc'],d['valid'])
            check_split(tr,te,d['rc'],d['y'],d['cluster_positive'])
            for a,b in inner_splits(tr,d):
                self.assertFalse(np.intersect1d(a,te).size)
                self.assertFalse(np.intersect1d(b,te).size)
    def test_heldout_perturbation_cannot_change_selection_or_scaler(self):
        d=load();te=np.flatnonzero(d['fold']==0)
        tr=training_indices(np.flatnonzero(d['fold']!=0),te,d['rc'],d['valid'])
        a=fit_transform(d['X'][tr],d['y'][tr],d['names'],12)
        X=d['X'].copy();y=d['y'].copy();X[te]=1e8;y[te]=1-y[te]
        b=fit_transform(X[tr],y[tr],d['names'],12)
        for key in ['indices','weights','auc']:np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(a['scaler'].data_max_,b['scaler'].data_max_)
        # 测试值超出训练范围时不裁剪，杜绝依赖测试分布重新归一化。
        self.assertGreater(a['scaler'].transform(X[te][:,a['indices']]).max(),1)
    def test_exact_k_and_tie_fraction(self):
        y=np.array([1,0,0,0]);score=np.ones(4)
        m=metrics(y,score)
        self.assertAlmostEqual(m['recall_top10pct'],.1)
        self.assertAlmostEqual(m['recall_top1pct'],.01)
        d=load()
        for k in [12,26,39]:self.assertEqual(len(fit_transform(d['X'],d['y'],d['names'],k)['indices']),k)

if __name__=='__main__':unittest.main()
