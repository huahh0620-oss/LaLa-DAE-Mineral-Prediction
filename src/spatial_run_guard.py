"""恢复实验前检查配置、源码和数据，拒绝混用不同实验的检查点。"""
import hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas,sklearn,torch,rasterio,scipy
import spatial_config as S


def guard():
    root=Path(__file__).resolve().parents[1]
    files=['config.py','spatial_config.py','src/spatial_validation.py','src/crbm.py','src/autoencoder.py','src/common.py']
    hashes={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}
    with np.load(S.OUTPUT/'data.npz',allow_pickle=False) as d:
        datahash={k:hashlib.sha256(np.ascontiguousarray(d[k]).tobytes()).hexdigest() for k in sorted(d.files)}
    current={'source_sha256':hashes,'data_array_sha256':datahash}
    target=S.OUTPUT/'execution_manifest.json'
    if target.exists():
        old=json.loads(target.read_text(encoding='utf-8'))
        if old['experiment_identity']!=current:raise RuntimeError('配置、源码或数据已变更。请更换spatial_config.OUTPUT目录，避免混用已完成结果。')
    else:
        target.write_text(json.dumps({'experiment_identity':current,'environment':{'python':platform.python_version(),
            'numpy':np.__version__,'pandas':pandas.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,
            'rasterio':rasterio.__version__,'scipy':scipy.__version__}},ensure_ascii=False,indent=2),encoding='utf-8')
