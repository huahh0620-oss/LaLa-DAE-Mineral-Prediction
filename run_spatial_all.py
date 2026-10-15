"""直接点击运行：空间分块、折内筛选调参、多种子留出评价、完整基准汇总。"""
from src.spatial_validation import run_step
from src.spatial_run_guard import guard
from export_spatial_oof_maps import export
from write_spatial_findings import append_findings
if __name__ == '__main__':
    run_step('prepare')
    guard()
    run_step('train')
    run_step('report')
    export()
    append_findings()
