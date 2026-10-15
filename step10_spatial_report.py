"""在IDE直接运行：结果汇总和留出区排名GIS图。"""
from src.spatial_validation import run_step
from export_spatial_oof_maps import export
from write_spatial_findings import append_findings
if __name__ == '__main__':
    run_step('report')
    export()
    append_findings()
