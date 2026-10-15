"""在IDE直接运行。先运行step08，完成后运行step10。"""
from src.spatial_validation import run_step
from src.spatial_run_guard import guard
if __name__ == '__main__':
    guard()
    run_step('train')
