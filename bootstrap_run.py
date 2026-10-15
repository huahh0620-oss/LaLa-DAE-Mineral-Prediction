# -*- coding: utf-8 -*-
"""
====================================================================
bootstrap_run.py —— 从 GitHub 实时拉取代码到本地执行（单文件引导脚本）
====================================================================
架构说明（对应"代码在 GitHub、数据在本地"的部署方式）：

    GitHub 仓库            本地机器
    -----------            --------
    代码 *.py      --拉取-->  本地缓存目录  --执行-->  读取本地数据
                                                      |
                                                      v
                                              图/表成果 --推送--> GitHub

本脚本只依赖 Python 标准库，可以单独拷贝到任意机器上运行。它会：
  1. 按清单从 GitHub raw 逐文件下载最新代码到本地缓存目录；
  2. 用本地数据路径（--data-path）运行 run_all.py；
  3. 若加 --publish，则把成果图/表推回 GitHub。

用法：
    python bootstrap_run.py --repo https://github.com/USER/REPO --branch main \
        --data-path "D:\\LaLa_Data"

    # 或用环境变量
    set LALA_GITHUB_RAW=https://raw.githubusercontent.com/USER/REPO/main
    python bootstrap_run.py --data-path "D:\\LaLa_Data" --publish

注意：仓库里只保存代码与成果图表，原始数据永远留在本地。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------
# 需要从 GitHub 拉取的代码清单
# ---------------------------------------------------------------------
CODE_MANIFEST = [
    "config.py",
    "run_all.py",
    "requirements.txt",
    "src/__init__.py",
    "src/common.py",
    "src/crbm.py",
    "src/autoencoder.py",
    "src/step01_prepare_data.py",
    "src/step02_pretrain_crbm.py",
    "src/step03_finetune.py",
    "src/step04_anomaly_map.py",
    "src/step05_evaluate.py",
    "src/step06_benchmark.py",
    "scripts/__init__.py",
    "scripts/publish_outputs.py",
]

DEFAULT_CACHE = Path.home() / ".lala_deep_autoencoder"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="从 GitHub 拉取代码并在本地运行拉拉式铜金矿找矿流程")
    p.add_argument("--repo", default=os.environ.get("LALA_GITHUB_REPO_URL"),
                   help="GitHub 仓库地址，例如 https://github.com/USER/REPO")
    p.add_argument("--branch", default=os.environ.get("LALA_GITHUB_BRANCH", "main"))
    p.add_argument("--raw-base", default=os.environ.get("LALA_GITHUB_RAW"),
                   help="raw 根地址；不填则由 --repo/--branch 推导")
    p.add_argument("--data-path", default=os.environ.get("LALA_DATA_PATH"),
                   help="本地数据根目录（必需）")
    p.add_argument("--output-dir", default=os.environ.get("LALA_OUTPUT_DIR"),
                   help="输出目录（默认在本地缓存目录下的 outputs）")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE), help="本地代码缓存目录")
    p.add_argument("--seeds", default=None, help="逗号分隔随机种子")
    p.add_argument("--skip-benchmark", action="store_true")
    p.add_argument("--publish", action="store_true", help="结束后把成果推回 GitHub")
    p.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"),
                   help="私有仓库访问令牌（可选）")
    return p.parse_args(argv)


def derive_raw_base(repo: str, branch: str) -> str:
    repo = repo.rstrip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    if repo.startswith("https://github.com/"):
        return f"https://raw.githubusercontent.com/{repo[len('https://github.com/'):]}/{branch}"
    if repo.startswith("git@github.com:"):
        slug = repo[len("git@github.com:"):]
        return f"https://raw.githubusercontent.com/{slug}/{branch}"
    raise ValueError(f"无法从 {repo} 推导 raw 地址，请用 --raw-base 显式指定。")


def download_file(url: str, dest: Path, token: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    dest.write_bytes(data)


def fetch_code(raw_base: str, cache_dir: Path, token: str | None = None) -> None:
    raw_base = raw_base.rstrip("/")
    print(f"[bootstrap] 从 {raw_base} 拉取 {len(CODE_MANIFEST)} 个代码文件 ...")
    for rel in CODE_MANIFEST:
        url = f"{raw_base}/{rel}"
        dest = cache_dir / rel
        try:
            download_file(url, dest, token)
            print(f"  [ok]   {rel}")
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {rel}  ({e})")
            raise


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.data_path:
        print("错误：必须通过 --data-path 或环境变量 LALA_DATA_PATH 指定本地数据目录。")
        return 2

    raw_base = args.raw_base or derive_raw_base(args.repo, args.branch)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir \
        else cache_dir / "outputs"

    print("=" * 68)
    print("拉拉式铜金矿深度自编码器找矿 —— GitHub 引导运行")
    print("=" * 68)
    print(f"  代码来源 : {raw_base}")
    print(f"  本地缓存 : {cache_dir}")
    print(f"  数据目录 : {args.data_path}")
    print(f"  输出目录 : {output_dir}")
    print("=" * 68)

    fetch_code(raw_base, cache_dir, args.token)

    env = dict(os.environ)
    env["LALA_DATA_PATH"] = str(args.data_path)
    env["LALA_OUTPUT_DIR"] = str(output_dir)
    env["PYTHONPATH"] = str(cache_dir) + os.pathsep + env.get("PYTHONPATH", "")
    if args.publish:
        env["LALA_PUBLISH"] = "1"
        repo_url = args.repo or env.get("LALA_GITHUB_REPO", "")
        if repo_url and not repo_url.endswith(".git"):
            repo_url = repo_url + ".git"
        env["LALA_GITHUB_REPO"] = repo_url
        env["LALA_GITHUB_BRANCH"] = args.branch

    cmd = [sys.executable, str(cache_dir / "run_all.py")]
    if args.seeds:
        cmd += ["--seeds", args.seeds]
    if args.skip_benchmark:
        cmd += ["--skip-benchmark"]
    if args.publish:
        cmd += ["--publish"]

    print(f"[bootstrap] 执行: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cache_dir), env=env)
    if proc.returncode == 0:
        print(f"[bootstrap] 运行完成，成果位于 {output_dir}")
    else:
        print(f"[bootstrap] 运行失败，返回码 {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())