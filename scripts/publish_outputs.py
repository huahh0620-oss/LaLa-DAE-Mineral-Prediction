# -*- coding: utf-8 -*-
"""
成果发布 / Publish outputs to GitHub
====================================================================
把本地运行的成果（图 figure、表 table、关键栅格）提交并推送到 GitHub 仓库，
使代码仓库只保存"可复现的代码 + 论文用图表"，而不包含原始数据。

用法（一般由 run_all.py --publish 自动调用）：
    python -m scripts.publish_outputs
环境变量：
    LALA_GITHUB_REPO   仓库地址（SSH 或 HTTPS），如 git@github.com:user/repo.git
    LALA_GITHUB_BRANCH 分支，默认 main
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config as C
from src.common import log

# 需要发布的成果类型（相对 OUTPUT_DIR）
PUBLISH_ITEMS = ["figures", "tables"]


def _run(cmd, cwd=None, check=True) -> subprocess.CompletedProcess:
    log.info("$ %s", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], cwd=cwd, check=check,
                          capture_output=False, text=True)


def _ensure_git() -> None:
    try:
        subprocess.run(["git", "--version"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError("未检测到 git，请先安装 git 才能发布成果。") from e


def publish(repo_url: str | None = None, branch: str | None = None) -> Path:
    _ensure_git()
    repo_url = repo_url or C.GITHUB_REPO_URL
    branch = branch or C.GITHUB_BRANCH
    if "<USER>" in repo_url:
        raise RuntimeError(
            "尚未配置 GitHub 仓库地址。请设置环境变量 LALA_GITHUB_REPO / "
            "LALA_GITHUB_BRANCH，或修改 config.py 的 GITHUB_* 常量。")

    work = Path(C.CACHE_DIR) / "gh_publish"
    if (work / ".git").exists():
        _run(["git", "fetch", "origin", branch], cwd=work)
        _run(["git", "checkout", branch], cwd=work)
        _run(["git", "reset", "--hard", f"origin/{branch}"], cwd=work)
    else:
        if work.exists():
            shutil.rmtree(work)
        work.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--branch", branch, "--depth", "1",
              repo_url, str(work)], check=True)

    # 复制成果
    for item in PUBLISH_ITEMS:
        src = Path(C.OUTPUT_DIR) / item
        if not src.exists():
            continue
        dst = work / "outputs" / item
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    # 关键栅格
    for tif in Path(C.OUTPUT_DIR).glob("*.tif"):
        dst = work / "outputs" / tif.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tif, dst)
    # 汇总索引
    idx = Path(C.OUTPUT_DIR) / "results_index.json"
    if idx.exists():
        (work / "outputs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(idx, work / "outputs" / idx.name)

    _run(["git", "add", "outputs"], cwd=work)
    status = subprocess.run(["git", "status", "--porcelain", "outputs"],
                            cwd=work, capture_output=True, text=True)
    if not status.stdout.strip():
        log.info("成果无变化，无需提交。")
        return work

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    _run(["git", "-c", "user.name=codex-bot",
          "-c", "user.email=codex-bot@users.noreply.github.com",
          "commit", "-m", f"Update results {stamp}"], cwd=work)
    _run(["git", "push", "origin", branch], cwd=work)
    log.info("成果已发布到 %s (%s)", repo_url, branch)
    return work


if __name__ == "__main__":
    try:
        publish()
    except Exception as e:  # noqa: BLE001
        log.error("发布失败: %s", e)
        sys.exit(1)