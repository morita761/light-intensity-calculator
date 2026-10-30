"""arrow_analysis 共通モジュール（パス・馬蹄形の向き推定・極座標ヒストグラム）

リポジトリ直下の shared/ を使うため、ここで sys.path にリポジトリ直下を追加する。
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
