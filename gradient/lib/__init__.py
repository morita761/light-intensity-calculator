"""gradient 共通モジュール（パス・長方形領域の輝度プロファイル）

リポジトリ直下の shared/ を使うため、ここで sys.path にリポジトリ直下を追加する。
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
