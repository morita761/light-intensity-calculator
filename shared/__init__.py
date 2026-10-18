"""リポジトリ共通モジュール

settings  : config/*.txt（INI形式）の読み込み、図の設定 (FigureSettings)
figure    : 図の保存、文字無し版の出力、共通コマンドライン引数
horseshoe : マスク画像からの馬蹄形抽出（色マスク・輪郭・左右分類・画像セット定義）

各フォルダの lib/__init__.py がリポジトリ直下を sys.path に追加してから import する。
"""
