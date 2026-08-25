"""
CLIスクリプト共通のエラー表示ヘルパー。

引数の指定ミス（未知のオプション・必須引数の欠落など）はargparseが
Usageとエラー内容を自動表示するため、ここでは扱わない。
このヘルパーは、argparse通過後に発生しうる失敗（存在しないファイル/
ディレクトリの指定、不正なデータ内容など）を、生のトレースバックではなく
原因が分かる日本語メッセージに変換する。
"""
import sys


def run_main(main_func):
    try:
        main_func()
    except FileNotFoundError as e:
        print(f"\nエラー: 指定されたファイルが見つかりません。\n  {e}", file=sys.stderr)
        print("パスが正しいか確認してください（-h でヘルプ・使い方を表示できます）。", file=sys.stderr)
        sys.exit(1)
    except NotADirectoryError as e:
        print(f"\nエラー: 指定されたディレクトリが存在しません。\n  {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nエラー: 入力データの内容が不正です。\n  {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n中断されました。", file=sys.stderr)
        sys.exit(130)


def require_file(path, description):
    """指定パスにファイルが存在しなければ、分かりやすい日本語メッセージで FileNotFoundError を送出する。"""
    import os

    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description}が見つかりません: {path}")


def require_dir(path, description):
    """指定パスにディレクトリが存在しなければ、分かりやすい日本語メッセージで NotADirectoryError を送出する。"""
    import os

    if not os.path.isdir(path):
        raise NotADirectoryError(f"{description}が見つかりません: {path}")
