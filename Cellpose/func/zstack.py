"""
ND2形式のZ-stack画像を読み込み、Z方向の範囲ごとに投影(プロジェクション)画像を作る。

背景:
    1枚のZスライスだけで馬蹄形検出を行うと、Z方向のノイズや、異なるZ位置に
    ある軸索が重なって見えることによる誤検出が起きやすい。そこで、Z-stackを
    複数のZ範囲に分けてそれぞれ重ね合わせ(投影)を作り、範囲ごとに独立して
    検出を行えるようにする。
"""
import cv2
import numpy as np


def load_nd2_zstack(path, channel=None):
    """
    ND2ファイルを読み込み、(Z, H, W) のグレースケールZ-stack配列を返す。

    channel: 複数チャンネルがある場合に使用するチャンネルのインデックス。
             省略時は先頭チャンネル(0)を使う。
    T(タイムラプス)軸がある場合は先頭フレーム(0)のみを使う。
    """
    try:
        import nd2
    except ImportError as e:
        raise ImportError(
            "ND2ファイルの読み込みには 'nd2' パッケージが必要です。"
            " `pip install nd2` でインストールしてください。"
        ) from e

    with nd2.ND2File(path) as f:
        sizes = f.sizes  # 例: {'T': 1, 'Z': 20, 'C': 2, 'Y': 512, 'X': 512}
        if "Z" not in sizes:
            raise ValueError(
                f"'{path}' にZ軸が見つかりません（sizes={sizes}）。Z-stack画像ではない可能性があります。"
            )
        data = f.asarray()
        dim_order = list(sizes.keys())

    axes = {name: i for i, name in enumerate(dim_order)}

    if "C" in axes:
        n_channels = sizes["C"]
        c_idx = 0 if channel is None else channel
        if not (0 <= c_idx < n_channels):
            raise ValueError(
                f"--channel の値が不正です: {c_idx}（'{path}' のチャンネル数は{n_channels}）"
            )
        data = np.take(data, c_idx, axis=axes["C"])
        dim_order = [d for d in dim_order if d != "C"]
        axes = {name: i for i, name in enumerate(dim_order)}

    if "T" in axes:
        data = np.take(data, 0, axis=axes["T"])
        dim_order = [d for d in dim_order if d != "T"]
        axes = {name: i for i, name in enumerate(dim_order)}

    if sorted(dim_order) != ["X", "Y", "Z"]:
        raise ValueError(
            f"'{path}' の次元構成に対応していません（残った軸={dim_order}）。"
            " Z, Y, X 以外にT, C以外の軸を含むND2ファイルは未対応です。"
        )
    data = np.moveaxis(data, [axes["Z"], axes["Y"], axes["X"]], [0, 1, 2])
    return data


def zstack_ranges(n_z, window_size=5, step=3, include_full=True):
    """
    比較用のZ範囲リストを作る。

    - include_full=True の場合、全Zスライスをまとめた範囲を先頭に含める。
    - window_size枚ずつ、step枚おきにずらした範囲を続けて生成する
      （例: n_z=20, window_size=5, step=3 -> Z1-5, Z4-8, Z7-11, Z10-14, Z13-17, Z16-20）。

    戻り値: [(label, [z_index, ...]), ...]  z_indexは0始まり。
    """
    if n_z <= 0:
        raise ValueError(f"Z-stackのスライス数が不正です: {n_z}")
    if window_size <= 0 or step <= 0:
        raise ValueError(
            f"--window-size / --step は正の整数にしてください: window_size={window_size}, step={step}"
        )

    ranges = []
    if include_full:
        ranges.append((f"full_z1-{n_z}", list(range(n_z))))

    start = 0
    while start + window_size <= n_z:
        z_indices = list(range(start, start + window_size))
        label = f"z{z_indices[0] + 1}-{z_indices[-1] + 1}"
        ranges.append((label, z_indices))
        start += step

    n_sliding = len(ranges) - (1 if include_full else 0)
    if n_sliding == 0:
        raise ValueError(
            f"--window-size={window_size} が n_z={n_z}枚のZ-stackに対して大きすぎるため、"
            "スライディング範囲を1つも作成できませんでした。--window-sizeを小さくしてください。"
        )

    return ranges


def project(stack, z_indices, method="max"):
    """
    Z-stack配列から指定したZインデックス群を抽出し、2D画像に投影する。
    """
    sub = stack[z_indices]
    if method == "max":
        projected = sub.max(axis=0)
    elif method == "mean":
        projected = sub.mean(axis=0)
    elif method == "sum":
        projected = sub.sum(axis=0).astype(np.float64)
    else:
        raise ValueError(
            f"未対応のprojection方式です: {method}（max/mean/sumのいずれかを指定してください）"
        )
    return projected


def to_uint8(image):
    """任意dtype/レンジの2D画像を、0-255のuint8に正規化する。"""
    normalized = cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)
