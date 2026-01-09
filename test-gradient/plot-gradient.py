import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# デバッグ画像表示のデフォルトサイズ
DEBUG_WINDOW_WIDTH = 800
DEBUG_WINDOW_HEIGHT = 600

# 検出可能な色の定義（HSV範囲）
COLOR_RANGES = {
    'blue': {
        'lower': np.array([90, 50, 50]),
        'upper': np.array([130, 255, 255]),
        'display': (255, 0, 0),  # BGR for rectangle drawing
    },
    'yellow': {
        'lower': np.array([20, 50, 50]),
        'upper': np.array([35, 255, 255]),
        'display': (0, 255, 255),  # BGR
    },
    'red': {
        # 赤は色相が0付近と180付近に分かれるため2つの範囲
        'lower': np.array([0, 50, 50]),
        'upper': np.array([10, 255, 255]),
        'lower2': np.array([160, 50, 50]),
        'upper2': np.array([180, 255, 255]),
        'display': (0, 0, 255),  # BGR
    },
    'green': {
        'lower': np.array([35, 50, 50]),
        'upper': np.array([85, 255, 255]),
        'display': (0, 255, 0),  # BGR
    },
    'cyan': {
        'lower': np.array([80, 50, 50]),
        'upper': np.array([100, 255, 255]),
        'display': (255, 255, 0),  # BGR
    },
    'magenta': {
        'lower': np.array([140, 50, 50]),
        'upper': np.array([160, 255, 255]),
        'display': (255, 0, 255),  # BGR
    },
}


def get_color_mask(hsv, color_name):
    """
    指定した色のマスクを取得します。

    Args:
        hsv: HSV画像
        color_name: 色の名前（COLOR_RANGESのキー）

    Returns:
        マスク画像
    """
    if color_name not in COLOR_RANGES:
        raise ValueError(f"未対応の色: {color_name}. 対応色: {list(COLOR_RANGES.keys())}")

    color_def = COLOR_RANGES[color_name]
    mask = cv2.inRange(hsv, color_def['lower'], color_def['upper'])

    # 赤のように2つの範囲がある場合
    if 'lower2' in color_def:
        mask2 = cv2.inRange(hsv, color_def['lower2'], color_def['upper2'])
        mask = cv2.bitwise_or(mask, mask2)

    return mask


def get_combined_color_mask(hsv, color_names):
    """
    複数の色のマスクを結合して取得します。

    Args:
        hsv: HSV画像
        color_names: 色の名前のリスト

    Returns:
        結合されたマスク画像
    """
    combined_mask = None
    for color_name in color_names:
        mask = get_color_mask(hsv, color_name)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    return combined_mask


def show_debug_image(img, title, width=DEBUG_WINDOW_WIDTH, height=DEBUG_WINDOW_HEIGHT):
    """
    デバッグ用に画像を指定サイズで表示します。

    Args:
        img: 表示する画像
        title: ウィンドウタイトル
        width: ウィンドウ幅
        height: ウィンドウ高さ
    """
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, width, height)
    cv2.imshow(title, img)


def show_debug_images_sequential(images_with_titles, width=DEBUG_WINDOW_WIDTH, height=DEBUG_WINDOW_HEIGHT):
    """
    複数のデバッグ画像を順番に表示します。

    Args:
        images_with_titles: [(img, title), ...] のリスト
        width: ウィンドウ幅
        height: ウィンドウ高さ
    """
    for i, (img, title) in enumerate(images_with_titles):
        window_title = f"[{i+1}/{len(images_with_titles)}] {title}"
        show_debug_image(img, window_title, width, height)
        print(f"画像 {i+1}/{len(images_with_titles)}: {title} - 任意のキーで次へ進む")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def interpolate_profiles(profiles, target_length=100):
    """
    線形補間により、すべてのプロファイルを指定された長さに揃える。

    Args:
        profiles (List[np.ndarray]): 各プロファイル（1次元配列）のリスト。
        target_length (int): 補間後の長さ。

    Returns:
        np.ndarray: (N, target_length) 形状の2次元配列（N: プロファイル数）
    """
    interpolated_profiles = []
    for profile in profiles:
        original_length = len(profile)
        x_original = np.linspace(0, 1, original_length)
        x_target = np.linspace(0, 1, target_length)
        interpolated = np.interp(x_target, x_original, profile)
        interpolated_profiles.append(interpolated)
    return np.array(interpolated_profiles)


def extract_raw_profiles(image_path, colors=None):
    """
    画像内の指定色の長方形領域の内側の緑色輝度プロファイルを抽出します（ノーマライズなし）。

    Args:
        image_path (str): 画像ファイルのパス。
        colors (list): 検出する色のリスト（例: ['blue', 'yellow']）。Noneの場合は['blue']。

    Returns:
        tuple: (all_raw_profiles, img_with_rects) または エラー時は (None, None)
    """
    if colors is None:
        colors = ['blue']

    img = cv2.imread(image_path)

    if img is None:
        print(f"エラー: 画像 '{image_path}' を読み込めませんでした。")
        return None, None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = get_combined_color_mask(hsv, colors)
    # RETR_CCOMPで階層構造を取得し、内側の輪郭を選択
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or hierarchy is None:
        print(f"エラー: 画像 '{image_path}' に指定色({colors})の長方形が見つかりませんでした。")
        return None, None

    all_raw_profiles = []

    # 内側の輪郭（親を持つ輪郭）のみを処理
    for i, contour in enumerate(contours):
        # hierarchy[0][i][3] は親輪郭のインデックス。-1でなければ内側の輪郭
        if hierarchy[0][i][3] == -1:
            continue  # 外側の輪郭はスキップ

        if cv2.contourArea(contour) < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        roi = img[y:y+h, x:x+w]
        _, g_channel, _ = cv2.split(roi)
        green_intensity_profile = np.mean(g_channel, axis=0)

        all_raw_profiles.append(green_intensity_profile)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

    if not all_raw_profiles:
        print(f"エラー: 画像 '{image_path}' に有効な指定色({colors})の内側領域が見つかりませんでした。")
        return None, None

    return all_raw_profiles, img


def extract_average_profile(image_path, colors=None, normalize=True, global_max=None):
    """
    画像内の指定色の長方形領域の緑色輝度プロファイルを抽出し、平均プロファイルを返します。

    Args:
        image_path (str): 画像ファイルのパス。
        colors (list): 検出する色のリスト（例: ['blue', 'yellow']）。Noneの場合は['blue']。
        normalize (bool): ノーマライズを行うかどうか。
        global_max (float): グローバルな最大値でノーマライズする場合に指定。

    Returns:
        tuple: (average_profile, all_profiles, img_with_rects) または エラー時は (None, None, None)
    """
    raw_profiles, img = extract_raw_profiles(image_path, colors)

    if raw_profiles is None:
        return None, None, None

    # ノーマライズ処理
    if normalize:
        if global_max is None:
            # 各プロファイルの最大値でノーマライズ
            all_intensity_profiles = []
            for profile in raw_profiles:
                max_val = np.max(profile)
                if max_val > 0:
                    all_intensity_profiles.append(profile / max_val)
                else:
                    all_intensity_profiles.append(profile)
        else:
            # グローバル最大値でノーマライズ
            all_intensity_profiles = [profile / global_max if global_max > 0 else profile for profile in raw_profiles]
    else:
        all_intensity_profiles = raw_profiles

    target_length = min([len(p) for p in all_intensity_profiles])
    interpolated_profiles = interpolate_profiles(all_intensity_profiles, target_length=target_length)
    average_profile = np.mean(interpolated_profiles, axis=0)

    return average_profile, all_intensity_profiles, img


def plot_multi_region_green_intensity_profiles(image_path, colors=None):
    """
    画像内の指定色の長方形領域の緑色輝度プロファイルを抽出し、正規化してプロットします。
    各領域のプロットと、それらの平均プロットの2種類を出力します。

    Args:
        image_path (str): 画像ファイルのパス。
        colors (list): 検出する色のリスト（例: ['blue', 'yellow']）。Noneの場合は['blue']。
    """
    if colors is None:
        colors = ['blue']

    # 1. 画像の読み込み
    img = cv2.imread(image_path)

    if img is None:
        print(f"エラー: 画像 '{image_path}' を読み込めませんでした。")
        return

    # 2. 指定色の長方形の識別
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = get_combined_color_mask(hsv, colors)

    # RETR_CCOMPで階層構造を取得し、内側の輪郭を選択
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or hierarchy is None:
        print(f"エラー: 画像内に指定色({colors})の長方形が見つかりませんでした。HSVの範囲を調整してみてください。")
        show_debug_image(mask, f"Color Mask ({colors}) for debug")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 各長方形のプロファイルを格納するリスト
    all_intensity_profiles = []
    detected_rects = []

    # 内側の輪郭（親を持つ輪郭）のみを処理
    for i, contour in enumerate(contours):
        # hierarchy[0][i][3] は親輪郭のインデックス。-1でなければ内側の輪郭
        if hierarchy[0][i][3] == -1:
            continue  # 外側の輪郭はスキップ

        # 小さすぎる輪郭はノイズとみなしてスキップ
        if cv2.contourArea(contour) < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # 指定領域内の緑色輝度を抽出
        roi = img[y:y+h, x:x+w]
        _, g_channel, _ = cv2.split(roi)

        # 緑色輝度プロファイルを作成 (各列の平均)
        green_intensity_profile = np.mean(g_channel, axis=0)

        # プロファイルのノーマライズ (0から1の範囲に)
        if np.max(green_intensity_profile) > 0:
            normalized_profile = green_intensity_profile / np.max(green_intensity_profile)
        else:
            normalized_profile = green_intensity_profile

        all_intensity_profiles.append(normalized_profile)
        detected_rects.append((x, y, w, h))

        # 検出された長方形を画像に描画 (白色で描画)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

    if not all_intensity_profiles:
        print(f"エラー: 有効な指定色({colors})の内側領域が見つかりませんでした。輪郭の面積閾値を調整してみてください。")
        return

    # -----------------------------------------------------------
    # プロット 1: 各指定領域のグラディエントを色分けで表示
    plt.figure(figsize=(12, 7))
    colors = plt.cm.jet(np.linspace(0, 1, len(all_intensity_profiles))) # 色のバリエーション
    
    for i, profile in enumerate(all_intensity_profiles):
        plt.plot(profile, color=colors[i], label=f'Region {i+1}')
        
    plt.title('Normalized Green Intensity Gradient for Each Detected Region')
    plt.xlabel('Horizontal Position (pixels within region)')
    plt.ylabel('Normalized Green Intensity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # プロット 2: 平均値のグラディエントを色分けで表示    
    # 全てのプロファイルを最も短い長さに調整
    target_length = min([len(p) for p in all_intensity_profiles])
    print(target_length)
    interpolated_profiles = interpolate_profiles(all_intensity_profiles, target_length=target_length)
    
    # 平均プロファイルを計算
    average_profile = np.mean(interpolated_profiles, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(average_profile, color='purple', label='Average Gradient') # 平均は単一色
    plt.title('Average Normalized Green Intensity Gradient')
    plt.xlabel('Horizontal Position (pixels)')
    plt.ylabel('Average Normalized Green Intensity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 検出された長方形が描画された画像を表示 (確認用)
    title = f"Detected Rectangles: {os.path.basename(image_path)}"
    show_debug_image(img, title)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_multiple_files(image_paths, colors=None):
    """
    複数の画像ファイルの平均プロファイルを一つのグラフに重ねて表示します。
    全画像の緑色輝度の最大値でノーマライズを行います。

    Args:
        image_paths (list): 画像ファイルのパスのリスト。
        colors (list): 検出する色のリスト（例: ['blue', 'yellow']）。Noneの場合は['blue']。
    """
    if colors is None:
        colors = ['blue']

    # 1. まず全画像から生のプロファイルを抽出し、グローバル最大値を計算
    all_raw_data = []  # [(file_name, raw_profiles, img), ...]
    global_max = 0

    for image_path in image_paths:
        raw_profiles, img = extract_raw_profiles(image_path, colors)
        if raw_profiles is not None:
            file_name = os.path.basename(image_path)
            all_raw_data.append((file_name, raw_profiles, img))
            # 各プロファイルの最大値を確認してグローバル最大値を更新
            for profile in raw_profiles:
                local_max = np.max(profile)
                if local_max > global_max:
                    global_max = local_max

    if not all_raw_data:
        print("エラー: 有効なプロファイルが取得できませんでした。")
        return

    print(f"グローバル最大値: {global_max}")

    # 2. グローバル最大値で全プロファイルをノーマライズし、各画像の平均プロファイルを計算
    all_average_profiles = []
    file_names = []
    debug_images = []

    for file_name, raw_profiles, img in all_raw_data:
        # グローバル最大値でノーマライズ
        normalized_profiles = [profile / global_max if global_max > 0 else profile for profile in raw_profiles]

        # 補間して平均を計算
        target_length = min([len(p) for p in normalized_profiles])
        interpolated = interpolate_profiles(normalized_profiles, target_length=target_length)
        avg_profile = np.mean(interpolated, axis=0)

        all_average_profiles.append(avg_profile)
        file_names.append(file_name)
        if img is not None:
            debug_images.append((img, file_name))

    # すべての平均プロファイルを同じ長さに補間
    target_length = min([len(p) for p in all_average_profiles])
    interpolated_profiles = interpolate_profiles(all_average_profiles, target_length=target_length)

    # 複数ファイルの平均プロファイルを重ねてプロット
    plt.figure(figsize=(12, 7))
    plot_colors = plt.cm.tab10(np.linspace(0, 1, len(interpolated_profiles)))

    for i, (profile, name) in enumerate(zip(interpolated_profiles, file_names)):
        plt.plot(profile, color=plot_colors[i], label=name)

    plt.title('Average Normalized Green Intensity Gradient (Global Normalization)')
    plt.xlabel('Horizontal Position (normalized)')
    plt.ylabel('Normalized Green Intensity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # デバッグ画像を順番に表示
    if debug_images:
        print(f"\n検出された長方形を順番に表示します（全{len(debug_images)}枚）")
        show_debug_images_sequential(debug_images)


# ハードコードされたデフォルトファイルリスト（引数なしの場合に使用）
DEFAULT_IMAGE_FILES = [
    './001_fz_Fz2_green_mask.tif',
    # './20250403_again_24%APF_Fz-GFP_fz-RNAi_sita_0063_green_grad_mask.tif',
    # './Fz2-GFP_1_ue_mask.tif',
]

# 個別プロットモード（True: 各ファイルを個別に, False: 重ねてプロット）
DEFAULT_SINGLE_MODE = False

# デフォルトの検出色（複数指定可）
# 対応色: blue, yellow, red, green, cyan, magenta
DEFAULT_COLORS = ['blue']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='画像内の指定色の長方形領域の緑色輝度プロファイルを抽出してプロットします。'
    )
    parser.add_argument(
        'image_files',
        nargs='*',
        help='処理する画像ファイルのパス（複数指定可、省略時はハードコードされたリストを使用）'
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help='各ファイルを個別にプロット（デフォルトは複数ファイルを重ねてプロット）'
    )
    parser.add_argument(
        '--colors', '-c',
        nargs='+',
        default=None,
        choices=list(COLOR_RANGES.keys()),
        help=f'検出する色（複数指定可）。対応色: {", ".join(COLOR_RANGES.keys())}。省略時はハードコード値を使用'
    )

    args = parser.parse_args()

    # 引数がなければハードコードされたリストを使用
    if args.image_files:
        image_files = args.image_files
        single_mode = args.single
    else:
        print("引数なし: ハードコードされたファイルリストを使用します")
        image_files = DEFAULT_IMAGE_FILES
        single_mode = DEFAULT_SINGLE_MODE

    # 色の設定
    if args.colors:
        colors = args.colors
    else:
        colors = DEFAULT_COLORS
    print(f"検出色: {colors}")

    if single_mode or len(image_files) == 1:
        # 個別プロットモード
        for image_file in image_files:
            print(f"Processing: {image_file}")
            plot_multi_region_green_intensity_profiles(image_file, colors)
    else:
        # 複数ファイル比較モード
        plot_multiple_files(image_files, colors)