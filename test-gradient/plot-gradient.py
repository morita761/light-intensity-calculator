import cv2
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

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


def plot_multi_region_green_intensity_profiles(image_path):
    """
    画像内の複数の青い長方形領域の緑色輝度プロファイルを抽出し、正規化してプロットします。
    各領域のプロットと、それらの平均プロットの2種類を出力します。

    Args:
        image_path (str): 画像ファイルのパス。
    """
    # 1. 画像の読み込み
    img = cv2.imread(image_path)

    if img is None:
        print(f"エラー: 画像 '{image_path}' を読み込めませんでした。")
        return

    # 2. 青い長方形の識別
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 青色のHSV範囲を定義 (必要に応じて調整)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # マスクから輪郭を見つける
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("エラー: 画像内に青い長方形が見つかりませんでした。HSVの範囲を調整してみてください。")
        cv2.imshow("Blue Mask (for debug)", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 各青い長方形のプロファイルを格納するリスト
    all_intensity_profiles = []
    detected_rects = []
    
    # 検出されたすべての長方形を処理
    for i, contour in enumerate(contours):
        # 小さすぎる輪郭はノイズとみなしてスキップ
        if cv2.contourArea(contour) < 100: # 面積の閾値は画像に応じて調整
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        # 横の長さがほぼ同じであるという前提を活用（例: 許容誤差範囲内か確認）
        # 初めて検出した長方形の幅を基準にするか、または事前に設定した基準幅と比較する
        # ここでは、単純に検出されたすべての長方形のプロファイルを取得します
        # もし厳密な幅のチェックが必要なら、この部分に条件を追加してください
        
        # 3. 指定領域内の緑色輝度を抽出
        roi = img[y:y+h, x:x+w]
        _, g_channel, _ = cv2.split(roi)

        # 4. 緑色輝度プロファイルを作成 (各列の平均)
        green_intensity_profile = np.mean(g_channel, axis=0)
        
        # 5. プロファイルのノーマライズ (0から1の範囲に)
        # ゼロ除算を避けるため、最大値が0でないことを確認
        if np.max(green_intensity_profile) > 0:
            normalized_profile = green_intensity_profile / np.max(green_intensity_profile)
        else:
            normalized_profile = green_intensity_profile # 全て0の場合はそのまま
        
        all_intensity_profiles.append(normalized_profile)
        detected_rects.append((x, y, w, h))
        
        # 検出された長方形を画像に描画 (確認用)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2) # 黄色で描画

    if not all_intensity_profiles:
        print("エラー: 有効な青い長方形領域が見つかりませんでした。輪郭の面積閾値を調整してみてください。")
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
    cv2.imshow("Image with All Detected Blue Rectangles", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用例:
# ここにあなたの画像ファイルのパスを指定してください
# image_file = './20250403_again_24%APF_Fz-GFP_fz-RNAi_sita_0063_green_grad_mask.tif' # ここを実際の画像パスに変更してください
image_file = './001_fz_Fz2_green_mask.tif' # ここを実際の画像パスに変更してください
# image_file = './Fz2-GFP_1_ue_mask.tif'

plot_multi_region_green_intensity_profiles(image_file)