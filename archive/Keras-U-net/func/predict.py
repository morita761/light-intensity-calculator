import cv2
import numpy as np
import func.padImage as pI

def predict_on_test_image(test_image_path, model, patch_size=256, stride=128, ave_map_threshold=0.5):
    img = cv2.imread(test_image_path)
    img = pI.pad_image_to_multiple(img, patch_size)
    h, w = img.shape[:2]

    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    patch_id = 0
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            patch_input = patch[None, ...] / 255.0
            pred = model.predict(patch_input, verbose=0)[0, ..., 0]
            # ~0.0なら予想が背景に偏っている
            print(f"Epoch {patch_id}: pred.mean() = {np.mean(pred)}")
            patch_id += 1

            prob_map[y:y+patch_size, x:x+patch_size] += pred
            count_map[y:y+patch_size, x:x+patch_size] += 1.0

            # 各パッチヒートマップ保存
            heatmap = (pred * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # cv2.imwrite(f"debug_heatmap_{patch_id}.png", heatmap_color)

    # 予想値agv_map
    avg_map = np.divide(prob_map, count_map, out=np.zeros_like(prob_map), where=count_map!=0)
    result_mask = (avg_map > ave_map_threshold).astype(np.uint8) * 255 # 白色で出力
    print("result mask size")
    print(result_mask.shape)

    # 最終ヒートマップ・マスク保存
    print("avg_map mean:", np.mean(avg_map))
    print("avg_map max:", np.max(avg_map))
    heatmap = (avg_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite("avg_heatmap.png", heatmap_color)

    cv2.namedWindow('Sample Image', cv2.WINDOW_NORMAL)
    # 読み込んだ画像の高さと幅を取得，整数で割り算
    height = img.shape[0] //2
    width = img.shape[1] //2
    cv2.resizeWindow('Sample Image', width, height)
    cv2.imshow("Sample Image", img)
    cv2.waitKey(3*1000)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    # cv2.namedWindow('Sample Image', cv2.WINDOW_NORMAL)
    # 読み込んだ画像の高さと幅を取得，整数で割り算
    height = img.shape[0] //2
    width = img.shape[1] //2
    # cv2.resizeWindow('Sample Image', width, height)
    cv2.imshow("Sample Image", result_mask[:624, :1254])
    cv2.waitKey(3*1000)
    # cv2.waitKey(0)

    cv2.destroyAllWindows()


    return result_mask  # オリジナルサイズに戻す


# def predict_on_test_image(test_image_path, model, patch_size=256, stride=128):
#     img = cv2.imread(test_image_path)
#     h, w = img.shape[:2]
#     output_mask = np.zeros((h, w))

#     for y in range(0, h - patch_size + 1, stride):
#         for x in range(0, w - patch_size + 1, stride):
#             patch = img[y:y+patch_size, x:x+patch_size]
#             patch_input = patch[None, ...] / 255.0
#             pred = model.predict(patch_input)[0, ..., 0]
#             output_mask[y:y+patch_size, x:x+patch_size] += pred

#     # 正規化して二値化
#     output_mask = (output_mask > 0.5).astype(np.uint8) * 255
#     return output_mask
