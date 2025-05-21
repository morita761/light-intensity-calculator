import cv2

# 馬蹄形の認識確認できてない。picにあるやつでためした。
def is_horseshoe_like(cnt):
    if len(cnt) < 5:
        return False

    # 面積フィルタ
    area = cv2.contourArea(cnt)
    if area < 250 or area > 900:
        return False

    # 楕円近似
    ellipse = cv2.fitEllipse(cnt)

    # 楕円形との形状比較（誤差が大きければ不採用）
    ellipse_cnt = cv2.ellipse2Poly(
        (int(ellipse[0][0]), int(ellipse[0][1])),
        (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
        int(ellipse[2]), 0, 360, 5
    )
    similarity = cv2.matchShapes(cnt, ellipse_cnt, cv2.CONTOURS_MATCH_I1, 0.0)

    if similarity > 0.2:  # 0 に近いほど形が似ている
        return False

    # === 下が開いているか ===
    # 輪郭の重心計算
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return False
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # 凸包との凹み比較（開口部の有無）
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area

    if solidity > 0.9:
        return False  # 開いていない（ほぼ凸形状）

    # y座標の下部の点が少なければ開いていると判断（簡易法）
    bottom_part = [pt for pt in cnt if pt[0][1] > cy + 10]
    if len(bottom_part) < 0.15 * len(cnt):  # 全体の15%未満ならOK
        return False

    return True
