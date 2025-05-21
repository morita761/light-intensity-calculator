import cv2

## 2025/05/22 1:44 使用
def matches_any_template(cnt, template_list, threshold=0.15):
    for template in template_list:
        area = cv2.contourArea(cnt)
        if 300 < area < 1000:  # 馬蹄形のサイズ範囲を想定（要調整）
            if len(cnt) < 5 or len(template) < 5:
                continue
            score = cv2.matchShapes(cnt, template, cv2.CONTOURS_MATCH_I3, 0.0)
            if score < threshold:
                return True
    return False
