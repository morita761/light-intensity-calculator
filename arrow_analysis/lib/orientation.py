"""馬蹄形の向き（凹＝開口部の方向）の推定と、画像セットからの角度収集

角度の定義: 画像の下 = 0°、右 = 90°、左 = -90°、上 = ±180°

method:
  skeleton : 馬蹄形を骨格化して両端点の中点を開口部とし、凸包の重心 → 端点中点 の向き
             （旧 detect_arrow_improved.py / group_comparison_ttest.py）
  moment   : 輪郭の重心 → バウンディングボックス中心 の向き
             （旧 detect_arrow.py / orientation-polar-histogram.py）
"""
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.morphology import skeletonize

from shared.horseshoe import (DEFAULT_COLORS, classify_side, dilate_to_occupancy, filled_local_mask,
                              image_center_x, imread, iter_horseshoe_contours)

SIDES = ('left', 'right')
SIDE_COLORS = {'left': (255, 0, 255), 'right': (255, 0, 0)}  # マゼンタ / 青 (BGR)
METHODS = ('skeleton', 'moment')


def _to_orientation(angle_deg_standard):
    """標準の角度（右=0°, 上=90°）→ 下=0°, 右=90° に変換して [-180, 180] に収める"""
    angle = angle_deg_standard + 90
    if angle > 180:
        angle -= 360
    elif angle <= -180:
        angle += 360
    return angle


# ---------------------------------------------------------------------------
# skeleton 法
# ---------------------------------------------------------------------------

def find_skeleton_endpoints_and_branches(skeleton):
    """骨格画像から端点（近傍1）と分岐点（近傍3以上）を検出する"""
    endpoints, branch_points = [], []
    rows, cols = skeleton.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j]:
                neighbors = skeleton[i - 1:i + 2, j - 1:j + 2].sum() - 1
                if neighbors == 1:
                    endpoints.append((j, i))
                elif neighbors >= 3:
                    branch_points.append((j, i))
    return endpoints, branch_points


def bfs_distance(skeleton, start, end):
    """骨格上で start から end までのBFS最短経路長（到達できなければ -1）"""
    rows, cols = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)
    queue = deque([(start[0], start[1], 0)])
    visited[start[1], start[0]] = True
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while queue:
        x, y, dist = queue.popleft()
        if (x, y) == end:
            return dist
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and skeleton[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny, dist + 1))
    return -1


def find_longest_path_endpoints(skeleton, endpoints):
    """骨格上で最長パスになる端点ペア"""
    if len(endpoints) < 2:
        return None
    max_dist = -1
    best_pair = (endpoints[0], endpoints[1])
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            dist = bfs_distance(skeleton, endpoints[i], endpoints[j])
            if dist > max_dist:
                max_dist = dist
                best_pair = (endpoints[i], endpoints[j])
    return best_pair, max_dist


def skeleton_orientation(contour, debug=False):
    """返り値: (角度, 始点x, 始点y, 先端x, 先端y) または None。debug=True なら (結果, debug_info)"""
    def done(result, info=None):
        return (result, info) if debug else result

    if len(contour) < 5:
        return done(None)

    x, y, w, h = cv2.boundingRect(contour)
    padding = 5
    mask = np.zeros((h + 2 * padding, w + 2 * padding), dtype=np.uint8)
    cv2.drawContours(mask, [contour - np.array([x - padding, y - padding])], -1, 255, cv2.FILLED)

    skeleton = skeletonize(mask > 0)
    endpoints, branch_points = find_skeleton_endpoints_and_branches(skeleton)

    debug_info = None
    if debug:
        vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        vis[skeleton] = (255, 255, 255)
        for ep in endpoints:
            cv2.circle(vis, ep, 3, (0, 255, 0), -1)
        for bp in branch_points:
            cv2.circle(vis, bp, 3, (0, 0, 255), -1)
        debug_info = {'skeleton_vis': vis, 'num_endpoints': len(endpoints),
                      'num_branch_points': len(branch_points)}

    if len(endpoints) < 2:
        if debug_info is not None:
            debug_info['selection_method'] = 'fallback'
        return done(moment_orientation(contour), debug_info)

    # 端点2つ・分岐なしならそのまま、それ以外は骨格上の最長パスの両端を使う
    if len(endpoints) == 2 and not branch_points:
        ep1, ep2 = endpoints
        method = 'direct'
    else:
        (ep1, ep2), path_dist = find_longest_path_endpoints(skeleton, endpoints)
        method = f'longest_path(dist={path_dist})'
    if debug_info is not None:
        debug_info['selected_endpoints'] = (ep1, ep2)
        debug_info['selection_method'] = method

    # 端点の中点（開口部の中心）
    midpoint_x = (ep1[0] + ep2[0]) / 2 + x - padding
    midpoint_y = (ep1[1] + ep2[1]) / 2 + y - padding

    # 凸包の重心 → 端点中点 が開口部の向き
    m = cv2.moments(cv2.convexHull(contour, returnPoints=True))
    if m['m00'] == 0:
        return done(None, debug_info)
    hull_cx = int(m['m10'] / m['m00'])
    hull_cy = int(m['m01'] / m['m00'])
    dx = midpoint_x - hull_cx
    dy = midpoint_y - hull_cy
    norm = np.sqrt(dx ** 2 + dy ** 2)
    if norm < 1e-6:
        return done(None, debug_info)

    cx, cy = int(midpoint_x), int(midpoint_y)
    arrow_length = max(w, h) * 0.6
    fx = int(cx + dx / norm * arrow_length)
    fy = int(cy + dy / norm * arrow_length)
    angle = _to_orientation(np.degrees(np.arctan2(-dy, dx)))
    return done((angle, cx, cy, fx, fy), debug_info)


# ---------------------------------------------------------------------------
# moment 法
# ---------------------------------------------------------------------------

def moment_orientation(contour, arrow_length_ratio=0.8):
    """輪郭の重心 C → バウンディングボックス中心 B の向き"""
    m = cv2.moments(contour)
    if m['m00'] == 0:
        return None
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    x, y, w, h = cv2.boundingRect(contour)
    bx, by = x + w // 2, y + h // 2
    angle = _to_orientation(np.degrees(np.arctan2(cy - by, bx - cx)))

    dx, dy = bx - cx, by - cy
    norm = np.sqrt(dx ** 2 + dy ** 2)
    if norm < 1e-6:
        return None
    arrow_length = max(w, h) * arrow_length_ratio
    return angle, cx, cy, int(cx + dx / norm * arrow_length), int(cy + dy / norm * arrow_length)


def legacy_moment_orientation(contour):
    """旧 detect_arrow.py の moment 法（先端 = バウンディングボックス中心）"""
    if len(contour) < 3:
        return None
    m = cv2.moments(contour)
    if m['m00'] == 0:
        return None
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    x, y, w, h = cv2.boundingRect(contour)
    bx, by = x + w // 2, y + h // 2
    return _to_orientation(np.degrees(np.arctan2(cy - by, bx - cx))), cx, cy, bx, by


def calculate_orientation(contour, method='skeleton', debug=False):
    if method == 'skeleton':
        return skeleton_orientation(contour, debug=debug)
    result = legacy_moment_orientation(contour)
    return (result, None) if debug else result


# ---------------------------------------------------------------------------
# 画像セットからの角度収集
# ---------------------------------------------------------------------------

@dataclass
class OrientationParams:
    method: str = 'skeleton'
    center_margin: int = 80
    min_area: int = 20
    dilate_occupancy: float = 0.3   # アノテーションの塗りの膨張（角度には影響しない）
    max_dilate: int = 2
    arrow_scale: float = 2.5        # アノテーションの矢印の長さ倍率
    colors: list = field(default_factory=lambda: list(DEFAULT_COLORS))

    @classmethod
    def from_settings(cls, s, **overrides):
        params = cls(
            method=s.str('method', cls.method),
            center_margin=s.int('center_margin', cls.center_margin),
            min_area=s.int('min_area', cls.min_area),
            dilate_occupancy=s.float('dilate_occupancy', cls.dilate_occupancy),
            max_dilate=s.int('max_dilate', cls.max_dilate),
            arrow_scale=s.float('arrow_scale', cls.arrow_scale),
            colors=s.list('annotation_colors', list(DEFAULT_COLORS)),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(params, key, value)
        return params


def _draw_arrow(img, cx, cy, fx, fy, scale, color, thickness=2):
    fx_ext = int(cx + (fx - cx) * scale)
    fy_ext = int(cy + (fy - cy) * scale)
    cv2.arrowedLine(img, (cx, cy), (fx_ext, fy_ext), color, thickness, tipLength=0.2)


def _roi_debug(original, contour, angle, cx, cy, fx, fy, scale, side, size=(150, 150)):
    """1つの馬蹄形の拡大画像に矢印を描いたもの"""
    x, y, w, h = cv2.boundingRect(contour)
    roi = cv2.resize(original[y:y + h, x:x + w].copy(), size, interpolation=cv2.INTER_LINEAR)
    sx, sy = size[0] / w, size[1] / h
    c = (int((cx - x) * sx), int((cy - y) * sy))
    f = (int((fx - x) * sx), int((fy - y) * sy))
    _draw_arrow(roi, *c, *f, scale, (0, 255, 255))
    return {'image': cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), 'angle': angle, 'side': side}


def _skeleton_debug(debug_info, angle, side, size=(150, 150)):
    vis = debug_info['skeleton_vis'].copy()
    if 'selected_endpoints' in debug_info:
        for ep in debug_info['selected_endpoints']:
            cv2.circle(vis, ep, 5, (255, 255, 0), 2)
    vis = cv2.resize(vis, size, interpolation=cv2.INTER_NEAREST)
    method = debug_info.get('selection_method', '')
    short = 'LP' if 'longest_path' in method else ('D' if method == 'direct' else 'FB')
    cv2.putText(vis, f"EP:{debug_info['num_endpoints']} BR:{debug_info['num_branch_points']}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(vis, short, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    return {'image': cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), 'angle': angle, 'side': side, 'method': method}


def collect_angles(pairs, params, annotate=True, debug_rois=False, debug_skeleton=False):
    """画像セットの全馬蹄形の向きを左右別に集める

    返り値: dict(angles={'left': [...], 'right': [...]}, annotated=[...], rois=[...], skeletons=[...])
    """
    result = {'angles': {side: [] for side in SIDES}, 'annotated': [], 'rois': [], 'skeletons': []}
    for pair in pairs:
        original = imread(pair['original'])
        mask_img = imread(pair['mask'])
        if original is None or mask_img is None:
            print(f"  画像を読み込めませんでした（スキップ）: {pair['original'].name}")
            continue

        center_x = image_center_x(mask_img)
        debug_img = None
        if annotate:
            debug_img = original.copy()
            cv2.line(debug_img, (center_x, 0), (center_x, debug_img.shape[0]), (255, 255, 255), 1)

        counts = {side: 0 for side in SIDES}
        for contour in iter_horseshoe_contours(mask_img, params.colors, params.min_area):
            side = classify_side(contour, center_x, params.center_margin)
            if side is None:
                continue
            want_skeleton = debug_skeleton and params.method == 'skeleton'
            orientation = calculate_orientation(contour, params.method, debug=want_skeleton)
            info = None
            if want_skeleton:
                orientation, info = orientation

            if orientation:
                angle, cx, cy, fx, fy = orientation
                result['angles'][side].append(angle)
                counts[side] += 1
                if debug_img is not None:
                    cv2.circle(debug_img, (cx, cy), 3, (255, 255, 255), -1)
                    cv2.circle(debug_img, (fx, fy), 3, (0, 255, 0), -1)
                    _draw_arrow(debug_img, cx, cy, fx, fy, params.arrow_scale, (255, 255, 0))
                if debug_rois:
                    result['rois'].append(_roi_debug(original, contour, angle, cx, cy, fx, fy,
                                                     params.arrow_scale, side))
                if info is not None:
                    result['skeletons'].append(_skeleton_debug(info, angle, side))

            # アノテーションの塗り（膨張しすぎる馬蹄形は塗らない。角度には影響しない）
            if debug_img is not None:
                local_mask, (x, y, _, _) = filled_local_mask(contour)
                if params.dilate_occupancy:
                    local_mask, iterations = dilate_to_occupancy(local_mask, params.dilate_occupancy)
                    if iterations > params.max_dilate:
                        continue
                final_contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if final_contours:
                    cv2.drawContours(debug_img, [final_contours[0] + np.array([x, y])], -1,
                                     SIDE_COLORS[side], -1)

        if debug_img is not None:
            result['annotated'].append(debug_img)
        print(f"  {pair['original'].name}: Ventral {counts['left']} 個, Dorsal {counts['right']} 個")
    return result
