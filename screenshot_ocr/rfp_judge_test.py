"""
【テスト】番号ごとに、対応する馬蹄形の輪郭の色から RFP（赤っぽい色）かどうかを判定する。

前提:
  馬蹄形は色付きの線で囲まれていて、マゼンタ（赤っぽい色）の線 = RFP、水色・青の線 = RFP ではない、とする。
  （写真そのものの赤色蛍光を測るのではなく、手で描いた輪郭の色で判定する）

処理の流れ:
  1. extract_label_coords.py と同じ方法で番号の座標を取る（label_coords CSV があればそれを使う）
  2. HSV の色相で輪郭の画素を COLOR_CLASSES の色ごとに分け、色ごとに連結成分 = 馬蹄形1個 とする
     番号の白文字が輪郭に重なると線が切れるので、白文字でつながっている小さな切れ端は同じ馬蹄形にまとめる
     （大きな輪郭どうしは白文字でつながっていても別の馬蹄形のままにする）
  3. 番号と馬蹄形を1対1で対応させる（番号の中心から輪郭までの距離の合計が最小になる組み合わせ）
  4. 対応した馬蹄形の色が RFP_COLORS なら RFP = Yes
"""
import os
import csv
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from extract_label_coords import extract_label_coords, load_label_coords, WHITE_THRESHOLD

# --- 設定（ハードコード） ---
INPUT_IMAGE = 'label_image.png'          # 番号と馬蹄形の輪郭が描かれた画像
LABEL_COORDS_CSV = 'label_coords.csv'    # 番号の座標（無ければ INPUT_IMAGE から作る）
OUTPUT_CSV = 'rfp_judge.csv'
OUTPUT_IMAGE = 'rfp_judge_check.png'     # 確認用画像（番号と対応した馬蹄形を線で結ぶ）

# 輪郭の色の分類（OpenCV の HSV: 色相 H は 0〜179）
# 添付画像の実測: 緑の蛍光 45〜60, 水色 85〜104, 青 108〜122, マゼンタ 155〜179 (+赤寄り 0〜10)
COLOR_CLASSES = {
    'magenta': [(150, 179), (0, 10)],
    'cyan': [(85, 104)],
    'blue': [(105, 125)],
}
RFP_COLORS = ['magenta']                 # RFP とみなす輪郭の色
MIN_SATURATION = 80                      # 彩度・明度がこれより低い画素は輪郭とみなさない（背景の緑の蛍光などを除く）
MIN_VALUE = 60
MIN_PIECE_AREA = 50                      # これより小さい色の塊はノイズとして捨てる（px）
FULL_OUTLINE_AREA = 600                  # これ以上の塊は切れていない馬蹄形1個とみなす（添付画像では1個 約800〜1350px）
MIN_OUTLINE_AREA = 300                   # 切れ端をまとめてもこれに満たないものは馬蹄形とみなさない（px）
MAX_DISTANCE = 150                       # 番号の中心から輪郭までがこれより遠い対応は信用せず警告する（px）


def find_outlines(img):
    """
    輪郭の色ごとに連結成分を取り、[{'color': 色名, 'mask': bool配列, 'center': (x, y)}, ...] を返す
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    colored = (s > MIN_SATURATION) & (v > MIN_VALUE)
    white = img.min(axis=2) > WHITE_THRESHOLD

    outlines = []
    for color, ranges in COLOR_CLASSES.items():
        mask = np.zeros(h.shape, bool)
        for lo, hi in ranges:
            mask |= colored & (h >= lo) & (h <= hi)

        # 色の画素だけの塊（pieces）と、白文字も含めてつながっている範囲（bridge）
        n, piece_labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        _, bridge_labels = cv2.connectedComponents((mask | white).astype(np.uint8), connectivity=8)

        # bridge ごとに、大きな塊はそれぞれ馬蹄形1個、小さな切れ端は一番近い大きな塊に足す
        # 大きな塊がない bridge は、切れ端をまとめて馬蹄形1個にする（文字で2つに切れた輪郭など）
        groups = {}
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < MIN_PIECE_AREA:
                continue
            ys, xs = np.nonzero(piece_labels == i)
            groups.setdefault(bridge_labels[ys[0], xs[0]], []).append(i)

        for pieces in groups.values():
            big = [i for i in pieces if stats[i, cv2.CC_STAT_AREA] >= FULL_OUTLINE_AREA]
            small = [i for i in pieces if i not in big]
            members = {i: [i] for i in big} if big else {small[0]: small}
            if big:
                for i in small:
                    nearest = min(big, key=lambda b: np.hypot(*(centroids[i] - centroids[b])))
                    members[nearest].append(i)
            for ids in members.values():
                component = np.isin(piece_labels, ids)
                if component.sum() >= MIN_OUTLINE_AREA:
                    ys, xs = np.nonzero(component)
                    outlines.append({'color': color, 'mask': component, 'center': (xs.mean(), ys.mean())})
    return outlines


def main():
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"エラー: 画像が読み込めません: {INPUT_IMAGE}")
        return

    if not os.path.exists(LABEL_COORDS_CSV):
        if not extract_label_coords(INPUT_IMAGE, LABEL_COORDS_CSV, 'label_coords_check.png'):
            return
    coords = load_label_coords(LABEL_COORDS_CSV)
    numbers = sorted(coords)

    outlines = find_outlines(img)
    counts = {c: sum(1 for o in outlines if o['color'] == c) for c in COLOR_CLASSES}
    print(f"🔍 番号 {len(numbers)}個, 馬蹄形の輪郭 {len(outlines)}個 ("
          + ', '.join(f'{c} {n}' for c, n in counts.items()) + ")")
    if len(outlines) != len(numbers):
        print("⚠️ 警告: 番号と輪郭の数が合いません。輪郭がくっついている・切れている可能性があります（確認用画像を見ること）")

    # 番号の中心から各輪郭の一番近い画素までの距離（距離変換で求める）
    cost = np.zeros((len(numbers), len(outlines)))
    for j, o in enumerate(outlines):
        dist = cv2.distanceTransform((~o['mask']).astype(np.uint8), cv2.DIST_L2, 5)
        for i, no in enumerate(numbers):
            x, y = coords[no]
            cost[i, j] = dist[int(round(y)), int(round(x))]

    # 番号と輪郭を1対1で対応させる（距離の合計が最小）
    rows_idx, cols_idx = linear_sum_assignment(cost)
    match = dict(zip(rows_idx, cols_idx))

    results = []
    for i, no in enumerate(numbers):
        x, y = coords[no]
        if i not in match:
            print(f"⚠️ 警告: 番号 {no} に対応する輪郭がありません")
            results.append([no, f'{x:.1f}', f'{y:.1f}', '', '', '', ''])
            continue
        o = outlines[match[i]]
        d = cost[i, match[i]]
        if d > MAX_DISTANCE:
            print(f"⚠️ 警告: 番号 {no} と対応した輪郭が遠すぎます（{d:.0f}px）")
        rfp = 'Yes' if o['color'] in RFP_COLORS else 'No'
        results.append([no, f'{x:.1f}', f'{y:.1f}', o['color'], rfp, f"{o['center'][0]:.1f}", f"{o['center'][1]:.1f}"])

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['number', 'label_x', 'label_y', 'outline_color', 'rfp', 'outline_x', 'outline_y'])
        writer.writerows(results)

    rfp_numbers = [r[0] for r in results if r[4] == 'Yes']
    print(f"✅ {OUTPUT_CSV} に出力しました。RFP = Yes: {len(rfp_numbers)}個 {rfp_numbers}")

    # 確認用画像: 番号 → 対応した輪郭の中心を線で結び、RFP は赤、それ以外は白で番号を描く
    check = img.copy()
    for r in results:
        if not r[3]:
            continue
        p1 = (int(float(r[1])), int(float(r[2])))
        p2 = (int(float(r[5])), int(float(r[6])))
        color = (0, 0, 255) if r[4] == 'Yes' else (255, 255, 255)
        cv2.line(check, p1, p2, color, 2)
        cv2.circle(check, p2, 5, color, -1)
        cv2.putText(check, 'RFP' if r[4] == 'Yes' else '-', (p1[0] + 25, p1[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(OUTPUT_IMAGE, check)
    print(f"   確認用画像を {OUTPUT_IMAGE} に保存しました。")


# --- 実行部分 ---
main()
