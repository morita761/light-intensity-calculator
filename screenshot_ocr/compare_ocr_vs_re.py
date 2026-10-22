"""
スクショから OCR で読んだ平均値（ocr_result-<NAME>.csv）と、Photoshop の計測ログ（<NAME>_grayvalue_RE.txt）の平均値を比較してプロットする。

対応のさせ方:
  計測ログは2行で1組（1行目 = ventral, 2行目 = dorsal）。ログの n 組目を <n>-V.png / <n>-D.png に対応させる
  （ラベルの「計測 番号」は削除した計測で飛ぶことがあるので使わず、並び順で対応させる）
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

# --- 設定（ハードコード） ---
NAME = '241106_Fz_control003'
OCR_CSV = 'ocr_result-' + NAME + '.csv'
RE_LOG = ('C:/Users/morit/OneDrive/デスクトップ/Fz-localization/25d_24%_Fz-GFP_control/analysis/'
          + NAME + '/' + NAME + '_grayvalue_RE.txt')
OUTPUT_PNG = 'compare_ocr_vs_re-' + NAME + '.png'
MEAN_LINE = 1                       # OCR 結果で平均値が書かれている行

COLORS = {'V': '#2a78d6', 'D': '#eb6834'}   # ventral: blue, dorsal: orange
MARKERS = {'V': 'o', 'D': 's'}
LABELS = {'V': 'Ventral', 'D': 'Dorsal'}


def load_re_log(path):
    """ Photoshop の計測ログ（UTF-16, タブ区切り）から平均値の列を返す """
    with open(path, 'r', encoding='utf-16') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    header, rows = rows[0], rows[1:]
    col = header.index('グレー値 (平均)')
    return [float(r[col]) for r in rows]


def load_ocr_mean(path):
    """ {'12-V.png': 平均値, ...} を返す """
    with open(path, 'r', encoding='utf-8-sig') as f:
        return {r['image']: float(r['text']) for r in csv.DictReader(f) if int(r['line']) == MEAN_LINE}


def main():
    re_mean = load_re_log(RE_LOG)
    ocr_mean = load_ocr_mean(OCR_CSV)

    n_pairs = len(re_mean) // 2
    if len(re_mean) % 2 or n_pairs * 2 != len(ocr_mean):
        print(f"⚠️ 警告: 件数が合いません（計測ログ {len(re_mean)}件, OCR {len(ocr_mean)}枚）")

    # branch ごとに (測定番号, OCR, ログ) を集める
    data = {'V': [], 'D': []}
    for i, value in enumerate(re_mean):
        no, branch = i // 2 + 1, 'VD'[i % 2]
        image = f'{no}-{branch}.png'
        if image not in ocr_mean:
            print(f"⚠️ 警告: {image} の OCR 結果がありません")
            continue
        data[branch].append((no, ocr_mean[image], value))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # 左: 散布図（OCR vs ログ）。y=x 上に乗れば一致
    all_values = [v for b in data.values() for _, o, r in b for v in (o, r)]
    lim = (min(all_values) - 3, max(all_values) + 3)
    ax1.plot(lim, lim, color='#9a9a94', linestyle='--', linewidth=1, label='y = x', zorder=1)
    for branch, items in data.items():
        _, ocr, re_ = zip(*items)
        ax1.scatter(re_, ocr, s=40, marker=MARKERS[branch], color=COLORS[branch],
                    edgecolor='white', linewidth=1, label=LABELS[branch], zorder=2)
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Photoshop log mean (grayvalue_RE)', fontsize=12)
    ax1.set_ylabel('Screenshot OCR mean', fontsize=12)
    ax1.set_title('OCR vs log', fontsize=14)
    ax1.legend(frameon=False)

    # 右: 測定番号ごとの差（OCR − ログ）
    print(f"{NAME}: OCR − ログ の差")
    for branch, items in data.items():
        no, ocr, re_ = map(np.array, zip(*items))
        diff = ocr - re_
        ax2.plot(no, diff, marker=MARKERS[branch], markersize=7, linewidth=2, color=COLORS[branch],
                 markeredgecolor='white', label=f'{LABELS[branch]} (mean {diff.mean():+.2f})')
        print(f"  {LABELS[branch]}: 平均 {diff.mean():+.2f}, SD {diff.std(ddof=1):.2f}, "
              f"範囲 {diff.min():+.2f} 〜 {diff.max():+.2f}, r = {np.corrcoef(ocr, re_)[0, 1]:.3f}")
        # 他から大きく外れた点に番号を付ける（平均から 3SD 以上）
        for n, d in zip(no, diff):
            if abs(d - diff.mean()) > 3 * diff.std(ddof=1):
                ax2.annotate(f'{n}-{branch}', (n, d), textcoords='offset points', xytext=(6, 6), fontsize=10)
                print(f"    ⚠️ 外れ値: {n}-{branch} (差 {d:+.2f})")
    ax2.axhline(0, color='#9a9a94', linewidth=1)
    ax2.set_xlabel('Measurement number', fontsize=12)
    ax2.set_ylabel('OCR − log (mean intensity)', fontsize=12)
    ax2.set_title('Difference per measurement', fontsize=14)
    ax2.legend(frameon=False)

    for ax in (ax1, ax2):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(color='#e6e5e0', linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle(NAME, fontsize=15)
    fig.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"✅ {OUTPUT_PNG} に保存しました。")
    plt.show()


# --- 実行部分 ---
main()
