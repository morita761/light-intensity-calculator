# サンプルのメモ（vd_intensity.csv の手動確認用）

`ocr_to_vd_csv.py` が仮の値で出力する `individual_id` / `flipped` を直すときの情報。
CSV を作り直したら、ここに書いた内容で手動で直すこと。

## 左右反転（flipped）

ImageJ で crop したときに左右反転させた画像。`flipped` を `Yes` にする。

| 画像セット | flipped | 備考 |
|---|---|---|
| 241106_Fz_control003 | Yes | |
| 20260918_22h_mosaic_fz_Fz-GFP_GRM-splitGFP1-10_ue | Yes | 同じ個体の _sita に向きを合わせるため反転 |

上の表にない画像セットは反転していない（`No`）。

## 同一個体（individual_id）

同じ個体から撮った画像セット。`individual_id` を同じ値にする。

| 個体 | 画像セット |
|---|---|
| 20260918 mosaic fz Fz | 20260918_22h_mosaic_fz_Fz-GFP_GRM-splitGFP1-10_sita, 20260918_22h_mosaic_fz_Fz-GFP_GRM-splitGFP1-10_ue |

## スクショの撮り直し・追加

- 20250423_24%APF_Fz-GFP_RFP_ue: 8番のスクショが抜けていたので追加（追加分は画面配置が違い、20250428 と同じ範囲で読む）
- 20250403_again_24%APF_Fz-GFP_fz-RNAi_sita_0063: 11番のスクショが抜けていたので追加（追加分はパネルの位置が違う）
- 20250404_24%APF_Fz-GFP_loco-vang-Cas9P_001_ue: 34〜49番を追加測定（追加分はパネルの位置・文字サイズが違う）
