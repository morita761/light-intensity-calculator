import re
import csv
import os
import pandas as pd
import numpy as np # 中央値と平均値の計算に使用

# 解析パターン: (Index), (Left Ave), (Left Med), (Right Ave), (Right Med)
data_pattern = re.compile(r'^\s*(\d+)\s*,\s*([\d\.]+)\s*,\s*(\d+)\s*,\s*([\d\.]+)\s*,\s*(\d+)\s*,')

def _parse_single_file(input_file_path, experiment_name):
    """
    一つのテキストファイルを読み込み、データを抽出してDataFrame形式で返すヘルパー関数。
    データフレームには、画像固有のIDである 'title' (current_title) を含める。
    """
    all_data = [] 
    current_section = None 
    current_title = None 

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line_num, original_line in enumerate(f, 1):
                line = original_line.strip()
                
                if not line:
                    continue
                
                if line.startswith('-'):
                    current_title = line.lstrip('-').strip() # 画像ID/タイトル
                    current_section = None 
                    continue
                
                if line == 'V' or line == 'D':
                    current_section = line
                    continue

                if current_section and current_title:
                    match = data_pattern.match(line)
                    
                    if match:
                        _, left_ave, left_med, right_ave, right_med = match.groups()
                        
                        try:
                            left_ave_f = float(left_ave)
                            right_ave_f = float(right_ave)
                            # Med (中央値) はintとして抽出
                            left_med_i = int(left_med)
                            right_med_i = int(right_med)

                            # データ収集: [実験名, セクション, 画像タイトル, left_ave, right_ave, left_med, right_med]
                            all_data.append([
                                experiment_name, current_section, current_title, 
                                left_ave_f, right_ave_f, 
                                left_med_i, right_med_i
                            ])
                                
                        except ValueError as e:
                            print(f"⚠️ 警告: ファイル {input_file_path}, 行{line_num}の数値変換に失敗しました: {original_line.strip()} (エラー: {e})")
                            continue
                    
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {input_file_path}")
        return pd.DataFrame()

    # 'title' カラムを追加
    df = pd.DataFrame(all_data, columns=['experiment', 'section', 'title', 'left_ave', 'right_ave', 'left_med', 'right_med'])
    print(f"✅ ファイル {input_file_path} から {len(df)} 行のデータを読み込みました。")
    return df

def main_normalize_and_save(control_file, cas9_file):
    """
    2つのファイルを読み込み、画像ごとの med_avg でノーマライズし、
    4種類のCSVファイルに分割して出力するメイン関数。
    """
    
    # 1. データの読み込みと統合
    df_control = _parse_single_file(control_file, 'control')
    df_cas9 = _parse_single_file(cas9_file, 'cas9')
    
    df_combined = pd.concat([df_control, df_cas9], ignore_index=True)
    
    if df_combined.empty:
        print("処理するデータがありません。終了します。")
        return

    # 2. ノーマライズ定数 (Global Constant) の計算
    
    # 全ての left_med と right_med の値を一つの Series に結合
    all_med_values = pd.concat([df_combined['left_med'], df_combined['right_med']]).dropna()
    
    # Global Constant: 全データの中央値 (Median)
    grand_median = all_med_values.median()
    
    if grand_median == 0:
        print("致命的なエラー: 全データの中央値が0のため、ノーマライズできません。")
        return
    
    print(f"\n📏 グローバルノーマライズ定数 (Grand Median): {grand_median:.4f}")

    # 3. 画像ごとのノーマライズファクター (med_avg) の計算と適用
    
    # グループキーを 'experiment' + 'title' (画像ID) に変更
    df_combined['image_key'] = df_combined['experiment'] + '_' + df_combined['title']
    
    # ローカルファクター (group_med_avg) を計算: 各画像内の left_med と right_med の平均値
    image_med_avgs = {} 
    print("\n🔬 画像ごとのローカルノーマライズファクター (Image Med Average) の計算:")
    
    # 画像キーごとにループ
    for key in df_combined['image_key'].unique():
        df_image = df_combined[df_combined['image_key'] == key]
        # 当該画像に属する全ての left_med と right_med を結合し、その平均を計算
        # V/Dセクションに関わらず、画像内の全ての med を使う
        image_med_avg_value = pd.concat([df_image['left_med'], df_image['right_med']]).dropna().mean()
        image_med_avgs[key] = image_med_avg_value
        print(f"   - {key} の Image Med Average: {image_med_avg_value:.4f}")

    # ローカルファクターをDataFrameにマージ 
    # カラム名を 'image_med_avg' に変更
    factors_df = pd.DataFrame(image_med_avgs.items(), columns=['image_key', 'image_med_avg'])
    df_combined = pd.merge(df_combined, factors_df, on='image_key', how='left')

    # ノーマライズ計算
    # Normalized Ave = Raw Ave * (Grand Median / Image Med Average)
    
    # Image Med Averageが0でないことを確認
    if (df_combined['image_med_avg'] == 0).any(): 
        print("致命的なエラー: いずれかの画像のグループ中央値平均が0です。ノーマライズできません。")
        return
    
    # 正規化係数 (Scaling Factor) を計算
    # Scaling Factor = Grand Median / Image Med Average
    df_combined['scaling_factor'] = grand_median / df_combined['image_med_avg']
    
    df_combined['normalized_left'] = df_combined['left_ave'] * df_combined['scaling_factor']
    df_combined['normalized_right'] = df_combined['right_ave'] * df_combined['scaling_factor']
    
    print("\n🔄 ノーマライズの計算を実行しました。")
    print(f"   - 使用された式: Normalized Ave = Raw Ave * (Grand Median / Image Med Average)")
    
    # 4. 4つのファイルへの分割と出力 (V/DとExperimentで分割)
    
    output_files = {
        'control_v.csv': df_combined[(df_combined['experiment'] == 'control') & (df_combined['section'] == 'V')],
        'control_d.csv': df_combined[(df_combined['experiment'] == 'control') & (df_combined['section'] == 'D')],
        'cas9_v.csv': df_combined[(df_combined['experiment'] == 'cas9') & (df_combined['section'] == 'V')],
        'cas9_d.csv': df_combined[(df_combined['experiment'] == 'cas9') & (df_combined['section'] == 'D')],
    }
    
    # CSV出力関数 (空欄区切り, ヘッダーあり, 正規化値のみ)
    def write_normalized_csv(filename, df_output):
        try:
            # 必要なカラム (正規化された左右の値) のみを選択
            # デバッグのために 'image_key', 'image_med_avg', 'scaling_factor' を含めると便利ですが、
            # 最終出力はノーマライズ値のみとするため、元の要件に従います。
            df_to_save = df_output[['normalized_left', 'normalized_right']]
            
            # ヘッダーを所定の形式にリネーム
            df_to_save.columns = ["left_intensity", "right_intensity"]
            
            # スペース区切りでCSVとして保存 (index=Falseでインデックスを出力しない)
            df_to_save.to_csv(filename, sep=' ', index=False, float_format='%.6f')
            
            print(f"✅ ノーマライズデータ ({len(df_output)}行) を {filename} に出力しました。")
        except Exception as e:
            print(f"❌ {filename} の書き込み中にエラーが発生しました: {e}")

    # 各ファイルを出力
    print("\n--- CSVファイルの出力 ---")
    for filename, df_data in output_files.items():
        write_normalized_csv(filename, df_data)

    print("\n--- 全ての処理が完了しました ---")


# --- 実行部分 ---
CONTROL_FILE = 'control.txt' 
CAS9_FILE = 'loco_vang_cas9.txt' 

# 実行時、control.txtとloco_vang_cas9.txtのファイルを読み込みます。
main_normalize_and_save(CONTROL_FILE, CAS9_FILE)