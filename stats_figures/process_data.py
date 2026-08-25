import re
import csv
from collections import defaultdict
import os

def process_text_file_v5(input_file_path, v_output_file='v.csv', d_output_file='d.csv', v_norm_output_file='v_normalize.csv', d_norm_output_file='d_normalize.csv'):
    """
    テキストファイルを読み込み、V/Dで分け、CSVとして出力する。
    区分ごとの中央値の平均を計算し、それを使ってleft_aveとright_aveの両方をノーマライズする。
    ノーマライズ結果を空欄区切りのCSVとして出力する。
    """
    
    # --- 1. データ収集フェーズ ---
    
    # [left_ave, right_ave, left_med, right_med, title] の形式で全データを収集
    all_data = [] 
    
    # 区分ごとの中央値データを格納: {区分名: {'left_med': [..], 'right_med': [..]}}
    median_data_by_section = defaultdict(lambda: {'left_med': [], 'right_med': []})
    
    current_section = None 
    current_title = None
    
    data_pattern = re.compile(r'^\s*(\d+)\s*,\s*([\d\.]+)\s*,\s*(\d+)\s*,\s*([\d\.]+)\s*,\s*(\d+)\s*,')

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line_num, original_line in enumerate(f, 1):
                line = original_line.strip()
                
                if not line:
                    continue
                
                # 区分名 (タイトル) を検出
                if line.startswith('-'):
                    current_title = line.lstrip('-').strip()
                    current_section = None 
                    continue
                
                # セクションの切り替えを検出
                if line == 'V' or line == 'D':
                    current_section = line
                    continue

                # データ行の解析
                if current_section and current_title:
                    match = data_pattern.match(line)
                    
                    if match:
                        _, left_ave, left_med, right_ave, right_med = match.groups()
                        
                        try:
                            # float/int に変換
                            left_ave_f = float(left_ave)
                            right_ave_f = float(right_ave)
                            left_med_i = int(left_med)
                            right_med_i = int(right_med)

                            # 全データ収集
                            # 最後の二つはtitleとsection
                            all_data.append([left_ave_f, right_ave_f, left_med_i, right_med_i, current_title, current_section])
                                
                            # 中央値の平均計算用データを格納
                            median_data_by_section[current_title]['left_med'].append(left_med_i)
                            median_data_by_section[current_title]['right_med'].append(right_med_i)

                        except ValueError as e:
                            print(f"⚠️ 警告: 行{line_num}の数値変換に失敗しました: {original_line.strip()} (エラー: {e})")
                            continue

                    else:
                        # 特殊なデータ形式 (30/52など) や、解析できないデータ形式の場合
                        if '/' in line:
                             print(f"❌ エラー: 行{line_num}で特殊なデータ形式が見つかりました: {original_line.strip()}")
                             continue
                        continue
        
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {input_file_path}")
        return
        
    # --- 2. 統計値の計算フェーズ ---

    all_left_meds = [d[2] for d in all_data]
    all_right_meds = [d[3] for d in all_data]
    
    # 全体の中央値の平均を計算
    total_avg_left_med = sum(all_left_meds) / len(all_left_meds) if all_left_meds else 0
    total_avg_right_med = sum(all_right_meds) / len(all_right_meds) if all_right_meds else 0
    
    # ノーマライズに使用する定数項 (total_right_med_average + total_left_med_average)
    normalization_constant = total_avg_left_med + total_avg_right_med
    
    if normalization_constant == 0:
        print("致命的なエラー: 全体の中央値の平均の合計が0です。ノーマライズを実行できません。")
        return

    # 区分ごとの平均と合計
    group_med_sums = {}
    for title, data in median_data_by_section.items():
        left_meds = data['left_med']
        right_meds = data['right_med']
        
        avg_left_med = sum(left_meds) / len(left_meds) if left_meds else 0
        avg_right_med = sum(right_meds) / len(right_meds) if right_meds else 0
        
        # ノーマライズに使用する除算項 (right_med_average + left_med_average)
        group_med_sum = avg_left_med + avg_right_med
        group_med_sums[title] = group_med_sum

    # --- 3. CSVデータ生成とノーマライズ計算フェーズ ---
    
    v_norm_data = [] # [normalized_left_ave, normalized_right_ave]
    d_norm_data = [] # [normalized_left_ave, normalized_right_ave]
    v_csv_data = []  # [total_index, left_ave, right_ave]
    d_csv_data = []  # [total_index, left_ave, right_ave]
    
    # 全データを通して連番を作成するためのカウンター
    total_index_counter = 1 
    
    for left_ave_f, right_ave_f, _, _, title, section in all_data:
        # ノーマライズの除算項を取得
        group_med_sum = group_med_sums.get(title, 0)
        
        if group_med_sum == 0:
            print(f"警告: 区分 {title} の中央値の合計が0です。スキップします。")
            continue
            
        # --- ノーマライズ値の計算 ---
        # 計算式: ave / (group_med_sum) * normalization_constant
        
        # left_aveのノーマライズ
        normalized_left_ave = (left_ave_f / group_med_sum) * normalization_constant
        
        # right_aveのノーマライズ
        normalized_right_ave = (right_ave_f / group_med_sum) * normalization_constant
        
        # V/Dで分類
        if section == 'V':
            v_norm_data.append([normalized_left_ave, normalized_right_ave])
            v_csv_data.append([total_index_counter, left_ave_f, right_ave_f])
        elif section == 'D':
            d_norm_data.append([normalized_left_ave, normalized_right_ave])
            d_csv_data.append([total_index_counter, left_ave_f, right_ave_f])
            
        total_index_counter += 1

    # --- 4. CSVファイル出力フェーズ ---
    
    # 4-1. v_normalize.csv, d_normalize.csv の出力 (空欄区切り, ノーマライズ値)
    
    def write_normalized_csv(filename, data):
        try:
            # 区切り文字をスペース(' ')に指定
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                # ヘッダーの書き込み
                csvfile.write("left_intensity right_intensity\n")
                
                # データ行を出力 (空欄区切り)
                for row in data:
                    # 値をスペース区切りで書き込み
                    csvfile.write(f"{row[0]} {row[1]}\n") 
                    
            print(f"✅ ノーマライズデータ (両側) を {filename} に出力しました。({len(data)}行, 空欄区切り)")
        except Exception as e:
            print(f"❌ {filename} の書き込み中にエラーが発生しました: {e}")

    write_normalized_csv(v_norm_output_file, v_norm_data)
    write_normalized_csv(d_norm_output_file, d_norm_data)

    # 4-2. v.csv, d.csv の出力 (元の形式 - カンマ区切り, 元の平均値)
    
    def write_original_csv(filename, data):
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                csvfile.write("left_intensity right_intensity\n")

                # データ行を出力 (空欄区切り)
                for row in data:
                    # 少数点以下1桁まで出力する例: f"{row[0]:.1f} {row[1]:.1f}\n"
                    # ここではデフォルトの精度で出力します
                    csvfile.write(f"{row[1]:.3f} {row[2]:.3f}\n") 
            print(f"✅ 元データを {filename} に出力しました。({len(data)}行, カンマ区切り)")
        except Exception as e:
            print(f"❌ {filename} の書き込み中にエラーが発生しました: {e}")

    write_original_csv(v_output_file, v_csv_data)
    write_original_csv(d_output_file, d_csv_data)

    # --- 5. 統計情報出力フェーズ ---
    
    print("\n--- 統計情報：中央値 (Med) の平均 ---")
    print(f"【全体平均に使用される定数】: {normalization_constant:.2f} ({total_avg_left_med:.2f} + {total_avg_right_med:.2f})")
    
    for title, data in median_data_by_section.items():
        left_meds = data['left_med']
        right_meds = data['right_med']
        
        if not left_meds:
            continue
            
        avg_left_med = sum(left_meds) / len(left_meds)
        avg_right_med = sum(right_meds) / len(right_meds)
        
        print(f"【区分: {title}】")
        print(f"  > left_med の平均: {avg_left_med:.2f}")
        print(f"  > right_med の平均: {avg_right_med:.2f}")


# --- 実行部分 ---
input_file_name = 'input_test.txt' 

process_text_file_v5(input_file_name, 'v.csv', 'd.csv', 'v_normalize.csv', 'd_normalize.csv')