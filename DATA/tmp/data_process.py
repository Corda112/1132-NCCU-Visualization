import pandas as pd
import numpy as np
import json
import glob
import re
import emoji
import time
import os

# --- NLP & ML Libraries ---
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- Utility to handle potential warnings ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# STEP 1: 數據讀取與基礎處理
# ==============================================================================
def load_and_merge_data(directory="."):
    """
    載入指定資料夾裡的所有 JSON 檔案並合併成一個 DataFrame。
    """
    path_pattern = os.path.join(directory, "*.json")
    json_files = glob.glob(path_pattern)
    if not json_files:
        print(f"錯誤：在資料夾 '{directory}' 中找不到任何 JSON 檔案。")
        return None
        
    print(f"找到 {len(json_files)} 個檔案: {json_files}")
    
    df_list = [pd.read_json(f) for f in json_files]
    df = pd.concat(df_list, ignore_index=True)
    
    # 移除重複的推文 (以 id 為準)
    df = df.drop_duplicates(subset='id').reset_index(drop=True)
    
    # 將 'createdAt' 轉換為 datetime 物件以便後續處理
    df['createdAt'] = pd.to_datetime(df['createdAt'])
    
    print(f"成功載入並合併資料，共 {len(df)} 篇不重複的貼文。")
    return df

# ==============================================================================
# STEP 2: 資料清洗
# ==============================================================================
def clean_text(text):
    """
    根據 TextVista 論文描述，對單篇貼文進行深度清洗。
    """
    # 1. 移除網址
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 2. 移除表情符號
    text = emoji.demojize(text)
    text = re.sub(r':[a-zA-Z_]+:', ' ', text)
    # 3. 移除使用者標註 (@mentions)
    text = re.sub(r'@\w+', '', text)
    # 4. 移除 Hashtag 的 '#' 符號，但保留文字
    text = text.replace('#', '')
    # 5. 移除股票代碼 '$' 符號
    text = text.replace('$', '')
    # 6. 移除換行符與多餘的空格
    text = re.sub(r'\s+', ' ', text).strip()
    # 7. 統一轉為小寫
    text = text.lower()
    
    return text

# ==============================================================================
# STEP 3: 產生輸出檔案
# ==============================================================================

def generate_frequency_file(df, output_filename="term_ngram_frequency.json"):
    """
    檔案1: 計算每日的 Term 和 N-gram 頻率。
    """
    print("\n--- 開始產生詞彙頻率檔案 (檔案 1/3) ---")
    df['date'] = df['createdAt'].dt.date.astype(str)
    
    # 使用 CountVectorizer 高效率地計算 Term 和 N-gram
    # ngram_range=(1, 3) 表示計算單詞、2-gram、3-gram
    # min_df=5 表示一個詞或詞組至少要在5篇文件中出現過才計算，過濾雜訊
    vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english', min_df=5)
    
    # 擬合整個語料庫以確定詞彙表
    vectorizer.fit(df['cleaned_text'])
    
    daily_freq = {}
    
    # 按日期分組計算頻率
    for date, group in df.groupby('date'):
        print(f"正在處理 {date} 的數據...")
        # 只轉換當天的文本
        X = vectorizer.transform(group['cleaned_text'])
        # 加總得到當天所有詞彙的頻率
        freqs = X.sum(axis=0).A1
        # 建立詞彙到頻率的映射
        term_freqs = {term: int(freq) for term, freq in zip(vectorizer.get_feature_names_out(), freqs) if freq > 0}
        daily_freq[date] = term_freqs
        
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(daily_freq, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 詞彙頻率檔案 '{output_filename}' 已成功建立。")

def generate_semantic_file(df, output_filename="semantic_clustering_sentiment.json"):
    """
    檔案2: 進行向量化、降維、分群與情緒分析。
    """
    print("\n--- 開始產生語意分析檔案 (檔案 2/3) ---")
    
    df_sample = df.copy()
    print(f"將對 {len(df_sample)} 篇文章進行語意分析...")

    # A. 文本向量化 (SBERT)
    print("步驟 1/4: 正在使用 SBERT 產生文本向量 (此步驟可能需要較長時間)...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sbert_model.encode(df_sample['cleaned_text'].tolist(), show_progress_bar=True)
    
    # B. 降維 (PCA)
    print("步驟 2/4: 正在使用 PCA 進行降維...")
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(embeddings)
    df_sample['x'] = coordinates[:, 0]
    df_sample['y'] = coordinates[:, 1]
    
    # C. 分群 (HAC)
    n_clusters = 20
    print(f"步驟 3/4: 正在使用 HAC 進行分群 (分成 {n_clusters} 群)...")
    hac = AgglomerativeClustering(n_clusters=n_clusters)
    df_sample['cluster_id'] = hac.fit_predict(embeddings)
    
    # D. 情緒分析
    print("步驟 4/4: 正在進行情緒分析...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    sentiments = sentiment_pipeline(
        df_sample['cleaned_text'].tolist(), 
        batch_size=64, 
        truncation=True, 
        max_length=512
    )
    sentiment_map = {'positive': 'Positive', 'neutral': 'Negative', 'negative': 'Negative'}
    df_sample['sentiment'] = [sentiment_map[s['label']] for s in sentiments]

    # 準備輸出結果
    output_data = df_sample[[
        'id', 'cleaned_text', 'createdAt', 'cluster_id', 'sentiment', 'x', 'y'
    ]].copy()
    
    # ============================  FIX starts here ============================
    # 修正：使用 .apply() 來轉換日期格式
    output_data['createdAt'] = output_data['createdAt'].apply(lambda x: x.isoformat())
    # ============================= FIX ends here ==============================
    
    output_data['cluster_id'] = output_data['cluster_id'].astype(int)
    output_data['x'] = output_data['x'].astype(float)
    output_data['y'] = output_data['y'].astype(float)
    
    # Replace NaN with None for valid JSON output
    output_data.replace({np.nan: None}, inplace=True)
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data.to_dict('records'), f, ensure_ascii=False, indent=4)
        
    print(f"✅ 語意分析檔案 '{output_filename}' 已成功建立。")


def generate_reading_pane_file(df, output_filename="reading_pane_data.json"):
    """
    檔案3: 建立一個從 ID 到原始貼文資料的映射。
    """
    print("\n--- 開始產生閱讀窗格對照檔 (檔案 3/3) ---")
    
    df_indexed = df.set_index('id')
    
    # ============================  FIX starts here ============================
    # 修正：使用 .apply() 來轉換日期格式
    df_indexed['createdAt'] = df_indexed['createdAt'].apply(lambda x: x.isoformat())
    # ============================= FIX ends here ==============================
    
    # Replace NaN with None for valid JSON output
    df_indexed.replace({np.nan: None}, inplace=True)
    
    reading_pane_dict = df_indexed.to_dict(orient='index')
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(reading_pane_dict, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 閱讀窗格檔案 '{output_filename}' 已成功建立。")


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. 載入資料
    main_df = load_and_merge_data()
    
    if main_df is not None:
        # 2. 清洗文本
        print("\n--- 正在進行文本清洗 ---")
        main_df['cleaned_text'] = main_df['text'].apply(clean_text)
        print("文本清洗完成。")
        
        # 3. 產生三個輸出檔案
        generate_frequency_file(main_df)
        generate_semantic_file(main_df)
        generate_reading_pane_file(main_df)
        
        end_time = time.time()
        print(f"\n🎉 所有處理完成！總共花費: {end_time - start_time:.2f} 秒。")
        print("您現在可以使用 term_ngram_frequency.json, semantic_clustering_sentiment.json 和 reading_pane_data.json 進行視覺化。")