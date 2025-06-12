'''
# 使用FinBert模型分類情緒，共分成正、反、一般三類
import pandas as pd
import numpy as np
import json
import glob
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
# STEP 1: 數據讀取與新格式處理 (已更新)
# ==============================================================================
def load_and_merge_data(path_pattern=None):
    """
    載入所有符合樣式的、已清洗過的 JSON 檔案，並從巢狀結構中提取所需資料。
    如果未提供 `path_pattern`，會自動從相對於腳本位置的 '../processed' 資料夾尋找。
    """
    if path_pattern is None:
        # 根據使用者提供的檔案結構，腳本在 'DATA/tmp'，資料在 'DATA/processed'。
        # 我們將基於腳本的絕對路徑來建構資料檔案的路徑。
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(script_dir, '..', 'processed'))
        path_pattern = os.path.join(data_dir, 'cleaned_*.json')
        print(f"未指定路徑，自動在資料夾 '{data_dir}' 中搜尋 'cleaned_*.json' 檔案...")

    json_files = glob.glob(path_pattern)
    if not json_files:
        # 如果自動偵測失敗，提供更明確的錯誤訊息
        searched_path = os.path.dirname(path_pattern)
        print(f"錯誤：在路徑 '{searched_path}' 中找不到任何符合 'cleaned_*.json' 的檔案。")
        print("請確認您的 'cleaned_*.json' 檔案確實存放在 'DATA/processed' 資料夾中。")
        return None
        
    print(f"找到 {len(json_files)} 個已清洗的檔案: {json_files}")
    
    df_list = [pd.read_json(f) for f in json_files]
    df = pd.concat(df_list, ignore_index=True)
    
    # 使用 pandas.json_normalize 來高效地處理巢狀JSON
    # 從 'cleaned_data' 和 'original_metadata' 中提取我們需要的欄位
    try:
        cleaned_data_df = pd.json_normalize(df['cleaned_data'])
        original_metadata_df = pd.json_normalize(df['original_metadata'])
    except KeyError as e:
        print(f"錯誤：輸入的 JSON 檔案缺少必要的欄位：{e}。請檢查檔案結構。")
        return None

    # 組合一個乾淨、扁平化的 DataFrame
    final_df = pd.DataFrame({
        'id': df['original_tweet_id'].astype(str),
        'cleaned_text': cleaned_data_df['cleaned_text'],
        'createdAt': pd.to_datetime(original_metadata_df['createdAt']),
        'original_data': df['original_metadata'] # 保留完整的原始資料以供閱讀窗格使用
    })
    
    # 移除 'cleaned_text' 為空值的行，以避免後續處理出錯
    initial_count = len(final_df)
    final_df.dropna(subset=['cleaned_text'], inplace=True)
    
    # 移除重複的推文 (以 id 為準)
    final_df = final_df.drop_duplicates(subset='id').reset_index(drop=True)
    
    print(f"成功載入並合併資料，共 {len(final_df)} 篇不重複的貼文 (已濾除空內容與重複項)。")
    if len(final_df) < initial_count:
        print(f"注意：在清理過程中移除了 {initial_count - len(final_df)} 筆資料 (因內容為空或重複)。")
        
    return final_df

# ==============================================================================
# STEP 2: 資料清洗函數 (此函數已不再需要，故刪除)
# ==============================================================================
# def clean_text(text):  <-- 此函數已被移除

# ==============================================================================
# STEP 3: 產生輸出檔案
# ==============================================================================

def generate_frequency_file(df, output_filename="term_ngram_frequency.json"):
    """
    檔案1: 計算每日的 Term 和 N-gram 頻率。
    """
    print("\n--- 開始產生詞彙頻率檔案 (檔案 1/3) ---")
    df['date'] = df['createdAt'].dt.date.astype(str)
    
    vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english', min_df=5)
    
    vectorizer.fit(df['cleaned_text'])
    
    daily_freq = {}
    
    for date, group in df.groupby('date'):
        print(f"正在處理 {date} 的數據...")
        X = vectorizer.transform(group['cleaned_text'])
        freqs = X.sum(axis=0).A1
        term_freqs = {term: int(freq) for term, freq in zip(vectorizer.get_feature_names_out(), freqs) if freq > 0}
        daily_freq[date] = term_freqs
        
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(daily_freq, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 詞彙頻率檔案 '{output_filename}' 已成功建立。")


def generate_semantic_file(df, output_filename="semantic_clustering_sentiment.json"):
    """
    檔案2: 進行向量化、降維、分群與 FinBERTa 情緒分析。
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
    
    # D. 情緒分析 (使用 FinBERTa)
    print("步驟 4/4: 正在進行情緒分析 (使用 FinBERTa 模型)...")
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    sentiments = sentiment_pipeline(
        df_sample['cleaned_text'].tolist(), 
        batch_size=64, 
        truncation=True, 
        max_length=512
    )
    sentiment_map = {'positive': 'Positive', 'neutral': 'Neutral', 'negative': 'Negative'}
    df_sample['sentiment'] = [sentiment_map[s['label']] for s in sentiments]

    # 準備輸出結果
    output_data = df_sample[[
        'id', 'cleaned_text', 'createdAt', 'cluster_id', 'sentiment', 'x', 'y'
    ]].copy()
    output_data['createdAt'] = output_data['createdAt'].apply(lambda x: x.isoformat())
    output_data['cluster_id'] = output_data['cluster_id'].astype(int)
    output_data['x'] = output_data['x'].astype(float)
    output_data['y'] = output_data['y'].astype(float)
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data.to_dict('records'), f, ensure_ascii=False, indent=4)
        
    print(f"✅ 語意分析檔案 '{output_filename}' 已成功建立。")


def generate_reading_pane_file(df, output_filename="reading_pane_data.json"):
    """
    檔案3: 建立一個從 ID 到原始貼文資料的映射 (已更新)。
    """
    print("\n--- 開始產生閱讀窗格對照檔 (檔案 3/3) ---")
    
    # 直接使用我們在讀取資料時保留的 'original_data' 欄位
    # 將其轉換為以 'id' 為 key 的字典
    reading_pane_dict = pd.Series(df.original_data.values, index=df.id).to_dict()
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(reading_pane_dict, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 閱讀窗格檔案 '{output_filename}' 已成功建立。")

# ==============================================================================
# MAIN EXECUTION BLOCK (已更新)
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. 載入並處理新格式的資料
    main_df = load_and_merge_data()
    
    if main_df is not None:
        # 2. 文本清洗步驟已被移除，因為我們直接使用檔案中的 'cleaned_text'
        
        # 3. 產生三個輸出檔案
        generate_frequency_file(main_df)
        generate_semantic_file(main_df)
        generate_reading_pane_file(main_df)
        
        end_time = time.time()
        print(f"\n🎉 所有處理完成！總共花費: {end_time - start_time:.2f} 秒。")
        print("您現在可以使用 term_ngram_frequency.json, semantic_clustering_sentiment.json 和 reading_pane_data.json 進行視覺化。")
'''








# 使用RoBert模型分類情緒，共分成正、反、一般三類
import pandas as pd
import numpy as np
import json
import glob
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
# STEP 1: 數據讀取與新格式處理 (已更新)
# ==============================================================================
def load_and_merge_data(path_pattern=None):
    """
    載入所有符合樣式的、已清洗過的 JSON 檔案，並從巢狀結構中提取所需資料。
    如果未提供 `path_pattern`，會自動從相對於腳本位置的 '../processed' 資料夾尋找。
    """
    if path_pattern is None:
        # 根據使用者提供的檔案結構，腳本在 'DATA/tmp'，資料在 'DATA/processed'。
        # 我們將基於腳本的絕對路徑來建構資料檔案的路徑。
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(script_dir, '..', 'processed'))
        path_pattern = os.path.join(data_dir, 'cleaned_*.json')
        print(f"未指定路徑，自動在資料夾 '{data_dir}' 中搜尋 'cleaned_*.json' 檔案...")

    json_files = glob.glob(path_pattern)
    if not json_files:
        # 如果自動偵測失敗，提供更明確的錯誤訊息
        searched_path = os.path.dirname(path_pattern)
        print(f"錯誤：在路徑 '{searched_path}' 中找不到任何符合 'cleaned_*.json' 的檔案。")
        print("請確認您的 'cleaned_*.json' 檔案確實存放在 'DATA/processed' 資料夾中。")
        return None
        
    print(f"找到 {len(json_files)} 個已清洗的檔案: {json_files}")
    
    df_list = [pd.read_json(f) for f in json_files]
    df = pd.concat(df_list, ignore_index=True)
    
    # 使用 pandas.json_normalize 來高效地處理巢狀JSON
    # 從 'cleaned_data' 和 'original_metadata' 中提取我們需要的欄位
    try:
        cleaned_data_df = pd.json_normalize(df['cleaned_data'])
        original_metadata_df = pd.json_normalize(df['original_metadata'])
    except KeyError as e:
        print(f"錯誤：輸入的 JSON 檔案缺少必要的欄位：{e}。請檢查檔案結構。")
        return None

    # 組合一個乾淨、扁平化的 DataFrame
    final_df = pd.DataFrame({
        'id': df['original_tweet_id'].astype(str),
        'cleaned_text': cleaned_data_df['cleaned_text'],
        'createdAt': pd.to_datetime(original_metadata_df['createdAt']),
        'original_data': df['original_metadata'] # 保留完整的原始資料以供閱讀窗格使用
    })
    
    # 移除 'cleaned_text' 為空值的行，以避免後續處理出錯
    initial_count = len(final_df)
    final_df.dropna(subset=['cleaned_text'], inplace=True)
    
    # 移除重複的推文 (以 id 為準)
    final_df = final_df.drop_duplicates(subset='id').reset_index(drop=True)
    
    print(f"成功載入並合併資料，共 {len(final_df)} 篇不重複的貼文 (已濾除空內容與重複項)。")
    if len(final_df) < initial_count:
        print(f"注意：在清理過程中移除了 {initial_count - len(final_df)} 筆資料 (因內容為空或重複)。")
        
    return final_df

# ==============================================================================
# STEP 2: 資料清洗函數 (此函數已不再需要，故刪除)
# ==============================================================================
# def clean_text(text):  <-- 此函數已被移除

# ==============================================================================
# STEP 3: 產生輸出檔案
# ==============================================================================

def generate_frequency_file(df, output_filename="term_ngram_frequency.json"):
    """
    檔案1: 計算每日的 Term 和 N-gram 頻率。
    """
    print("\n--- 開始產生詞彙頻率檔案 (檔案 1/3) ---")
    df['date'] = df['createdAt'].dt.date.astype(str)
    
    vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english', min_df=5)
    
    vectorizer.fit(df['cleaned_text'])
    
    daily_freq = {}
    
    for date, group in df.groupby('date'):
        print(f"正在處理 {date} 的數據...")
        X = vectorizer.transform(group['cleaned_text'])
        freqs = X.sum(axis=0).A1
        term_freqs = {term: int(freq) for term, freq in zip(vectorizer.get_feature_names_out(), freqs) if freq > 0}
        daily_freq[date] = term_freqs
        
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(daily_freq, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 詞彙頻率檔案 '{output_filename}' 已成功建立。")


def generate_semantic_file(df, output_filename="semantic_clustering_sentiment.json"):
    """
    檔案2: 進行向量化、降維、分群與 RoBERTa 三分類情緒分析。
    """
    print("\n--- 開始產生語意分析檔案 (檔案 2/3) ---")
    
    df_sample = df.copy()
    print(f"將對 {len(df_sample)} 篇文章進行語意分析...")

    # A. 文本向量化 (SBERT) - 此部分不變
    print("步驟 1/4: 正在使用 SBERT 產生文本向量 (此步驟可能需要較長時間)...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sbert_model.encode(df_sample['cleaned_text'].tolist(), show_progress_bar=True)
    
    # B. 降維 (PCA) - 此部分不變
    print("步驟 2/4: 正在使用 PCA 進行降維...")
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(embeddings)
    df_sample['x'] = coordinates[:, 0]
    df_sample['y'] = coordinates[:, 1]
    
    # C. 分群 (HAC) - 此部分不變
    n_clusters = 20
    print(f"步驟 3/4: 正在使用 HAC 進行分群 (分成 {n_clusters} 群)...")
    hac = AgglomerativeClustering(n_clusters=n_clusters)
    df_sample['cluster_id'] = hac.fit_predict(embeddings)
    
    # D. 情緒分析 (使用 RoBERTa 模型進行三分類)
    print("步驟 4/4: 正在進行情緒分析 (使用 Twitter RoBERTa 模型)...")
    
    # ============================  CHANGE starts here ============================
    # 修正：將模型換回 Twitter RoBERTa 模型
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    # ============================= CHANGE ends here ==============================

    sentiments = sentiment_pipeline(
        df_sample['cleaned_text'].tolist(), 
        batch_size=64, 
        truncation=True, 
        max_length=512
    )
    
    # 這個 sentiment_map 保持不變，因為它已經包含了處理三種分類的邏輯
    sentiment_map = {'positive': 'Positive', 'neutral': 'Neutral', 'negative': 'Negative'}
    df_sample['sentiment'] = [sentiment_map[s['label']] for s in sentiments]

    # 準備輸出結果 - 此部分不變
    output_data = df_sample[[
        'id', 'cleaned_text', 'createdAt', 'cluster_id', 'sentiment', 'x', 'y'
    ]].copy()
    output_data['createdAt'] = output_data['createdAt'].apply(lambda x: x.isoformat())
    output_data['cluster_id'] = output_data['cluster_id'].astype(int)
    output_data['x'] = output_data['x'].astype(float)
    output_data['y'] = output_data['y'].astype(float)
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data.to_dict('records'), f, ensure_ascii=False, indent=4)
        
    print(f"✅ 語意分析檔案 '{output_filename}' 已成功建立。")


def generate_reading_pane_file(df, output_filename="reading_pane_data.json"):
    """
    檔案3: 建立一個從 ID 到原始貼文資料的映射 (已更新)。
    """
    print("\n--- 開始產生閱讀窗格對照檔 (檔案 3/3) ---")
    
    # 直接使用我們在讀取資料時保留的 'original_data' 欄位
    # 將其轉換為以 'id' 為 key 的字典
    reading_pane_dict = pd.Series(df.original_data.values, index=df.id).to_dict()
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(reading_pane_dict, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 閱讀窗格檔案 '{output_filename}' 已成功建立。")

# ==============================================================================
# MAIN EXECUTION BLOCK (已更新)
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. 載入並處理新格式的資料
    main_df = load_and_merge_data()
    
    if main_df is not None:
        # 2. 文本清洗步驟已被移除，因為我們直接使用檔案中的 'cleaned_text'
        
        # 3. 產生三個輸出檔案
        generate_frequency_file(main_df)
        generate_semantic_file(main_df)
        generate_reading_pane_file(main_df)
        
        end_time = time.time()
        print(f"\n🎉 所有處理完成！總共花費: {end_time - start_time:.2f} 秒。")
        print("您現在可以使用 term_ngram_frequency.json, semantic_clustering_sentiment.json 和 reading_pane_data.json 進行視覺化。")