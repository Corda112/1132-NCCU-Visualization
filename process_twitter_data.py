#!/usr/bin/env python3
"""
Twitter推文資料處理主程式
完整的資料清洗和預處理流程，包含六個專業處理步驟

使用方法:
    python process_twitter_data.py --help                    # 顯示幫助
    python process_twitter_data.py --all                     # 處理所有檔案
    python process_twitter_data.py --files file1.json file2.json  # 處理指定檔案
    python process_twitter_data.py --validate                # 驗證設定
    python process_twitter_data.py --sample                  # 處理樣本檔案測試
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional

# 添加項目根目錄到Python路徑
sys.path.insert(0, str(Path(__file__).parent))

from data_processing import load_config, setup_logging, DataProcessor


def setup_argument_parser() -> argparse.ArgumentParser:
    """設定命令行參數解析器"""
    parser = argparse.ArgumentParser(
        description="Twitter推文資料處理工具 - 專業文本清洗與預處理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  python process_twitter_data.py --all                        # 處理所有raw資料
  python process_twitter_data.py --files 20241201*.json      # 處理特定檔案
  python process_twitter_data.py --validate                   # 驗證系統設定
  python process_twitter_data.py --sample --limit 100        # 處理樣本資料
  python process_twitter_data.py --all --workers 4           # 使用4個執行緒
  python process_twitter_data.py --all --no-metadata         # 不保留原始metadata

處理結果將儲存到 DATA/processed/ 目錄
品質報告將自動生成並顯示處理統計
        """
    )
    
    # 主要操作選項
    main_group = parser.add_mutually_exclusive_group(required=True)
    main_group.add_argument(
        "--all", 
        action="store_true",
        help="處理所有raw資料檔案"
    )
    main_group.add_argument(
        "--files", 
        nargs="+", 
        metavar="FILE",
        help="指定要處理的檔案名稱（支援萬用字元）"
    )
    main_group.add_argument(
        "--validate", 
        action="store_true",
        help="驗證系統設定和依賴"
    )
    main_group.add_argument(
        "--sample",
        action="store_true", 
        help="處理樣本資料進行測試"
    )
    
    # 處理選項
    parser.add_argument(
        "--config",
        type=str,
        default="data_processing/config.yaml",
        help="配置檔案路徑 (預設: data_processing/config.yaml)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="並行處理執行緒數量 (覆蓋配置檔案設定)"
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="不保留原始推文metadata以節省空間"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="限制處理的檔案數量（用於測試）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="自訂輸出目錄（覆蓋配置檔案設定）"
    )
    
    # 日誌選項
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="顯示詳細日誌"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true", 
        help="只顯示錯誤訊息"
    )
    
    return parser


def validate_dependencies():
    """檢查必要的依賴包"""
    missing_deps = []
    
    try:
        import spacy
        # 檢查spaCy模型
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("❌ spaCy英文模型未安裝")
            print("請執行: python -m spacy download en_core_web_sm")
            return False
    except ImportError:
        missing_deps.append("spacy")
    
    try:
        import emoji
    except ImportError:
        missing_deps.append("emoji")
    
    try:
        import orjson
    except ImportError:
        missing_deps.append("orjson")
    
    try:
        import yaml
    except ImportError:
        missing_deps.append("pyyaml")
    
    try:
        import tqdm
    except ImportError:
        missing_deps.append("tqdm")
    
    if missing_deps:
        print(f"❌ 缺少必要依賴: {', '.join(missing_deps)}")
        print("請執行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依賴檢查通過")
    return True


def setup_spacy_model():
    """設定spaCy模型"""
    try:
        import spacy
        print("📥 正在載入spaCy模型...")
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy模型載入成功")
        return True
    except OSError:
        print("❌ 找不到spaCy英文模型")
        print("正在嘗試下載...")
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ spaCy模型下載成功")
                return True
            else:
                print(f"❌ spaCy模型下載失敗: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ 自動下載失敗: {e}")
            print("請手動執行: python -m spacy download en_core_web_sm")
            return False


def expand_file_patterns(patterns: List[str], base_dir: Path) -> List[str]:
    """展開檔案匹配模式"""
    expanded_files = []
    
    for pattern in patterns:
        if '*' in pattern or '?' in pattern:
            # 萬用字元匹配
            matched_files = list(base_dir.glob(pattern))
            expanded_files.extend([f.name for f in matched_files])
        else:
            # 直接檔案名
            expanded_files.append(pattern)
    
    return list(set(expanded_files))  # 去重


def display_processing_header():
    """顯示處理開始的標題"""
    print("\n" + "="*70)
    print("🚀 Twitter推文資料處理系統")
    print("="*70)
    print("📋 處理流程:")
    print("   1️⃣  轉小寫 & 過濾轉推")
    print("   2️⃣  移除URL/Mention，保留Hashtag") 
    print("   3️⃣  Emoji轉換為文字描述")
    print("   4️⃣  自訂斷詞（保留$BTC等符號）")
    print("   5️⃣  停用詞過濾")
    print("   6️⃣  詞形還原")
    print("="*70)


def display_results_summary(results: dict):
    """顯示處理結果摘要"""
    if "error" in results:
        print(f"\n❌ 處理失敗: {results['error']}")
        return
    
    summary = results.get("processing_summary", {})
    print(f"\n📊 處理完成摘要:")
    print(f"   📁 處理檔案: {summary.get('successful_files', 0)}/{summary.get('total_files', 0)}")
    print(f"   📝 處理推文: {summary.get('total_tweets_input', 0):,} → {summary.get('total_tweets_output', 0):,}")
    print(f"   ⏱️  總處理時間: {summary.get('total_processing_time', 0):.2f}秒")
    
    if summary.get('failed_files', 0) > 0:
        print(f"   ⚠️  失敗檔案: {len(results.get('failed_files', []))}")
    
    # 顯示品質統計
    quality = results.get("quality_summary", {})
    if quality and "overall_statistics" in quality:
        stats = quality["overall_statistics"]
        print(f"\n📈 品質統計:")
        print(f"   ✅ 成功率: {stats.get('successful_rate', 0):.2%}")
        print(f"   🔄 轉推過濾: {stats.get('retweet_rate', 0):.2%}")
        print(f"   ❌ 錯誤率: {stats.get('error_rate', 0):.2%}")


def main():
    """主程式入口"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # 檢查依賴
    if not validate_dependencies():
        sys.exit(1)
    
    # 驗證模式
    if args.validate:
        print("🔍 系統驗證模式")
        
        # 檢查spaCy模型
        if not setup_spacy_model():
            sys.exit(1)
        
        # 載入配置
        try:
            config = load_config(args.config)
            print("✅ 配置檔案載入成功")
        except Exception as e:
            print(f"❌ 配置檔案載入失敗: {e}")
            sys.exit(1)
        
        # 設定日誌
        logger = setup_logging(config)
        
        # 初始化處理器並驗證
        try:
            processor = DataProcessor(config)
            validation_results = processor.validate_processing_setup()
            
            print("\n📋 系統驗證結果:")
            
            # 目錄檢查
            dirs = validation_results["directories"]
            for name, status in dirs.items():
                icon = "✅" if status["exists"] and status["is_directory"] else "❌"
                print(f"   {icon} {name}: {status}")
            
            # 文本清洗器檢查
            cleaner = validation_results["text_cleaner"]
            if cleaner.get("validation_failed"):
                print(f"   ❌ 文本清洗器: {cleaner.get('error')}")
            else:
                success_count = sum(1 for v in cleaner.values() if v is True)
                print(f"   ✅ 文本清洗器: {success_count}/{len(cleaner)} 測試通過")
            
            # 樣本處理檢查
            sample = validation_results["sample_processing"]
            if sample["success"]:
                print("   ✅ 樣本處理: 成功")
            else:
                print(f"   ❌ 樣本處理: {sample.get('error', '失敗')}")
            
            print("\n🎉 系統驗證完成!")
            
        except Exception as e:
            print(f"❌ 驗證過程發生錯誤: {e}")
            sys.exit(1)
        
        return
    
    # 確保spaCy模型可用
    if not setup_spacy_model():
        sys.exit(1)
    
    # 載入配置
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ 配置檔案載入失敗: {e}")
        sys.exit(1)
    
    # 調整配置（根據命令行參數）
    if args.workers:
        config["processing"]["performance"]["max_workers"] = args.workers
    
    if args.output_dir:
        config["processing"]["paths"]["processed_data"] = args.output_dir
    
    if args.verbose:
        config["logging"]["level"] = "DEBUG"
    elif args.quiet:
        config["logging"]["level"] = "ERROR"
    
    # 設定日誌
    logger = setup_logging(config)
    
    # 顯示處理標題
    display_processing_header()
    
    # 初始化處理器
    try:
        processor = DataProcessor(config)
    except Exception as e:
        print(f"❌ 處理器初始化失敗: {e}")
        sys.exit(1)
    
    # 執行處理任務
    results = None
    preserve_metadata = not args.no_metadata
    
    try:
        if args.all:
            print("📁 處理所有raw資料檔案...")
            results = processor.quick_process_all()
            
        elif args.files:
            # 展開檔案模式
            raw_dir = Path(config["processing"]["paths"]["raw_data"])
            expanded_files = expand_file_patterns(args.files, raw_dir)
            
            if args.limit:
                expanded_files = expanded_files[:args.limit]
            
            print(f"📁 處理指定檔案: {len(expanded_files)} 個檔案")
            results = processor.process_specific_files(expanded_files)
            
        elif args.sample:
            # 樣本處理模式
            raw_dir = Path(config["processing"]["paths"]["raw_data"])
            sample_files = list(raw_dir.glob("*.json"))
            
            if not sample_files:
                print("❌ 沒有找到樣本檔案")
                sys.exit(1)
            
            # 限制樣本檔案數量
            limit = args.limit or 2
            sample_files = sample_files[:limit]
            
            print(f"🧪 樣本處理模式: {len(sample_files)} 個檔案")
            results = processor.process_specific_files([f.name for f in sample_files])
        
        # 顯示結果
        if results:
            display_results_summary(results)
            
            # 儲存處理統計
            output_dir = Path(config["processing"]["paths"]["processed_data"])
            stats_file = output_dir / f"processing_stats_{results['generated_at'][:10]}.json"
            
            try:
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
                print(f"\n📊 詳細統計已儲存至: {stats_file}")
            except Exception as e:
                logger.warning(f"統計檔案儲存失敗: {e}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 處理被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 處理過程發生錯誤: {e}")
        logger.exception("處理錯誤詳情")
        sys.exit(1)
    
    print("\n🎉 所有任務完成!")


if __name__ == "__main__":
    main() 