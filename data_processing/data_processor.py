"""
主要資料處理器
整合文本清洗、品質監控和檔案處理功能的核心處理器
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import orjson
import uuid
from datetime import datetime

from .text_cleaner import TextCleaner
from .quality_monitor import QualityMonitor
from .utils import (
    safe_json_load, safe_json_save, generate_uuid_from_tweet_id,
    validate_file_structure, get_file_stats
)


class DataProcessor:
    """主要資料處理器：負責協調整個資料處理流程"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化資料處理器
        
        Args:
            config: 完整配置字典
        """
        self.config = config
        self.processing_config = config["processing"]
        self.paths_config = self.processing_config["paths"]
        self.perf_config = self.processing_config["performance"]
        
        self.logger = logging.getLogger("data_processor.main")
        
        # 初始化子模組
        self.text_cleaner = TextCleaner(config)
        self.quality_monitor = QualityMonitor(config)
        
        # 設定路徑
        self._setup_directories()
        
        self.logger.info("DataProcessor 初始化完成")
    
    def _setup_directories(self):
        """設定必要的目錄結構"""
        directories = [
            self.paths_config["processed_data"],
            self.paths_config["logs"],
            self.paths_config["temp"]
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.debug("目錄結構設定完成")
    
    def process_single_file(self, 
                          input_file: Path, 
                          output_file: Optional[Path] = None,
                          preserve_metadata: bool = True) -> Dict[str, Any]:
        """
        處理單個JSON檔案
        
        Args:
            input_file: 輸入檔案路徑
            output_file: 輸出檔案路徑（可選）
            preserve_metadata: 是否保留原始metadata
            
        Returns:
            處理結果統計
        """
        self.logger.info(f"開始處理檔案: {input_file}")
        
        # 載入資料
        raw_data = safe_json_load(input_file)
        if raw_data is None:
            return {"error": f"無法載入檔案: {input_file}"}
        
        # 驗證檔案結構
        required_fields = ["id", "text"]
        if not validate_file_structure(input_file, required_fields):
            return {"error": f"檔案結構不符合要求: {input_file}"}
        
        # 處理資料
        start_time = time.time()
        processed_data = self._process_data_chunks(raw_data, preserve_metadata)
        processing_time = time.time() - start_time
        
        # 品質監控
        raw_texts = [item["text"] for item in raw_data]
        cleaned_results = [item.get("cleaned_data") for item in processed_data 
                          if item.get("cleaned_data") is not None]
        
        quality_report = self.quality_monitor.monitor_batch(
            raw_texts, cleaned_results, processing_time
        )
        
        # 儲存結果
        if output_file is None:
            output_file = self._generate_output_path(input_file)
        
        success = safe_json_save(processed_data, output_file)
        
        # 組裝結果統計
        result_stats = {
            "input_file": str(input_file),
            "output_file": str(output_file) if success else None,
            "input_count": len(raw_data),
            "output_count": len(processed_data),
            "processing_time": processing_time,
            "success": success,
            "quality_report": quality_report,
            "file_stats": {
                "input": get_file_stats(input_file),
                "output": get_file_stats(output_file) if success else None
            }
        }
        
        self.logger.info(
            f"檔案處理完成: {input_file} -> {output_file} "
            f"({len(raw_data)} -> {len(processed_data)} 條)"
        )
        
        return result_stats
    
    def process_directory(self, 
                         input_dir: Optional[Path] = None,
                         output_dir: Optional[Path] = None,
                         file_pattern: str = "*.json") -> Dict[str, Any]:
        """
        批次處理目錄中的所有檔案
        
        Args:
            input_dir: 輸入目錄（預設使用config中的raw_data）
            output_dir: 輸出目錄（預設使用config中的processed_data）
            file_pattern: 檔案匹配模式
            
        Returns:
            整體處理統計
        """
        if input_dir is None:
            input_dir = Path(self.paths_config["raw_data"])
        if output_dir is None:
            output_dir = Path(self.paths_config["processed_data"])
        
        self.logger.info(f"開始批次處理目錄: {input_dir}")
        
        # 找到所有匹配的檔案
        input_files = list(input_dir.glob(file_pattern))
        if not input_files:
            return {"error": f"未找到匹配的檔案: {input_dir}/{file_pattern}"}
        
        self.logger.info(f"找到 {len(input_files)} 個檔案待處理")
        
        # 重置品質監控統計
        self.quality_monitor.reset_stats()
        
        # 處理所有檔案
        all_results = []
        failed_files = []
        
        # 使用進度條顯示處理進度
        with tqdm(total=len(input_files), desc="處理檔案", unit="file") as pbar:
            
            if self.perf_config["max_workers"] > 1:
                # 多執行緒處理
                all_results = self._process_files_parallel(
                    input_files, output_dir, pbar, failed_files
                )
            else:
                # 單執行緒處理
                all_results = self._process_files_sequential(
                    input_files, output_dir, pbar, failed_files
                )
        
        # 生成整體統計
        overall_stats = self._generate_overall_stats(all_results, failed_files)
        
        # 儲存品質報告
        quality_report_path = output_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.quality_monitor.save_report(quality_report_path)
        
        # 顯示摘要
        self.quality_monitor.print_summary()
        
        self.logger.info(f"批次處理完成，共處理 {len(input_files)} 個檔案")
        
        return overall_stats
    
    def _process_files_parallel(self, 
                               input_files: List[Path], 
                               output_dir: Path,
                               pbar: tqdm,
                               failed_files: List[str]) -> List[Dict[str, Any]]:
        """多執行緒處理檔案"""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.perf_config["max_workers"]) as executor:
            # 提交所有任務
            future_to_file = {
                executor.submit(
                    self.process_single_file,
                    input_file,
                    output_dir / f"cleaned_{input_file.name}"
                ): input_file
                for input_file in input_files
            }
            
            # 收集結果
            for future in as_completed(future_to_file):
                input_file = future_to_file[future]
                try:
                    result = future.result()
                    if "error" in result:
                        failed_files.append(str(input_file))
                        self.logger.error(f"處理失敗: {input_file} - {result['error']}")
                    else:
                        all_results.append(result)
                except Exception as e:
                    failed_files.append(str(input_file))
                    self.logger.error(f"處理異常: {input_file} - {e}")
                finally:
                    pbar.update(1)
        
        return all_results
    
    def _process_files_sequential(self, 
                                 input_files: List[Path], 
                                 output_dir: Path,
                                 pbar: tqdm,
                                 failed_files: List[str]) -> List[Dict[str, Any]]:
        """單執行緒處理檔案"""
        all_results = []
        
        for input_file in input_files:
            try:
                output_file = output_dir / f"cleaned_{input_file.name}"
                result = self.process_single_file(input_file, output_file)
                
                if "error" in result:
                    failed_files.append(str(input_file))
                    self.logger.error(f"處理失敗: {input_file} - {result['error']}")
                else:
                    all_results.append(result)
                    
            except Exception as e:
                failed_files.append(str(input_file))
                self.logger.error(f"處理異常: {input_file} - {e}")
            finally:
                pbar.update(1)
        
        return all_results
    
    def _process_data_chunks(self, 
                           raw_data: List[Dict[str, Any]], 
                           preserve_metadata: bool) -> List[Dict[str, Any]]:
        """分塊處理資料以提升效能"""
        chunk_size = self.perf_config["chunk_size"]
        processed_data = []
        
        for i in range(0, len(raw_data), chunk_size):
            chunk = raw_data[i:i + chunk_size]
            chunk_results = self._process_chunk(chunk, preserve_metadata)
            processed_data.extend(chunk_results)
        
        return processed_data
    
    def _process_chunk(self, 
                      chunk: List[Dict[str, Any]], 
                      preserve_metadata: bool) -> List[Dict[str, Any]]:
        """處理單個資料塊"""
        processed_chunk = []
        
        for tweet_data in chunk:
            try:
                # 提取文本
                text = tweet_data.get("text", "")
                tweet_id = tweet_data.get("id", "")
                
                # 文本清洗
                cleaned_result = self.text_cleaner.clean_text(text)
                
                # 組裝輸出資料
                output_item = {
                    "uuid": generate_uuid_from_tweet_id(tweet_id),
                    "original_tweet_id": tweet_id,
                    "processing_timestamp": datetime.now().isoformat(),
                }
                
                # 添加清洗結果
                if cleaned_result is not None:
                    output_item["cleaned_data"] = cleaned_result
                    output_item["status"] = "success"
                else:
                    output_item["status"] = "filtered_or_failed"
                    output_item["reason"] = "retweet_filtered_or_processing_failed"
                
                # 保留原始metadata（可選）
                if preserve_metadata:
                    output_item["original_metadata"] = {
                        key: value for key, value in tweet_data.items()
                        if key not in ["text"]  # 避免重複存儲text
                    }
                
                processed_chunk.append(output_item)
                
            except Exception as e:
                self.logger.warning(f"處理推文失敗 {tweet_data.get('id', 'unknown')}: {e}")
                # 添加失敗記錄
                processed_chunk.append({
                    "uuid": generate_uuid_from_tweet_id(tweet_data.get("id", "unknown")),
                    "original_tweet_id": tweet_data.get("id", ""),
                    "status": "error",
                    "error": str(e),
                    "processing_timestamp": datetime.now().isoformat(),
                })
        
        return processed_chunk
    
    def _generate_output_path(self, input_file: Path) -> Path:
        """為輸入檔案生成對應的輸出路徑"""
        output_dir = Path(self.paths_config["processed_data"])
        return output_dir / f"cleaned_{input_file.name}"
    
    def _generate_overall_stats(self, 
                               all_results: List[Dict[str, Any]], 
                               failed_files: List[str]) -> Dict[str, Any]:
        """生成整體處理統計"""
        total_files = len(all_results) + len(failed_files)
        successful_files = len(all_results)
        
        total_input = sum(r["input_count"] for r in all_results)
        total_output = sum(r["output_count"] for r in all_results)
        total_time = sum(r["processing_time"] for r in all_results)
        
        overall_stats = {
            "processing_summary": {
                "total_files": total_files,
                "successful_files": successful_files,
                "failed_files": len(failed_files),
                "success_rate": successful_files / total_files if total_files > 0 else 0,
                "total_tweets_input": total_input,
                "total_tweets_output": total_output,
                "total_processing_time": total_time,
                "avg_processing_time_per_file": total_time / successful_files if successful_files > 0 else 0
            },
            "failed_files": failed_files,
            "file_results": all_results,
            "quality_summary": self.quality_monitor.generate_report(),
            "processing_config": self.processing_config,
            "generated_at": datetime.now().isoformat()
        }
        
        return overall_stats
    
    # ==================== 便利方法 ====================
    
    def quick_process_all(self) -> Dict[str, Any]:
        """快速處理所有raw資料檔案"""
        return self.process_directory()
    
    def process_specific_files(self, file_names: List[str]) -> Dict[str, Any]:
        """處理指定的檔案列表"""
        input_dir = Path(self.paths_config["raw_data"])
        output_dir = Path(self.paths_config["processed_data"])
        
        input_files = [input_dir / name for name in file_names]
        # 檢查檔案是否存在
        existing_files = [f for f in input_files if f.exists()]
        missing_files = [f for f in input_files if not f.exists()]
        
        if missing_files:
            self.logger.warning(f"以下檔案不存在: {[str(f) for f in missing_files]}")
        
        if not existing_files:
            return {"error": "沒有找到任何指定的檔案"}
        
        # 重置品質監控
        self.quality_monitor.reset_stats()
        
        # 處理存在的檔案
        all_results = []
        failed_files = []
        
        with tqdm(total=len(existing_files), desc="處理指定檔案", unit="file") as pbar:
            for input_file in existing_files:
                try:
                    output_file = output_dir / f"cleaned_{input_file.name}"
                    result = self.process_single_file(input_file, output_file)
                    
                    if "error" in result:
                        failed_files.append(str(input_file))
                    else:
                        all_results.append(result)
                        
                except Exception as e:
                    failed_files.append(str(input_file))
                    self.logger.error(f"處理異常: {input_file} - {e}")
                finally:
                    pbar.update(1)
        
        return self._generate_overall_stats(all_results, failed_files)
    
    def validate_processing_setup(self) -> Dict[str, Any]:
        """驗證處理器設定"""
        validation_results = {
            "directories": {},
            "text_cleaner": {},
            "config": {},
            "sample_processing": {}
        }
        
        # 檢查目錄
        for path_name, path_value in self.paths_config.items():
            path_obj = Path(path_value)
            validation_results["directories"][path_name] = {
                "exists": path_obj.exists(),
                "is_directory": path_obj.is_dir() if path_obj.exists() else False,
                "writable": path_obj.exists() and path_obj.is_dir()
            }
        
        # 檢查文本清洗器
        validation_results["text_cleaner"] = self.text_cleaner.validate_setup()
        
        # 檢查配置
        validation_results["config"] = {
            "max_workers": self.perf_config["max_workers"] > 0,
            "batch_size": self.perf_config["batch_size"] > 0,
            "chunk_size": self.perf_config["chunk_size"] > 0
        }
        
        # 測試樣本處理
        sample_tweet = {
            "id": "test123",
            "text": "$BTC is going to the moon! 🚀 #crypto https://example.com @user",
            "createdAt": "2024-01-01"
        }
        
        try:
            result = self._process_chunk([sample_tweet], preserve_metadata=True)
            validation_results["sample_processing"] = {
                "success": len(result) > 0,
                "has_cleaned_data": result[0].get("cleaned_data") is not None if result else False
            }
        except Exception as e:
            validation_results["sample_processing"] = {
                "success": False,
                "error": str(e)
            }
        
        return validation_results 