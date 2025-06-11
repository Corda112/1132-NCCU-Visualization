"""
資料品質監控模組
監控資料處理過程中的品質指標，並生成品質報告
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import Counter, defaultdict
import json
from pathlib import Path


class QualityMonitor:
    """資料品質監控器：追蹤和分析資料處理品質"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化品質監控器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.quality_config = config["processing"]["quality_checks"]
        self.logger = logging.getLogger("data_processor.quality_monitor")
        
        # 初始化統計數據
        self.reset_stats()
        
        self.logger.info("QualityMonitor 初始化完成")
    
    def reset_stats(self):
        """重置統計數據"""
        self.stats = {
            "total_processed": 0,
            "successful_cleaned": 0,
            "retweets_filtered": 0,
            "empty_after_cleaning": 0,
            "processing_errors": 0,
            "text_length_distribution": defaultdict(int),
            "token_count_distribution": defaultdict(int),
            "processing_steps_stats": {
                "urls_mentions_removed": 0,
                "emojis_converted": 0,
                "stopwords_removed": 0,
                "lemmatized": 0
            },
            "quality_issues": {
                "too_short": 0,
                "too_long": 0,
                "no_meaningful_content": 0
            },
            "special_tokens": {
                "crypto_symbols": Counter(),
                "emoji_types": Counter(),
                "hashtags": Counter()
            },
            "processing_times": []
        }
    
    def monitor_batch(self, 
                     raw_texts: List[str], 
                     cleaned_results: List[Optional[Dict[str, Any]]],
                     processing_time: float) -> Dict[str, Any]:
        """
        監控一個批次的處理結果
        
        Args:
            raw_texts: 原始文本列表
            cleaned_results: 清洗結果列表
            processing_time: 處理時間（秒）
            
        Returns:
            批次品質報告
        """
        batch_stats = {
            "batch_size": len(raw_texts),
            "successful_rate": 0,
            "retweet_rate": 0,
            "processing_time": processing_time,
            "avg_processing_time_per_item": processing_time / len(raw_texts) if raw_texts else 0
        }
        
        successful_count = 0
        retweet_count = 0
        
        for i, (raw_text, result) in enumerate(zip(raw_texts, cleaned_results)):
            self.stats["total_processed"] += 1
            
            # 分析原始文本
            if raw_text and raw_text.strip().startswith("RT "):
                retweet_count += 1
                self.stats["retweets_filtered"] += 1
            
            # 分析處理結果
            if result is None:
                if raw_text and not raw_text.strip().startswith("RT "):
                    self.stats["processing_errors"] += 1
            else:
                successful_count += 1
                self.stats["successful_cleaned"] += 1
                self._analyze_cleaned_result(result)
        
        # 計算批次統計
        batch_stats["successful_rate"] = successful_count / len(raw_texts) if raw_texts else 0
        batch_stats["retweet_rate"] = retweet_count / len(raw_texts) if raw_texts else 0
        
        # 記錄處理時間
        self.stats["processing_times"].append(processing_time)
        
        # 品質檢核
        quality_alerts = self._check_quality_thresholds(batch_stats)
        
        return {
            "batch_stats": batch_stats,
            "quality_alerts": quality_alerts
        }
    
    def _analyze_cleaned_result(self, result: Dict[str, Any]):
        """分析單個清洗結果"""
        # 文本長度分析
        original_length = len(result["original_text"])
        cleaned_length = len(result["cleaned_text"])
        token_count = result["token_count"]
        
        # 長度分佈統計
        length_category = self._categorize_length(original_length)
        self.stats["text_length_distribution"][length_category] += 1
        
        token_category = self._categorize_token_count(token_count)
        self.stats["token_count_distribution"][token_category] += 1
        
        # 處理步驟統計
        processing_steps = result["processing_steps"]
        for step, occurred in processing_steps.items():
            if occurred:
                self.stats["processing_steps_stats"][step] += 1
        
        # 品質問題檢測
        if original_length < self.quality_config["min_text_length"]:
            self.stats["quality_issues"]["too_short"] += 1
        elif original_length > self.quality_config["max_text_length"]:
            self.stats["quality_issues"]["too_long"] += 1
        
        if token_count == 0:
            self.stats["quality_issues"]["no_meaningful_content"] += 1
            self.stats["empty_after_cleaning"] += 1
        
        # 特殊token分析
        self._analyze_special_tokens(result["final_tokens"])
    
    def _categorize_length(self, length: int) -> str:
        """將文本長度分類"""
        if length < 50:
            return "very_short"
        elif length < 100:
            return "short"
        elif length < 200:
            return "medium"
        elif length < 300:
            return "long"
        else:
            return "very_long"
    
    def _categorize_token_count(self, count: int) -> str:
        """將token數量分類"""
        if count == 0:
            return "empty"
        elif count < 5:
            return "very_few"
        elif count < 10:
            return "few"
        elif count < 20:
            return "medium"
        elif count < 50:
            return "many"
        else:
            return "very_many"
    
    def _analyze_special_tokens(self, tokens: List[str]):
        """分析特殊token"""
        import re
        
        crypto_pattern = re.compile(r'\$[a-zA-Z]{2,10}')
        emoji_pattern = re.compile(r':[a-zA-Z_]+:')
        
        for token in tokens:
            # 加密貨幣符號
            if crypto_pattern.match(token):
                self.stats["special_tokens"]["crypto_symbols"][token] += 1
            
            # Emoji描述
            elif emoji_pattern.match(token):
                self.stats["special_tokens"]["emoji_types"][token] += 1
            
            # Hashtag（已移除#的）
            elif token.lower() in ["bitcoin", "crypto", "defi", "eth", "btc"]:
                self.stats["special_tokens"]["hashtags"][token] += 1
    
    def _check_quality_thresholds(self, batch_stats: Dict[str, Any]) -> List[str]:
        """檢查品質閾值，生成警告"""
        alerts = []
        
        # 轉推比例警告
        if batch_stats["retweet_rate"] > self.quality_config["retweet_threshold"]:
            alerts.append(
                f"高轉推比例警告: {batch_stats['retweet_rate']:.2%} "
                f"(閾值: {self.quality_config['retweet_threshold']:.2%})"
            )
        
        # 成功率警告
        if batch_stats["successful_rate"] < 0.8:
            alerts.append(
                f"低成功率警告: {batch_stats['successful_rate']:.2%} "
                f"(建議 > 80%)"
            )
        
        # 處理速度警告
        if batch_stats["avg_processing_time_per_item"] > 0.1:
            alerts.append(
                f"處理速度警告: {batch_stats['avg_processing_time_per_item']:.3f}秒/條 "
                f"(建議 < 0.1秒/條)"
            )
        
        return alerts
    
    def generate_report(self) -> Dict[str, Any]:
        """生成完整的品質報告"""
        if self.stats["total_processed"] == 0:
            return {"error": "尚未處理任何資料"}
        
        # 計算整體統計
        overall_stats = {
            "total_processed": self.stats["total_processed"],
            "successful_rate": self.stats["successful_cleaned"] / self.stats["total_processed"],
            "retweet_rate": self.stats["retweets_filtered"] / self.stats["total_processed"],
            "error_rate": self.stats["processing_errors"] / self.stats["total_processed"],
            "empty_after_cleaning_rate": self.stats["empty_after_cleaning"] / self.stats["total_processed"]
        }
        
        # 處理效能統計
        performance_stats = {}
        if self.stats["processing_times"]:
            import statistics
            times = self.stats["processing_times"]
            performance_stats = {
                "total_batches": len(times),
                "total_processing_time": sum(times),
                "avg_batch_time": statistics.mean(times),
                "median_batch_time": statistics.median(times),
                "max_batch_time": max(times),
                "min_batch_time": min(times)
            }
        
        # 品質分析
        quality_analysis = {
            "text_length_distribution": dict(self.stats["text_length_distribution"]),
            "token_count_distribution": dict(self.stats["token_count_distribution"]),
            "processing_steps_effectiveness": self._calculate_step_effectiveness(),
            "quality_issues_summary": dict(self.stats["quality_issues"]),
            "special_tokens_summary": {
                "top_crypto_symbols": dict(self.stats["special_tokens"]["crypto_symbols"].most_common(10)),
                "top_emojis": dict(self.stats["special_tokens"]["emoji_types"].most_common(10)),
                "top_hashtags": dict(self.stats["special_tokens"]["hashtags"].most_common(10))
            }
        }
        
        # 建議和警告
        recommendations = self._generate_recommendations(overall_stats, quality_analysis)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "overall_statistics": overall_stats,
            "performance_statistics": performance_stats,
            "quality_analysis": quality_analysis,
            "recommendations": recommendations,
            "config_used": self.quality_config
        }
        
        return report
    
    def _calculate_step_effectiveness(self) -> Dict[str, float]:
        """計算各處理步驟的效果"""
        total_successful = self.stats["successful_cleaned"]
        if total_successful == 0:
            return {}
        
        effectiveness = {}
        for step, count in self.stats["processing_steps_stats"].items():
            effectiveness[step] = count / total_successful
        
        return effectiveness
    
    def _generate_recommendations(self, 
                                overall_stats: Dict[str, Any], 
                                quality_analysis: Dict[str, Any]) -> List[str]:
        """生成改進建議"""
        recommendations = []
        
        # 成功率建議
        if overall_stats["successful_rate"] < 0.9:
            recommendations.append(
                f"成功率較低 ({overall_stats['successful_rate']:.2%})，"
                "建議檢查文本品質或調整清洗參數"
            )
        
        # 轉推比例建議
        if overall_stats["retweet_rate"] > 0.15:
            recommendations.append(
                f"轉推比例較高 ({overall_stats['retweet_rate']:.2%})，"
                "可能需要調整資料收集策略"
            )
        
        # 空內容建議
        if overall_stats["empty_after_cleaning_rate"] > 0.1:
            recommendations.append(
                f"清洗後空內容比例較高 ({overall_stats['empty_after_cleaning_rate']:.2%})，"
                "建議調整停用詞設定或文本處理參數"
            )
        
        # Token分佈建議
        token_dist = quality_analysis["token_count_distribution"]
        total_successful = overall_stats["successful_rate"] * overall_stats.get("total_processed", 1)
        if token_dist.get("empty", 0) + token_dist.get("very_few", 0) > total_successful * 0.2:
            recommendations.append(
                "過多短文本，建議檢查資料來源品質或調整最小文本長度閾值"
            )
        
        # 特殊token建議
        crypto_symbols = quality_analysis["special_tokens_summary"]["top_crypto_symbols"]
        if crypto_symbols:
            recommendations.append(
                f"發現 {len(crypto_symbols)} 種加密貨幣符號，"
                "文本清洗器成功保留了特殊符號"
            )
        
        if not recommendations:
            recommendations.append("資料品質良好，無需特別調整")
        
        return recommendations
    
    def save_report(self, output_path: Path) -> bool:
        """儲存品質報告到檔案"""
        try:
            report = self.generate_report()
            
            # 確保目錄存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"品質報告已儲存至: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"報告儲存失敗: {e}")
            return False
    
    def print_summary(self):
        """打印簡要統計摘要"""
        if self.stats["total_processed"] == 0:
            print("尚未處理任何資料")
            return
        
        total = self.stats["total_processed"]
        successful = self.stats["successful_cleaned"]
        retweets = self.stats["retweets_filtered"]
        errors = self.stats["processing_errors"]
        
        print("\n" + "="*50)
        print("📊 資料處理品質摘要")
        print("="*50)
        print(f"總處理數量: {total:,}")
        print(f"成功清洗: {successful:,} ({successful/total:.2%})")
        print(f"轉推過濾: {retweets:,} ({retweets/total:.2%})")
        print(f"處理錯誤: {errors:,} ({errors/total:.2%})")
        
        if self.stats["processing_times"]:
            avg_time = sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
            print(f"平均批次處理時間: {avg_time:.2f}秒")
        
        # 顯示前5個加密貨幣符號
        top_crypto = self.stats["special_tokens"]["crypto_symbols"].most_common(5)
        if top_crypto:
            print(f"\n🪙 熱門加密貨幣符號:")
            for symbol, count in top_crypto:
                print(f"  {symbol}: {count}")
        
        print("="*50) 