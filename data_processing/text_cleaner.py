"""
文本清洗核心模組
實作六個主要預處理步驟，針對Twitter推文資料進行專業清洗
"""

import re
import emoji
import spacy
from spacy.tokens import Token
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path


class TextCleaner:
    """文本清洗器：專為Twitter推文設計的文本預處理工具"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文本清洗器
        
        Args:
            config: 配置字典，包含text_cleaning相關設定
        """
        self.config = config["processing"]["text_cleaning"]
        self.logger = logging.getLogger("data_processor.text_cleaner")
        
        # 編譯正則表達式模式
        self._compile_patterns()
        
        # 初始化spaCy模型
        self._setup_spacy()
        
        self.logger.info("TextCleaner 初始化完成")
    
    def _compile_patterns(self):
        """編譯所有正則表達式模式"""
        # URL和Mention模式 (組合模式以提升效能)
        url_pattern = self.config["url_pattern"]
        mention_pattern = self.config["mention_pattern"]
        self.url_mention_pattern = re.compile(
            f"({url_pattern})|({mention_pattern})",
            flags=re.IGNORECASE
        )
        
        # 加密貨幣符號模式
        self.crypto_pattern = re.compile(
            self.config["crypto_symbol_pattern"],
            flags=re.IGNORECASE
        )
        
        # Hashtag模式
        self.hashtag_pattern = re.compile(
            self.config["hashtag_pattern"],
            flags=re.IGNORECASE
        )
        
        # 多空白字元清理
        self.whitespace_pattern = re.compile(r"\s{2,}")
        
        self.logger.debug("正則表達式模式編譯完成")
    
    def _setup_spacy(self):
        """設定spaCy NLP管道"""
        try:
            # 只載入必要組件以提升速度
            self.nlp = spacy.load(
                "en_core_web_sm",
                disable=["parser", "ner", "textcat"]
            )
            
            # 設定自訂tokenizer
            self._setup_custom_tokenizer()
            
            # 添加自訂停用詞
            self._add_custom_stopwords()
            
            self.logger.debug("spaCy模型載入完成")
            
        except OSError:
            self.logger.error("spaCy模型載入失敗，請執行: python -m spacy download en_core_web_sm")
            raise
    
    def _setup_custom_tokenizer(self):
        """設定自訂tokenizer以正確處理加密貨幣符號和連字符"""
        # 在原有prefix中排除$符號
        modified_prefixes = [
            p for p in self.nlp.Defaults.prefixes if p != r'\$'
        ]
        self.nlp.tokenizer.prefix_search = compile_prefix_regex(modified_prefixes).search
        
        # 自訂tokenizer以保留特殊符號
        def custom_tokenizer(nlp):
            # 特殊token模式：加密貨幣符號和連字符詞
            special_patterns = [
                self.crypto_pattern,  # $BTC, $ETH等
                re.compile(r'[A-Za-z]+-[0-9A-Za-z]+'),  # Layer-2, Web3-based等
                re.compile(r':[a-zA-Z_]+:')  # emoji描述 :rocket:
            ]
            
            def special_token_match(text):
                for pattern in special_patterns:
                    match = pattern.match(text)
                    if match:
                        return match
                return None
            
            # 修改infix模式，不在連字符詞中分割
            infix_re = re.compile(r'''[~]''')  # 移除連字符分割
            
            return Tokenizer(
                nlp.vocab,
                prefix_search=nlp.tokenizer.prefix_search,
                suffix_search=nlp.tokenizer.suffix_search,
                infix_finditer=infix_re.finditer,
                token_match=special_token_match
            )
        
        self.nlp.tokenizer = custom_tokenizer(self.nlp)
        self.logger.debug("自訂tokenizer設定完成")
    
    def _add_custom_stopwords(self):
        """添加自訂停用詞"""
        extra_stopwords = set(self.config.get("extra_stopwords", []))
        
        for word in extra_stopwords:
            self.nlp.vocab[word].is_stop = True
        
        self.logger.debug(f"添加 {len(extra_stopwords)} 個自訂停用詞")
    
    # ==================== 六個預處理步驟 ====================
    
    def step1_normalize_and_filter_retweets(self, text: str) -> Optional[str]:
        """
        步驟1: 轉小寫並過濾轉推
        
        Args:
            text: 原始推文文本
            
        Returns:
            處理後的文本，若為轉推則返回None
        """
        if not text or not isinstance(text, str):
            return None
            
        # 檢查是否為轉推
        if self.config.get("remove_retweets", True):
            if text.strip().startswith("RT ") or text.strip().startswith("RT@"):
                return None
        
        # 轉換為小寫
        return text.lower().strip()
    
    def step2_remove_url_mention_preserve_hashtag(self, text: str) -> str:
        """
        步驟2: 移除URL和Mention，保留Hashtag文字
        
        Args:
            text: 輸入文本
            
        Returns:
            清理後的文本
        """
        if not text:
            return ""
        
        # 首先處理hashtag：移除#號但保留文字
        if self.config.get("preserve_hashtags", True):
            text = re.sub(r"#([a-zA-Z0-9_]+)", r" \1", text)
        
        # 移除URL和Mention (更強健的模式)
        # URL模式：匹配http/https/www/pic.twitter.com
        text = re.sub(r"(https?://\S+)|(www\.\S+)|(pic\.twitter\.com/\S+)", "", text, flags=re.IGNORECASE)
        # Mention模式：匹配@username
        text = re.sub(r"@\w+", "", text)
        
        # 清理多餘空白
        text = self.whitespace_pattern.sub(" ", text).strip()
        
        return text
    
    def step3_convert_emoji(self, text: str) -> str:
        """
        步驟3: 將Emoji轉換為文字描述
        
        Args:
            text: 輸入文本
            
        Returns:
            轉換後的文本
        """
        if not text or not self.config.get("convert_emojis", True):
            return text
        
        # 使用emoji.demojize轉換，統一格式
        converted = emoji.demojize(text, language="en", delimiters=(":", ":"))
        # 確保emoji描述前後有單個空格
        converted = re.sub(r"\s*(:[\w_]+:)\s*", r" \1 ", converted)
        # 清理多餘空白
        converted = re.sub(r"\s+", " ", converted).strip()
        
        return converted
    
    def step4_tokenize_with_crypto_symbols(self, text: str) -> List[str]:
        """
        步驟4: 自訂斷詞，保持加密貨幣符號完整
        
        Args:
            text: 輸入文本
            
        Returns:
            Token列表
        """
        if not text:
            return []
        
        try:
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
            return tokens
        except Exception as e:
            self.logger.warning(f"斷詞處理失敗: {e}")
            # 備用方案：簡單空格分割
            return text.split()
    
    def step5_apply_stopwords(self, tokens: List[str]) -> List[str]:
        """
        步驟5: 移除停用詞
        
        Args:
            tokens: Token列表
            
        Returns:
            過濾後的Token列表
        """
        if not tokens:
            return []
        
        filtered_tokens = []
        
        for token in tokens:
            # 檢查是否為停用詞
            spacy_token = self.nlp.vocab[token.lower()]
            
            # 保留特殊符號（加密貨幣符號、emoji描述）
            is_crypto_symbol = self.crypto_pattern.match(token)
            is_emoji_desc = token.startswith(":") and token.endswith(":")
            
            if not spacy_token.is_stop or is_crypto_symbol or is_emoji_desc:
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def step6_lemmatize(self, tokens: List[str]) -> List[str]:
        """
        步驟6: 詞形還原
        
        Args:
            tokens: Token列表
            
        Returns:
            還原後的詞根列表
        """
        if not tokens or not self.config.get("apply_lemmatization", True):
            return tokens
        
        try:
            # 重組文本進行詞形還原
            text = " ".join(tokens)
            doc = self.nlp(text)
            
            lemmatized = []
            for token in doc:
                # 保留特殊符號和emoji描述不進行還原
                if (self.crypto_pattern.match(token.text) or 
                    (token.text.startswith(":") and token.text.endswith(":")) or
                    not token.is_alpha):
                    lemmatized.append(token.text)
                else:
                    lemmatized.append(token.lemma_)
            
            return lemmatized
            
        except Exception as e:
            self.logger.warning(f"詞形還原失敗: {e}")
            return tokens
    
    # ==================== 整合處理函數 ====================
    
    def clean_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        完整的文本清洗流程
        
        Args:
            text: 原始推文文本
            
        Returns:
            清洗結果字典，包含各階段處理結果，失敗則返回None
        """
        if not text or not isinstance(text, str):
            return None
        
        try:
            # 步驟1: 正規化和轉推過濾
            step1_result = self.step1_normalize_and_filter_retweets(text)
            if step1_result is None:
                return None  # 轉推被過濾
            
            # 步驟2: URL/Mention移除，Hashtag保留
            step2_result = self.step2_remove_url_mention_preserve_hashtag(step1_result)
            
            # 步驟3: Emoji轉換
            step3_result = self.step3_convert_emoji(step2_result)
            
            # 步驟4: 自訂斷詞
            step4_result = self.step4_tokenize_with_crypto_symbols(step3_result)
            
            # 步驟5: 停用詞過濾
            step5_result = self.step5_apply_stopwords(step4_result)
            
            # 步驟6: 詞形還原
            step6_result = self.step6_lemmatize(step5_result)
            
            # 品質檢查
            if len(step6_result) == 0 and not self.config.get("allow_empty_after_cleaning", False):
                return None
            
            # 組裝結果
            result = {
                "original_text": text,
                "normalized_text": step1_result,
                "cleaned_text": step3_result,  # 保留emoji轉換後的可讀文本
                "tokens": step4_result,
                "filtered_tokens": step5_result,
                "final_tokens": step6_result,
                "token_count": len(step6_result),
                "processing_steps": {
                    "retweet_filtered": step1_result != text.lower(),
                    "urls_mentions_removed": step2_result != step1_result,
                    "emojis_converted": step3_result != step2_result,
                    "stopwords_removed": len(step5_result) < len(step4_result),
                    "lemmatized": step6_result != step5_result
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"文本清洗失敗: {text[:100]}... 錯誤: {e}")
            return None
    
    def batch_clean(self, texts: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        批次處理文本清洗
        
        Args:
            texts: 文本列表
            
        Returns:
            清洗結果列表
        """
        results = []
        
        for text in texts:
            result = self.clean_text(text)
            results.append(result)
        
        return results
    
    # ==================== 驗證和統計 ====================
    
    def validate_setup(self) -> Dict[str, bool]:
        """驗證清洗器設定是否正確"""
        test_cases = [
            ("$BTC is going to the moon! 🚀 #crypto", True),   # 應該成功處理
            ("RT @user: This is a retweet", False),            # 應該被過濾 (返回None)
            ("Check out https://example.com @mention #DeFi", True),  # 應該成功處理
            ("Layer-2 solutions are amazing 💯", True)        # 應該成功處理
        ]
        
        results = {}
        
        try:
            # 測試基本功能
            for i, (test_text, should_succeed) in enumerate(test_cases, 1):
                result = self.clean_text(test_text)
                actual_success = result is not None
                results[f"test_case_{i}"] = actual_success == should_succeed
            
            # 測試特殊token
            tokens = self.step4_tokenize_with_crypto_symbols("$BTC Layer-2 test")
            results["crypto_symbol_preserved"] = "$BTC" in tokens
            results["hyphen_preserved"] = "Layer-2" in tokens
            
            self.logger.info("設定驗證完成")
            return results
            
        except Exception as e:
            self.logger.error(f"設定驗證失敗: {e}")
            return {"validation_failed": True, "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """取得清洗器統計資訊"""
        return {
            "config": self.config,
            "spacy_model": self.nlp.meta["name"] if hasattr(self.nlp, 'meta') else "unknown",
            "custom_stopwords_count": len(self.config.get("extra_stopwords", [])),
            "patterns_compiled": {
                "url_mention": bool(self.url_mention_pattern),
                "crypto_symbol": bool(self.crypto_pattern),
                "hashtag": bool(self.hashtag_pattern)
            }
        } 