"""
共用工具函數
配置載入、日誌設定、檔案處理等基礎功能
"""

import yaml
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
import orjson
from datetime import datetime


def load_config(config_path: str = "data_processing/config.yaml") -> Dict[str, Any]:
    """載入YAML配置文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件格式錯誤: {e}")


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """設定日誌系統"""
    log_config = config.get("logging", {})
    
    # 創建logs目錄
    log_dir = Path(config["processing"]["paths"]["logs"])
    log_dir.mkdir(exist_ok=True)
    
    # 設定日誌格式
    formatter = logging.Formatter(
        log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # 創建logger
    logger = logging.getLogger("data_processor")
    logger.setLevel(getattr(logging, log_config.get("level", "INFO")))
    
    # 清除既有的handlers
    logger.handlers.clear()
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 檔案handler (支援輪轉)
    if log_config.get("file_rotation", True):
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"data_processing_{datetime.now().strftime('%Y%m%d')}.log",
            maxBytes=_parse_size(log_config.get("max_file_size", "10MB")),
            backupCount=log_config.get("backup_count", 5),
            encoding="utf-8"
        )
    else:
        file_handler = logging.FileHandler(
            log_dir / f"data_processing_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8"
        )
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def _parse_size(size_str: str) -> int:
    """解析檔案大小字串 (如 '10MB') 為位元組數"""
    size_str = size_str.upper()
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


def safe_json_load(file_path: Path) -> Optional[list]:
    """安全載入JSON檔案，處理格式錯誤"""
    try:
        with open(file_path, "rb") as f:
            return orjson.loads(f.read())
    except orjson.JSONDecodeError as e:
        logging.error(f"JSON解析錯誤 {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"檔案讀取錯誤 {file_path}: {e}")
        return None


def safe_json_save(data: Any, file_path: Path) -> bool:
    """安全儲存JSON檔案"""
    try:
        # 確保目錄存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        return True
    except Exception as e:
        logging.error(f"JSON儲存錯誤 {file_path}: {e}")
        return False


def generate_uuid_from_tweet_id(tweet_id: str, prefix: str = "tweet") -> str:
    """從推文ID生成一致性UUID"""
    import hashlib
    
    # 使用SHA256確保一致性
    hash_object = hashlib.sha256(f"{prefix}_{tweet_id}".encode())
    return f"{prefix}_{hash_object.hexdigest()[:16]}"


def validate_file_structure(file_path: Path, required_fields: list) -> bool:
    """驗證JSON檔案結構"""
    data = safe_json_load(file_path)
    if not data:
        return False
        
    if not isinstance(data, list) or len(data) == 0:
        return False
        
    # 檢查第一筆資料是否包含必要欄位
    first_item = data[0]
    for field in required_fields:
        if field not in first_item:
            logging.warning(f"檔案 {file_path} 缺少必要欄位: {field}")
            return False
    
    return True


def get_file_stats(file_path: Path) -> Dict[str, Any]:
    """取得檔案基本統計資訊"""
    if not file_path.exists():
        return {"exists": False}
        
    stat = file_path.stat()
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat()
    } 