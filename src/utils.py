"""
Utils 模块 - 通用工具函数
"""

import json
import re


def compact(text: str) -> str:
    """去除多余空格和换行，将文本压缩为紧凑的一行

    支持 JSON 字符串（先解析再紧凑序列化）和普通文本。

    Args:
        text: 输入文本

    Returns:
        紧凑后的单行文本
    """
    text = text.strip()
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    except (json.JSONDecodeError, TypeError):
        return re.sub(r'\s+', ' ', text)
