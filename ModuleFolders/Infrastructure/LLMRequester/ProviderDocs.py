"""Provider 文档：解析、外链校验与展示条目。

文档内容放在 Resource/platforms/preset.json 的 docs 块里，所以更新说明只需要
替换一个数据文件，不必重新打包 —— 与选项 schema 走同一条热补丁路径。

安全边界（来自技术评审团的结论）：docs 里的链接会被渲染成可点击外链，而
preset.json 恰恰是被鼓励 drop-in 替换的文件。因此：

* 只接受 https 且必须有 host，file: / javascript: / 自定义 scheme 一律丢弃；
* notes 只按纯文本渲染，不做任何 HTML 插值，避免数据文件变成注入面。

本模块不导入 PyQt5，纯 unittest 即可覆盖上述规则。
"""

from urllib.parse import urlparse

from ModuleFolders.Infrastructure.LLMRequester.OptionSchema import (
    localized_text,
    resolve_preset_key,
)

# 展示顺序与各自的标题；标题自带语言映射，不走 tra()（理由同选项标签）
DOC_LINK_FIELDS = (
    ("api_key_url", {"简中": "获取 API Key", "English": "Get an API key"}),
    ("docs_url", {"简中": "官方文档", "English": "Official docs"}),
    ("models_url", {"简中": "模型列表", "English": "Model list"}),
)

# 允许的 scheme。http 也不放行：官方文档站没有必须用明文 http 的理由。
ALLOWED_URL_SCHEME = "https"


def safe_external_url(url) -> str:
    """返回可安全交给系统浏览器打开的 URL，否则返回空串。

    只做 scheme 与 host 校验：这里不是域名白名单，而是防止数据文件把
    file:// 或自定义 scheme 塞进应用自己的可信界面。
    """
    if not isinstance(url, str):
        return ""
    text = url.strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except ValueError:
        return ""
    if parsed.scheme.lower() != ALLOWED_URL_SCHEME or not parsed.netloc:
        return ""
    return text


def resolve_docs(platform_config, preset_platforms) -> dict:
    """取该平台声明的文档块；没有就返回空字典。"""
    preset_key = resolve_preset_key(platform_config or {}, preset_platforms or {})
    if preset_key is None:
        return {}
    docs = (preset_platforms or {}).get(preset_key, {}).get("docs")
    return docs if isinstance(docs, dict) else {}


def doc_links(docs, language: str = "简中") -> list:
    """返回 [(字段, 标题, 安全 URL)]，非法或缺失的链接直接不出现在列表里。"""
    links = []
    for field, label in DOC_LINK_FIELDS:
        url = safe_external_url((docs or {}).get(field))
        if not url:
            continue
        links.append((field, localized_text(label, language, field), url))
    return links


def doc_notes(docs, language: str = "简中") -> str:
    """返回说明正文（纯文本）。"""
    return localized_text((docs or {}).get("notes"), language, "")


def has_docs(docs) -> bool:
    """是否有值得展示的内容 —— 决定界面要不要出文档卡片。"""
    return bool(doc_links(docs)) or bool(doc_notes(docs))
