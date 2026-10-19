"""界面语言。默认中文，英文是并列的一套文案，不改变估算数字。"""


def norm_lang(lang: str | None) -> str:
    return "en" if lang == "en" else "zh"


def t(lang: str | None, zh: str, en: str) -> str:
    return en if norm_lang(lang) == "en" else zh


def sim(lang: str | None) -> str:
    return "[sim]" if norm_lang(lang) == "en" else "[模拟]"
