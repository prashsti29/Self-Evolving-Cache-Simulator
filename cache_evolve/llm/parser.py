from __future__ import annotations

import re


def extract_python_class(source: str) -> str:
    text = source.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    match = re.search(r"class\s+\w+\s*\(.*Policy.*\):", text)
    if match:
        start = text.rfind("\n", 0, match.start())
        start = 0 if start < 0 else start + 1
        return text[start:].strip()
    return text
