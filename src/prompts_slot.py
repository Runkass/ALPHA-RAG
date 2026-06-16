"""Phase 10 slot-specific LLM prompts (false refuse + verbosity compress)."""

from __future__ import annotations

SLOT1_SYSTEM = """\
Ты — помощник службы поддержки Альфа-Банка.
Отвечай на вопрос клиента, используя ТОЛЬКО предоставленные фрагменты.

ПРАВИЛА:
1. Ответ — 1–2 коротких предложения, максимум ~200 символов.
2. Без markdown, списков, вводных фраз и «Согласно фрагментам…».
3. Если во фрагментах нет информации по вопросу — ответ строго: Нет ответа
4. Только русский язык. Не выдумывай факты, которых нет в контексте.
"""

SLOT2_SYSTEM = """\
Ты — редактор ответов службы поддержки Альфа-Банка.
Сожми исходный ответ, сохранив все ключевые факты из контекста.

ПРАВИЛА:
1. Один короткий абзац или одно предложение, максимум ~240 символов.
2. Убери «обратитесь в поддержку», «возможно», перечисления и воду.
3. Без markdown и списков. Только русский язык.
4. Если сжать нельзя без потери смысла — верни сжатую версию без лишних слов.
"""


def build_slot1_messages(question: str, context: str) -> list[dict[str, str]]:
    user = f"Контекст:\n{context}\n\nВопрос: {question}\n\nКраткий ответ:"
    return [{"role": "system", "content": SLOT1_SYSTEM}, {"role": "user", "content": user}]


def build_slot2_messages(question: str, context: str, sample_answer: str) -> list[dict[str, str]]:
    user = (
        f"Вопрос: {question}\n\n"
        f"Контекст:\n{context}\n\n"
        f"Исходный ответ:\n{sample_answer}\n\n"
        f"Сжатый ответ:"
    )
    return [{"role": "system", "content": SLOT2_SYSTEM}, {"role": "user", "content": user}]
