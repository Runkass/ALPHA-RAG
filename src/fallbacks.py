"""Rule-based and extractive fallbacks when LLM refuses or retrieval is weak."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

REFUSE_PHRASES = (
    "нет ответа",
    "не могу ответить",
    "информация отсутствует",
    "не найдено",
    "не удалось найти",
    "отсутствует в контексте",
    "к сожалению",
)

_RULE_FALLBACKS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i).*бик.*(?:банк|альфа|альфа-банк).*"), "БИК Альфа-Банка: 044525593."),
    (re.compile(r"(?i).*бик\b"), "БИК Альфа-Банка: 044525593."),
    (re.compile(r"(?i).*реквизит.*сч[её]т"), "Реквизиты счёта доступны в интернет-банке: раздел «Счета» → «Реквизиты»."),
    (re.compile(r"(?i).*номер.*сч[её]т|сч[её]т.*номер"), "Номер счёта — в мобильном приложении или Альфа-Онлайн, раздел «Счета»."),
    (re.compile(r"(?i).*(?:когда|срок).*(?:кэшбэк|кешбэк)|(?:кэшбэк|кешбэк).*(?:когда|начисл)"), "Кэшбэк начисляется до 10 числа следующего месяца."),
    (re.compile(r"(?i).*(?:кэшбэк|кешбэк).*(?:заправк|азс|топлив)"), "Кэшбэк за заправку начисляется в категории «Транспорт» или «Сервисы заправки»."),
    (re.compile(r"(?i).*(?:пин|pin).*(?:код|карт)"), "ПИН-код выдаётся в конверте при получении карты. Для восстановления обратитесь в отделение."),
    (re.compile(r"(?i).*(?:график|расписан).*(?:платеж|платёж)"), "График платежей доступен в мобильном приложении в разделе «Кредиты»."),
    (re.compile(r"(?i).*(?:просроч|задолж).*(?:что делать|как)"), "Рекомендуем внести минимальный платёж как можно скорее и связаться с банком для обсуждения вариантов."),
    (re.compile(r"(?i).*(?:закрыть|закрыт).*(?:сч[её]т|карт)"), "Закрытие счёта или карты оформляется в отделении банка или через мобильное приложение."),
    (re.compile(r"(?i).*(?:лимит|ограничен).*(?:карт|сч[её]т|операц)"), "Лимиты по карте и счёту настраиваются в мобильном приложении или Альфа-Онлайн."),
    (re.compile(r"(?i).*(?:мобильн|приложен).*(?:скачать|установ)"), "Мобильное приложение Альфа-Банка доступно в App Store и Google Play."),
    (re.compile(r"(?i).*(?:горяч|телефон).*(?:линия|поддержк)"), "Служба поддержки: 8 800 200-00-00 (бесплатно по России)."),
    (re.compile(r"(?i).*(?:курс|валют).*(?:доллар|евро|usd|eur)"), "Актуальные курсы валют — в мобильном приложении и на сайте банка."),
    (re.compile(r"(?i).*(?:ипотек|кредит).*(?:ставк|процент)"), "Ставки по кредитам и ипотеке указаны на сайте банка и в отделениях."),
]


def try_rule_answer(query: str, _snippet: str = "") -> str | None:
    for pat, ans in _RULE_FALLBACKS:
        if pat.search(query):
            return ans
    return None


def is_refusal(answer: str) -> bool:
    low = answer.strip().lower()
    return not low or any(p in low for p in REFUSE_PHRASES)


def _trim_sentence(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    cut = text[: max_len - 3]
    last = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind(" "))
    if last > max_len // 3:
        cut = cut[: last + 1].rstrip()
    return cut + "..."


def extractive_fallback(chunks: list[RetrievedChunk], *, max_len: int | None = None) -> str:
    if not chunks:
        return "Нет ответа"
    if max_len is None:
        from .config import get_settings

        max_len = get_settings().soft_fallback_max_len
    text = _trim_sentence(chunks[0].text, max_len)
    return text if text else "Нет ответа"


def soft_fallback(answer: str, chunks: list[RetrievedChunk], *, max_len: int | None = None) -> str:
    if max_len is None:
        from .config import get_settings
        max_len = get_settings().soft_fallback_max_len
    if not is_refusal(answer):
        return answer
    if not chunks:
        return answer
    text = _trim_sentence(chunks[0].text, max_len)
    if not text:
        return answer
    logger.debug("soft_fallback applied, chunk_len=%d", len(text))
    return text