"""Recall-L metric from Alfa x MFTI case (BERTScore recall x length penalty)."""

from __future__ import annotations

from dataclasses import dataclass

REFUSAL_TEXT = "нет ответа"


def length_multiplier(l_answer: float, l_reference: float) -> float:
    """L(q) from case appendix: 1 up to 1.5*lr, linear to 0 at 3*lr."""
    if l_reference <= 0:
        return 1.0 if l_answer <= 0 else 0.0
    if l_answer <= 1.5 * l_reference:
        return 1.0
    if l_answer >= 3.0 * l_reference:
        return 0.0
    return max(0.0, 1.0 - (l_answer - 1.5 * l_reference) / (1.5 * l_reference))


def is_refusal(text: str) -> bool:
    return (text or "").strip().lower().startswith(REFUSAL_TEXT)


def _token_len(text: str, tokenizer) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    return len(tokenizer.encode(t, add_special_tokens=False))


def bert_recall(pred: str, gold: str, *, lang: str = "ru") -> float:
    from bert_score import score

    _, recall, _ = score(
        [pred or ""],
        [gold or ""],
        lang=lang,
        verbose=False,
        rescale_with_baseline=False,
    )
    return float(recall[0].item())


def recall_l_pair(
    pred: str,
    gold: str,
    tokenizer,
    *,
    lang: str = "ru",
) -> tuple[float, float, float]:
    r = bert_recall(pred, gold, lang=lang)
    la = _token_len(pred, tokenizer)
    lr = _token_len(gold, tokenizer)
    l_mult = length_multiplier(la, lr)
    return r * l_mult, r, l_mult


@dataclass
class RecallLReport:
    recall_l: float
    r_bert_mean: float
    l_mult_mean: float
    n: int
    pred_refusal_rate: float
    gold_refusal_rate: float
    false_refuse: int
    false_answer: int
    answer_only_recall_l: float
    refusal_only_recall_l: float
    answer_only_n: int
    refusal_only_n: int


def recall_l_corpus(
    preds: list[str],
    golds: list[str],
    tokenizer,
    *,
    lang: str = "ru",
    batch_size: int = 32,
) -> RecallLReport:
    from bert_score import score

    if len(preds) != len(golds):
        raise ValueError("preds and golds length mismatch")
    n = len(preds)
    if n == 0:
        return RecallLReport(
            0.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0, 0
        )

    _, recall_t, _ = score(
        [p or "" for p in preds],
        [g or "" for g in golds],
        lang=lang,
        batch_size=batch_size,
        verbose=True,
        rescale_with_baseline=False,
    )
    recalls = recall_t.tolist()
    scores: list[float] = []
    l_mults: list[float] = []
    for pred, gold, r in zip(preds, golds, recalls):
        la = _token_len(pred, tokenizer)
        lr = _token_len(gold, tokenizer)
        lm = length_multiplier(la, lr)
        l_mults.append(lm)
        scores.append(float(r) * lm)

    gold_ref = [is_refusal(g) for g in golds]
    pred_ref = [is_refusal(p) for p in preds]
    false_refuse = sum(1 for g, p in zip(gold_ref, pred_ref) if not g and p)
    false_answer = sum(1 for g, p in zip(gold_ref, pred_ref) if g and not p)

    answer_scores = [s for s, g in zip(scores, gold_ref) if not g]
    refusal_scores = [s for s, g in zip(scores, gold_ref) if g]

    return RecallLReport(
        recall_l=sum(scores) / n,
        r_bert_mean=sum(recalls) / n,
        l_mult_mean=sum(l_mults) / n,
        n=n,
        pred_refusal_rate=sum(pred_ref) / n,
        gold_refusal_rate=sum(gold_ref) / n,
        false_refuse=false_refuse,
        false_answer=false_answer,
        answer_only_recall_l=(
            sum(answer_scores) / len(answer_scores) if answer_scores else 0.0
        ),
        refusal_only_recall_l=(
            sum(refusal_scores) / len(refusal_scores) if refusal_scores else 0.0
        ),
        answer_only_n=len(answer_scores),
        refusal_only_n=len(refusal_scores),
    )
