from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Callable

DEFAULT_UI_LANGUAGE = "tr"
DEFAULT_ANSWER_ABS_TOL = 0.05
QUIZ_SLOT_COUNT = 5

Question = dict[str, Any]
QuestionBuilder = Callable[[random.Random, bool, float], Question]
QuestionValidator = Callable[[Question], None]
QuestionDefinition = dict[str, Any]


class QuestionValidationError(ValueError):
    """Raised when a generated question breaks its mathematical invariants."""


def _approx_equal(left: float, right: float, tol: float = 1e-9) -> bool:
    return abs(left - right) <= tol


def _fail_validation(question_id: str, detail: str) -> None:
    raise QuestionValidationError(f"{question_id}: {detail}")


def _ensure(condition: bool, question_id: str, detail: str) -> None:
    if not condition:
        _fail_validation(question_id, detail)


def _as_float(value: Any, question_id: str, label: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise QuestionValidationError(f"{question_id}: {label} must be numeric") from exc
    _ensure(math.isfinite(normalized), question_id, f"{label} must be finite")
    return normalized


def _ensure_probability(value: float, question_id: str, label: str) -> float:
    normalized = _as_float(value, question_id, label)
    _ensure(0.0 <= normalized <= 1.0, question_id, f"{label} must stay in [0, 1]")
    return normalized


def _require_visual(question_id: str, question: Question) -> dict[str, Any]:
    visual = question.get("visual")
    _ensure(isinstance(visual, dict), question_id, "visual must be a dict")
    return visual


def _require_visual_kind(question_id: str, question: Question, *allowed_kinds: str) -> dict[str, Any]:
    visual = _require_visual(question_id, question)
    kind = str(visual.get("kind") or "").strip()
    _ensure(kind in allowed_kinds, question_id, f"visual.kind must be one of {allowed_kinds}, got {kind!r}")
    return visual


def _require_numeric_sequence(question_id: str, container: dict[str, Any], key: str) -> list[float]:
    raw_values = container.get(key)
    _ensure(isinstance(raw_values, (list, tuple)) and len(raw_values) > 0, question_id, f"{key} must be a non-empty list")
    return [_as_float(value, question_id, f"{key}[{idx}]") for idx, value in enumerate(raw_values)]


def _require_string_sequence(question_id: str, container: dict[str, Any], key: str) -> list[str]:
    raw_values = container.get(key)
    _ensure(isinstance(raw_values, (list, tuple)) and len(raw_values) > 0, question_id, f"{key} must be a non-empty list")
    normalized = [str(value).strip() for value in raw_values]
    _ensure(all(normalized), question_id, f"{key} entries must be non-empty strings")
    return normalized


def _validate_question_shell(question_id: str, question: Question) -> float:
    _ensure(isinstance(question, dict), question_id, "question must be a dict")
    _ensure(str(question.get("title") or "").strip() != "", question_id, "title must be non-empty")
    _ensure(str(question.get("text") or "").strip() != "", question_id, "text must be non-empty")
    tolerance = _as_float(question.get("tolerance", DEFAULT_ANSWER_ABS_TOL), question_id, "tolerance")
    _ensure(0.0 <= tolerance <= 1.0, question_id, "tolerance must stay in [0, 1]")
    return _as_float(question.get("answer"), question_id, "answer")


def _validate_pmf_visual(question_id: str, question: Question) -> tuple[float, list[float], list[float], dict[str, Any]]:
    answer = _validate_question_shell(question_id, question)
    visual = _require_visual_kind(question_id, question, "pmf_table", "pmf_bar")
    x_values = _require_numeric_sequence(question_id, visual, "x_values")
    p_values = _require_numeric_sequence(question_id, visual, "p_values")
    _ensure(len(x_values) == len(p_values), question_id, "x_values and p_values must have the same length")
    _ensure(len(set(x_values)) == len(x_values), question_id, "x_values must be unique")
    for idx, value in enumerate(p_values):
        _ensure_probability(value, question_id, f"p_values[{idx}]")
    _ensure(_approx_equal(sum(p_values), 1.0), question_id, "PMF probabilities must sum to 1")
    return answer, x_values, p_values, visual


def _validate_probability_tree_visual(
    question_id: str,
    question: Question,
) -> tuple[float, list[str], list[float], list[float], dict[str, Any]]:
    answer = _validate_question_shell(question_id, question)
    visual = _require_visual_kind(question_id, question, "probability_tree")
    sources = _require_string_sequence(question_id, visual, "sources")
    priors = _require_numeric_sequence(question_id, visual, "priors")
    cond_event = _require_numeric_sequence(question_id, visual, "cond_event")
    _ensure(len(sources) == len(priors) == len(cond_event), question_id, "tree lists must have the same length")
    for idx, value in enumerate(priors):
        _ensure_probability(value, question_id, f"priors[{idx}]")
    for idx, value in enumerate(cond_event):
        _ensure_probability(value, question_id, f"cond_event[{idx}]")
    _ensure(_approx_equal(sum(priors), 1.0), question_id, "prior probabilities must sum to 1")
    return answer, sources, priors, cond_event, visual


def _validate_integer_support(question_id: str, x_values: list[float]) -> list[int]:
    ints = [int(round(value)) for value in x_values]
    for idx, value in enumerate(x_values):
        _ensure(_approx_equal(value, ints[idx]), question_id, f"x_values[{idx}] must be an integer")
    return ints


def _question_definition(
    question_id: str,
    name_tr: str,
    name_en: str,
    builder: QuestionBuilder,
    validator: QuestionValidator,
) -> QuestionDefinition:
    return {
        "id": question_id,
        "name_tr": name_tr,
        "name_en": name_en,
        "builder": builder,
        "validator": validator,
    }


def _normalize_language(language: str) -> str:
    normalized = str(language or "").strip().lower()
    return "en" if normalized == "en" else "tr"


def _sanitize_answer_tolerance(value: float) -> float:
    """Clamp tolerance into [0, 1] for safe scoring."""
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return DEFAULT_ANSWER_ABS_TOL
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return normalized


def rng_from_student(student_id: str, quiz_session: str) -> random.Random:
    """Build deterministic RNG for the same student and quiz session."""
    digest = hashlib.sha256(f"{student_id.strip()}::{quiz_session.strip()}".encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    return random.Random(seed)


def _weights_to_probabilities(weights: tuple[int, ...]) -> list[float]:
    total = sum(weights)
    return [weight / total for weight in weights]


def _standard_normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_cdf(x: float, mean: float, std_dev: float) -> float:
    return _standard_normal_cdf((x - mean) / std_dev)


def _erlang_survival(shape: int, rate: float, threshold: float) -> float:
    return math.exp(-rate * threshold) * sum(((rate * threshold) ** idx) / math.factorial(idx) for idx in range(shape))


def _factory_machine_scenario(r: random.Random) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    shares = r.choice(
        [
            (0.20, 0.35, 0.45),
            (0.25, 0.35, 0.40),
            (0.25, 0.40, 0.35),
            (0.30, 0.30, 0.40),
            (0.30, 0.45, 0.25),
            (0.35, 0.25, 0.40),
            (0.40, 0.35, 0.25),
        ]
    )
    defect_rates = r.choice(
        [
            (0.01, 0.02, 0.03),
            (0.01, 0.03, 0.04),
            (0.02, 0.03, 0.04),
            (0.02, 0.04, 0.03),
            (0.03, 0.02, 0.05),
            (0.03, 0.04, 0.02),
            (0.01, 0.04, 0.05),
        ]
    )
    return shares, defect_rates


def _software_error_distribution(r: random.Random) -> tuple[int, list[int], list[float]]:
    code_lines = r.choice([80, 100, 120, 150])
    x_start = r.choice([1, 2, 3])
    weights = r.choice(
        [
            (4, 18, 34, 28, 16),
            (6, 20, 32, 26, 16),
            (5, 22, 31, 27, 15),
            (8, 18, 30, 28, 16),
            (7, 19, 33, 25, 16),
            (3, 17, 36, 29, 15),
        ]
    )
    x_values = list(range(x_start, x_start + len(weights)))
    return code_lines, x_values, _weights_to_probabilities(weights)


def _fabric_defect_distribution(r: random.Random) -> tuple[int, list[int], list[float]]:
    scenarios = [
        (8, [0, 1, 2, 3, 4], (46, 28, 15, 7, 4)),
        (8, [0, 1, 2, 3, 4, 5], (34, 27, 18, 11, 6, 4)),
        (10, [0, 1, 2, 3, 4, 5], (31, 26, 20, 11, 7, 5)),
        (10, [0, 1, 2, 3, 4, 5], (29, 24, 21, 13, 8, 5)),
        (12, [0, 1, 2, 3, 4, 5], (28, 23, 20, 14, 9, 6)),
        (12, [0, 1, 2, 3, 4, 5, 6], (24, 22, 19, 14, 10, 7, 4)),
        (15, [0, 1, 2, 3, 4, 5, 6], (22, 20, 18, 15, 11, 8, 6)),
        (15, [0, 1, 2, 3, 4, 5, 6], (19, 21, 18, 16, 12, 8, 6)),
    ]
    fabric_length, x_values, weights = r.choice(scenarios)
    return fabric_length, x_values, _weights_to_probabilities(weights)


def _build_q_backup_power_union(r: random.Random, is_en: bool, tolerance: float) -> Question:
    p_a, p_b = r.choice(
        [
            (0.90, 0.80),
            (0.88, 0.82),
            (0.92, 0.78),
            (0.86, 0.84),
        ]
    )
    answer = 1 - (1 - p_a) * (1 - p_b)
    return {
        "title": "Backup Power Unit" if is_en else "Yedek Güç Ünitesi",
        "text": (
            (
                f"Generator A and B are independent.\nP(A)={p_a:.2f}, P(B)={p_b:.2f}.\n"
                "Find the probability that at least one generator works: P(A union B)."
            )
            if is_en
            else (
                f"Jeneratör A ve B bağımsızdır.\nP(A)={p_a:.2f}, P(B)={p_b:.2f}.\n"
                "En az bir jeneratörün çalışma olasılığını bulunuz: P(A birleşim B)."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "visual": {
            "kind": "venn_prob",
            "left_label": "A",
            "right_label": "B",
            "left": p_a,
            "right": p_b,
            "both": p_a * p_b,
        },
    }


def _build_q_hotel_chain_total_probability(r: random.Random, is_en: bool, tolerance: float) -> Question:
    p_r, p_s, p_l, f_r, f_s, f_l = r.choice(
        [
            (0.20, 0.50, 0.30, 0.05, 0.04, 0.08),
            (0.25, 0.45, 0.30, 0.04, 0.05, 0.07),
            (0.30, 0.40, 0.30, 0.03, 0.05, 0.06),
        ]
    )
    answer = p_r * f_r + p_s * f_s + p_l * f_l
    return {
        "title": "Hotel Chain - Total Probability" if is_en else "Otel Zinciri - Toplam Olasılık",
        "text": (
            (
                "A firm sends guests to three hotels.\n"
                f"P(R)=Ramada={p_r:.2f}, P(S)=Sheraton={p_s:.2f}, P(L)=Lakeview={p_l:.2f}.\n"
                f"P(F|R)={f_r:.2f}, P(F|S)={f_s:.2f}, P(F|L)={f_l:.2f}.\n"
                "Find overall faulty-plumbing probability P(F)."
            )
            if is_en
            else (
                "Bir firma konuklarını üç otele gönderiyor.\n"
                f"P(R)=Ramada={p_r:.2f}, P(S)=Sheraton={p_s:.2f}, P(L)=Lakeview={p_l:.2f}.\n"
                f"P(F|R)={f_r:.2f}, P(F|S)={f_s:.2f}, P(F|L)={f_l:.2f}.\n"
                "Toplam tesisat arızası olasılığını bulunuz: P(F)."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "visual": {
            "kind": "probability_tree",
            "sources": ["R", "S", "L"],
            "priors": [p_r, p_s, p_l],
            "event_label": "F",
            "cond_event": [f_r, f_s, f_l],
        },
    }


def _build_q_medical_test_bayes(r: random.Random, is_en: bool, tolerance: float) -> Question:
    p_h, p_pos_h, p_pos_hc = r.choice(
        [
            (0.010, 0.98, 0.05),
            (0.015, 0.97, 0.04),
            (0.020, 0.96, 0.05),
            (0.030, 0.95, 0.06),
        ]
    )
    p_pos = p_pos_h * p_h + p_pos_hc * (1 - p_h)
    answer = (p_pos_h * p_h) / p_pos
    return {
        "title": "Medical Diagnosis - Bayes" if is_en else "Tıbbi Tanı - Bayes",
        "text": (
            (
                f"Prevalence P(H)={p_h:.3f}, sensitivity P(+|H)={p_pos_h:.2f}, false positive P(+|H')={p_pos_hc:.2f}.\n"
                "A person tested positive. Find P(H|+)."
            )
            if is_en
            else (
                f"Yaygınlık P(H)={p_h:.3f}, duyarlılık P(+|H)={p_pos_h:.2f}, yanlış pozitif P(+|H')={p_pos_hc:.2f}.\n"
                "Kişi testte pozitif çıktı. P(H|+) değerini bulunuz."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "visual": {
            "kind": "probability_tree",
            "sources": ["H", "H'"],
            "priors": [p_h, 1 - p_h],
            "event_label": "+",
            "cond_event": [p_pos_h, p_pos_hc],
        },
    }


def _build_q_tv_sets_p_ge_1(r: random.Random, is_en: bool, tolerance: float) -> Question:
    total, defective, draw = r.choice(
        [
            (7, 2, 3),
            (8, 2, 3),
            (9, 3, 4),
            (10, 3, 4),
        ]
    )
    non_defective = total - defective
    max_x = min(defective, draw)
    x_values = list(range(0, max_x + 1))
    probs = [
        (math.comb(defective, x) * math.comb(non_defective, draw - x)) / math.comb(total, draw)
        for x in x_values
    ]
    p_zero = probs[0]
    answer = 1 - p_zero
    return {
        "title": "Television Sets - PMF/CDF" if is_en else "Televizyon Setleri - PMF/CDF",
        "text": (
            (
                f"A hotel purchasing department randomly acquires {draw} televisions from a warehouse shipment of {total} TVs, where {defective} units are unfortunately defective.\n"
                "Let X be the number of defective TVs received by the hotel. Find the probability that the hotel receives at least one defective unit: P(X>=1)."
            )
            if is_en
            else (
                f"Bir otel işletmesi, misafir odalarına yerleştirmek üzere deposunda {defective} tane arızalı panel bulunan {total} adetlik bir stoktan rastgele {draw} adet televizyon almıştır.\n"
                "X, otelin teslim aldığı arızalı televizyon sayısı olduğuna göre, otele ulaşan televizyonlar içinde en az bir tane arızalı TV çıkma olasılığını hesaplayınız: P(X>=1)."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "visual": {
            "kind": "pmf_bar",
            "x_values": x_values,
            "p_values": probs,
            "population_size": total,
            "defective_count": defective,
            "draw_count": draw,
            "caption": "Dağılım Şekli" if not is_en else "Distribution Shape",
            "show_prob_labels": False,
            "show_y_axis_values": False,
        },
    }


def _build_q_software_bug_variance(r: random.Random, is_en: bool, tolerance: float) -> Question:
    code_lines, x_values, p_values = _software_error_distribution(r)
    probs = list(zip(x_values, p_values))
    ex = sum(x * p for x, p in probs)
    ex2 = sum((x**2) * p for x, p in probs)
    answer = ex2 - ex * ex
    return {
        "title": "Software Error Analysis - Variance" if is_en else "Yazılım Hata Analizi - Varyans",
        "text": (
            (
                f"The PMF table below presents the distribution of software bugs (X) detected by the testing team in a {code_lines}-line code module.\n"
                "Using the formula Var(X)=E(X^2)-[E(X)]^2, calculate the variance Var(X)."
            )
            if is_en
            else (
                f"{code_lines} satırlık bir yazılım modülünde test ekibi tarafından tespit edilen yazılım hatası (bug) sayısı X'in PMF tablosu aşağıda verilmiştir.\n"
                "Var(X)=E(X^2)-[E(X)]^2 formülünü kullanarak hataların varyansını hesaplayınız."
            )
        ),
        "answer": answer,
        "tolerance": tolerance,
        "visual": {
            "kind": "pmf_table",
            "x_values": x_values,
            "p_values": p_values,
            "caption": "PMF: Yazılım hatası sayısı" if not is_en else "PMF: Software error count",
        },
    }


def _build_q_factory_total_probability(r: random.Random, is_en: bool, tolerance: float) -> Question:
    (p_b1, p_b2, p_b3), (p_a_b1, p_a_b2, p_a_b3) = _factory_machine_scenario(r)
    answer = p_b1 * p_a_b1 + p_b2 * p_a_b2 + p_b3 * p_a_b3
    return {
        "title": "Assembly Factory - Total Probability" if is_en else "Montaj Fabrikası - Toplam Olasılık",
        "text": (
            (
                f"Machine shares: P(B1)={p_b1:.2f}, P(B2)={p_b2:.2f}, P(B3)={p_b3:.2f}.\n"
                f"Defect rates: P(A|B1)={p_a_b1:.2f}, P(A|B2)={p_a_b2:.2f}, P(A|B3)={p_a_b3:.2f}.\n"
                "Find P(A) with Total Probability."
            )
            if is_en
            else (
                f"Makine payları: P(B1)={p_b1:.2f}, P(B2)={p_b2:.2f}, P(B3)={p_b3:.2f}.\n"
                f"Hata oranları: P(A|B1)={p_a_b1:.2f}, P(A|B2)={p_a_b2:.2f}, P(A|B3)={p_a_b3:.2f}.\n"
                "Toplam Olasılık ile P(A) değerini bulunuz."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "visual": {
            "kind": "probability_tree",
            "sources": ["B1", "B2", "B3"],
            "priors": [p_b1, p_b2, p_b3],
            "event_label": "A",
            "cond_event": [p_a_b1, p_a_b2, p_a_b3],
        },
    }


def _build_q_factory_bayes_machine3(r: random.Random, is_en: bool, tolerance: float) -> Question:
    (p_b1, p_b2, p_b3), (p_a_b1, p_a_b2, p_a_b3) = _factory_machine_scenario(r)
    p_a = p_b1 * p_a_b1 + p_b2 * p_a_b2 + p_b3 * p_a_b3
    answer = (p_b3 * p_a_b3) / p_a
    return {
        "title": "Defect Source - Bayes" if is_en else "Hata Kaynağı - Bayes",
        "text": (
            (
                f"Assembly factory data: P(B1)={p_b1:.2f}, P(B2)={p_b2:.2f}, P(B3)={p_b3:.2f} "
                f"and P(A|Bi)=({p_a_b1:.2f},{p_a_b2:.2f},{p_a_b3:.2f}).\n"
                "Given product is defective (A), find P(B3|A)."
            )
            if is_en
            else (
                f"Montaj fabrikası verisi: P(B1)={p_b1:.2f}, P(B2)={p_b2:.2f}, P(B3)={p_b3:.2f} "
                f"ve P(A|Bi)=({p_a_b1:.2f},{p_a_b2:.2f},{p_a_b3:.2f}).\n"
                "Ürün hatalı (A) veriliyor. P(B3|A) değerini bulunuz."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "visual": {
            "kind": "probability_tree",
            "sources": ["B1", "B2", "B3"],
            "priors": [p_b1, p_b2, p_b3],
            "event_label": "A",
            "cond_event": [p_a_b1, p_a_b2, p_a_b3],
        },
    }


def _build_q_airbag_pmf_exact_two(r: random.Random, is_en: bool, tolerance: float) -> Question:
    n = r.choice([4, 5])
    p = 0.50
    k = 2
    x_values = list(range(0, n + 1))
    probs = [math.comb(n, x) * (p**x) * ((1 - p) ** (n - x)) for x in x_values]
    answer = probs[k]
    return {
        "title": "Airbag Sales - PMF" if is_en else "Hava Yastığı Satışı - PMF",
        "text": (
            (
                f"Out of {n} independent car buyers at a dealership, each has a 0.50 probability of opting for the 'extra safety package' (airbags).\n"
                "Let X be the number of cars sold with this package today. Calculate the probability that exactly 2 customers purchase the upgrade: P(X=2)."
            )
            if is_en
            else (
                f"Bir otomobil galerisine gelen {n} bağımsız müşterinin her birinin satın aldığı araca ek güvenlik paketi (hava yastıkları) ekletme olasılığı 0.50'dir.\n"
                "X, gün sonundaki hava yastıklı araç satış adedi olduğuna göre, müşterilerden tam olarak 2 tanesinin bu paketi tercih etme olasılığını bulunuz: P(X=2)."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "visual": {
            "kind": "pmf_bar",
            "x_values": x_values,
            "p_values": probs,
            "highlight_x": k,
            "trial_count": n,
            "success_prob": p,
            "caption": "Dağılım Şekli" if not is_en else "Distribution Shape",
            "show_prob_labels": False,
            "show_y_axis_values": False,
            "show_highlight": False,
        },
    }


def _build_q_vacuum_pdf_prob(r: random.Random, is_en: bool, tolerance: float) -> Question:
    threshold = r.choice([1.10, 1.20, 1.30, 1.40])
    answer = 2 * threshold - (threshold**2) / 2 - 1
    return {
        "title": "Vacuum Usage - Piecewise PDF" if is_en else "Elektrik Süpürgesi - Parçalı PDF",
        "text": (
            (
                "The duration (X, in hours) a family spends using their vacuum cleaner during weekend chores has the following density function (PDF):\n"
                "f(x)=x for 0<x<1, f(x)=2-x for 1<=x<2, and 0 otherwise.\n"
                f"Calculate the probability that the family uses the vacuum for less than {threshold:.2f} hours: P(X<{threshold:.2f})."
            )
            if is_en
            else (
                "Ev temizliği yapan bir ailenin elektrik süpürgesi kullanım süresi (X saat), aşağıdaki parçalı yoğunluk fonksiyonuna (PDF) sahiptir:\n"
                "0<x<1 için f(x)=x, 1<=x<2 için f(x)=2-x, diğer durumlarda 0.\n"
                f"Ailenin süpürgeyi {threshold:.2f} saatten daha az kullanmış olma olasılığını hesaplayınız: P(X<{threshold:.2f})."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "visual": {
            "kind": "piecewise_pdf_vacuum",
            "threshold": threshold,
        },
    }


def _build_q_circuit_reliability(r: random.Random, is_en: bool, tolerance: float) -> Question:
    p_c1, p_c2, p_c3 = r.choice(
        [
            (0.90, 0.90, 0.80),
            (0.92, 0.88, 0.82),
            (0.94, 0.86, 0.84),
        ]
    )
    p_parallel = 1 - (1 - p_c1) * (1 - p_c2)
    answer = p_parallel * p_c3
    return {
        "title": "Series-Parallel Circuit" if is_en else "Seri-Paralel Devre",
        "text": (
            (
                f"Components are independent. P(C1)={p_c1:.2f}, P(C2)={p_c2:.2f}, P(C3)={p_c3:.2f}.\n"
                "C1 and C2 are parallel, then in series with C3. Find system reliability."
            )
            if is_en
            else (
                f"Bileşenler bağımsızdır. P(C1)={p_c1:.2f}, P(C2)={p_c2:.2f}, P(C3)={p_c3:.2f}.\n"
                "C1 ve C2 paralel, sonra C3 ile seri. Sistemin çalışma olasılığını bulunuz."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "visual": {
            "kind": "circuit_c123",
            "p_c1": p_c1,
            "p_c2": p_c2,
            "p_c3": p_c3,
        },
    }


def _build_q_fabric_expected_value(r: random.Random, is_en: bool, tolerance: float) -> Question:
    fabric_length, x_values, p_values = _fabric_defect_distribution(r)
    answer = sum(x * p for x, p in zip(x_values, p_values))
    return {
        "title": "Fabric Defect Count - Expected Value" if is_en else "Kumaş Kusur Sayısı - Beklenen Değer",
        "text": (
            (
                f"The table below shows the PMF for the number of defects (X) found in a {fabric_length}m roll of synthetic fabric produced in a textile mill.\n"
                "Calculate the expected value E(X) for the defect count based on this given distribution."
            )
            if is_en
            else (
                f"Bir tekstil fabrikasında üretilen {fabric_length} metre uzunluğundaki sentetik kumaş topunda rastlanan defolu iplik veya ilmek gibi kusur sayısı X'in PMF tablosu aşağıda verilmiştir.\n"
                "Bu olasılık dağılımını kullanarak bir top kumaşta çıkması beklenen ortalama kusur sayısını, yani beklenen değeri E(X) hesaplayınız."
            )
        ),
        "answer": answer,
        "tolerance": tolerance,
        "visual": {
            "kind": "pmf_table",
            "x_values": x_values,
            "p_values": p_values,
            "caption": "PMF: Kumaş kusur sayısı" if not is_en else "PMF: Fabric defect count",
        },
    }


def _build_q_discrete_uniform_helpdesk(r: random.Random, is_en: bool, tolerance: float) -> Question:
    room_count, threshold = r.choice(
        [
            (5, 4),
            (6, 5),
            (7, 5),
            (8, 6),
        ]
    )
    x_values = list(range(1, room_count + 1))
    p_values = [1 / room_count for _ in x_values]
    answer = sum(prob for x, prob in zip(x_values, p_values) if x >= threshold)
    return {
        "title": "Help Desk Direction - Discrete Uniform" if is_en else "Danışma Yönlendirmesi - Ayrık Uniform",
        "text": (
            (
                f"A student visiting the university counseling center is randomly directed (with equal probability) to one of the {room_count} available meeting rooms.\n"
                f"If X is the assigned room number, find the probability that the student is directed to room number {threshold} or higher: P(X>={threshold})."
            )
            if is_en
            else (
                f"Üniversitenin öğrenci destek merkezine gelen bir öğrenci, boş olan {room_count} farklı görüşme odasından birine rastgele (eş olasılıkla) yönlendirilmektedir.\n"
                f"X değişkeni yönlendirilen odanın numarasını temsil ettiğine göre, öğrencinin {threshold} veya daha büyük numaralı bir odaya yönlendirilme olasılığını bulunuz: P(X>={threshold})."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "threshold": threshold,
        "visual": {
            "kind": "pmf_bar",
            "x_values": x_values,
            "p_values": p_values,
            "caption": "Dağılım Şekli" if not is_en else "Distribution Shape",
            "show_prob_labels": False,
            "show_y_axis_values": False,
        },
    }


def _build_q_bernoulli_message_reply(r: random.Random, is_en: bool, tolerance: float) -> Question:
    success_prob = r.choice([0.62, 0.68, 0.74, 0.78, 0.82])
    answer = success_prob * (1 - success_prob)
    return {
        "title": "Same-Day Reply - Bernoulli" if is_en else "Aynı Gün Dönüş - Bernoulli",
        "text": (
            (
                f"The probability of a student receiving a same-day reply to an email sent to their academic advisor is {success_prob:.2f}.\n"
                "Let X=1 if the reply arrives on the same day, and X=0 otherwise. Calculate the variance of this indicator variable: Var(X)."
            )
            if is_en
            else (
                f"Bir öğrencinin akademik danışmanına gönderdiği e-postaya aynı gün içinde yanıt alma olasılığı {success_prob:.2f} olarak bilinmektedir.\n"
                "X rassal değişkeni, mesaja aynı gün dönüş gelmesi durumunda 1, gelmemesi durumunda 0 değerini almaktadır. Bu durum için X'in varyansını hesaplayınız: Var(X)."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "visual": {
            "kind": "pmf_bar",
            "x_values": [0, 1],
            "p_values": [1 - success_prob, success_prob],
            "caption": "Dağılım Şekli" if not is_en else "Distribution Shape",
            "show_prob_labels": False,
            "show_y_axis_values": False,
        },
    }


def _build_q_binomial_assignment_uploads(r: random.Random, is_en: bool, tolerance: float) -> Question:
    trial_count, success_prob, target_successes = r.choice(
        [
            (6, 0.65, 4),
            (7, 0.70, 5),
            (8, 0.60, 5),
            (9, 0.75, 7),
        ]
    )
    x_values = list(range(0, trial_count + 1))
    p_values = [
        math.comb(trial_count, x) * (success_prob**x) * ((1 - success_prob) ** (trial_count - x))
        for x in x_values
    ]
    answer = p_values[target_successes]
    return {
        "title": "Assignment Uploads - Binomial" if is_en else "Ödev Yüklemeleri - Binom",
        "text": (
            (
                f"In a group project, {trial_count} students independently manage their work. The probability of each student submitting their assignment before the deadline without delay is {success_prob:.2f}.\n"
                f"If X represents the number of students who submit on time, calculate the probability that exactly {target_successes} students successfully meet the deadline: P(X={target_successes})."
            )
            if is_en
            else (
                f"Proje yönetimi dersindeki bir grupta, {trial_count} öğrencinin son teslim tarihinden önce ödevini sisteme yükleme olasılığı birbirinden bağımsız olarak {success_prob:.2f}'dir.\n"
                f"X değişkeni ödevini zamanında teslim eden öğrenci sayısını gösterdiğine göre, tam olarak {target_successes} öğrencinin ödevini yetiştirme olasılığını hesaplayınız: P(X={target_successes})."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "visual": {
            "kind": "pmf_bar",
            "x_values": x_values,
            "p_values": p_values,
            "highlight_x": target_successes,
            "trial_count": trial_count,
            "success_prob": success_prob,
            "caption": "Dağılım Şekli" if not is_en else "Distribution Shape",
            "show_prob_labels": False,
            "show_y_axis_values": False,
            "show_highlight": False,
        },
    }


def _build_q_multinomial_feedback_mix(r: random.Random, is_en: bool, tolerance: float) -> Question:
    trial_count, probabilities, counts = r.choice(
        [
            (5, (0.50, 0.30, 0.20), (2, 2, 1)),
            (5, (0.40, 0.35, 0.25), (2, 2, 1)),
            (6, (0.50, 0.30, 0.20), (3, 2, 1)),
            (6, (0.45, 0.35, 0.20), (2, 3, 1)),
        ]
    )
    coefficient = math.factorial(trial_count)
    for count in counts:
        coefficient /= math.factorial(count)
    answer = coefficient
    for probability, count in zip(probabilities, counts):
        answer *= probability**count
    labels_tr = ["olumlu", "nötr", "olumsuz"]
    labels_en = ["positive", "neutral", "negative"]
    labels = labels_en if is_en else labels_tr
    return {
        "title": "Daily Feedback Mix - Multinomial" if is_en else "Günlük Geri Bildirim Dağılımı - Multinomial",
        "text": (
            (
                f"A newly launched mobile app receives {trial_count} independent user reviews in a single day. Based on historical data, the probability of a review being positive, neutral, or negative is ({probabilities[0]:.2f}, {probabilities[1]:.2f}, {probabilities[2]:.2f}) respectively.\n"
                ""

                f"Find the probability that the day's feedback precisely consists of {counts[0]} {labels[0]}, {counts[1]} {labels[1]}, and {counts[2]} {labels[2]} reviews."
            )
            if is_en
            else (
                f"Yeni piyasaya sürülen bir mobil uygulama için App Store'a bir gün içinde birbirinden bağımsız {trial_count} kullanıcı değerlendirmesi gelmiştir. Her bir yorumun olumlu, nötr veya olumsuz olma olasılıkları sırasıyla ({probabilities[0]:.2f}, {probabilities[1]:.2f}, {probabilities[2]:.2f}) şeklindedir.\n"
                ""

                f"Gelen bu değerlendirmelerin tam olarak {counts[0]} tanesinin {labels[0]}, {counts[1]} tanesinin {labels[1]} ve {counts[2]} tanesinin {labels[2]} olma olasılığını bulunuz."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "trial_count": trial_count,
        "category_probs": list(probabilities),
        "category_counts": list(counts),
        "category_labels": labels,
    }


def _build_q_hypergeom_cookie_box(r: random.Random, is_en: bool, tolerance: float) -> Question:
    population_size, success_count, draw_count, target_successes = r.choice(
        [
            (10, 3, 4, 1),
            (12, 4, 5, 2),
            (14, 5, 4, 1),
            (15, 5, 6, 2),
        ]
    )
    support_start = max(0, draw_count - (population_size - success_count))
    support_end = min(success_count, draw_count)
    x_values = list(range(support_start, support_end + 1))
    p_values = [
        (math.comb(success_count, x) * math.comb(population_size - success_count, draw_count - x))
        / math.comb(population_size, draw_count)
        for x in x_values
    ]
    answer = p_values[x_values.index(target_successes)]
    return {
        "title": "Cookie Box Choice - Hypergeometric" if is_en else "Kurabiye Kutusu Seçimi - Hipergeometrik",
        "text": (
            (
                f"A bakery displays a fresh batch of {population_size} mixed cookies, of which exactly {success_count} are hazelnut flavor. A customer randomly picks {draw_count} cookies to be boxed for their friends (drawn without replacement).\n"
                f"If X is the number of hazelnut cookies in the box, find the probability that there are exactly {target_successes} hazelnut cookies in the selection: P(X={target_successes})."
            )
            if is_en
            else (
                f"Bir kafenin vitrinindeki taze pişmiş {population_size} adet karışık kurabiyenin içinde tam olarak {success_count} tanesi fındıklıdır. İçeri giren bir müşteri, arkadaşlarına ikram etmek üzere rastgele seçtiği {draw_count} adet kurabiyeyi paketletiyor (geriye koymaksızın çekim).\n"
                f"X paketlenen fındıklı kurabiye sayısı olduğuna göre, pakette tam olarak {target_successes} adet fındıklı kurabiye bulunma olasılığını hesaplayınız: P(X={target_successes})."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "visual": {
            "kind": "pmf_bar",
            "x_values": x_values,
            "p_values": p_values,
            "population_size": population_size,
            "success_count": success_count,
            "draw_count": draw_count,
            "highlight_x": target_successes,
            "caption": "Dağılım Şekli" if not is_en else "Distribution Shape",
            "show_prob_labels": False,
            "show_y_axis_values": False,
            "show_highlight": False,
        },
    }


def _build_q_geometric_online_payment(r: random.Random, is_en: bool, tolerance: float) -> Question:
    success_prob, trial_number = r.choice(
        [
            (0.30, 3),
            (0.35, 4),
            (0.40, 4),
            (0.45, 5),
        ]
    )
    answer = ((1 - success_prob) ** (trial_number - 1)) * success_prob
    return {
        "title": "Online Payment Attempt - Geometric" if is_en else "Online Ödeme Denemesi - Geometrik",
        "text": (
            (
                f"During a busy holiday sale on an e-commerce platform, a credit card payment attempt has a {success_prob:.2f} probability of succeeding on any single try. Each attempt is independent.\n"
                f"Calculate the probability that a user checking out achieves their first successful payment on their exactly {trial_number}th attempt: P(X={trial_number})."
            )
            if is_en
            else (
                f"Popüler bir e-ticaret sitesinde kampanya dönemindeki yoğunluk nedeniyle, kredi kartıyla yapılan bir ödeme işleminin tek bir denemede başarıyla sonuçlanma olasılığı {success_prob:.2f}'dir. Her deneme birbirinden bağımsızdır.\n"
                f"Sepetini onaylamaya çalışan bir kullanıcının, ilk başarılı ödemeyi tam olarak {trial_number}. denemesinde gerçekleştirme olasılığını bulunuz: P(X={trial_number})."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "success_prob": success_prob,
        "trial_number": trial_number,
    }


def _build_q_negative_binomial_call_reach(r: random.Random, is_en: bool, tolerance: float) -> Question:
    success_target, trial_number, success_prob = r.choice(
        [
            (2, 4, 0.55),
            (2, 5, 0.60),
            (3, 5, 0.55),
            (3, 6, 0.60),
        ]
    )
    answer = math.comb(trial_number - 1, success_target - 1) * (success_prob**success_target) * (
        (1 - success_prob) ** (trial_number - success_target)
    )
    return {
        "title": "Reaching a Customer - Negative Binomial" if is_en else "Müşteriye Ulaşma - Negatif Binom",
        "text": (
            (
                f"A customer support agent has a {success_prob:.2f} probability of successfully reaching an active subscriber on any independent outbound call.\n"
                f"Calculate the probability that the agent achieves their {success_target}th successful conversation precisely on the {trial_number}th call attempt of the day: P(X={trial_number})."
            )
            if is_en
            else (
                f"Bir çağrı merkezindeki müşteri temsilcisinin, aradığı abonelere ulaşabilme (telefonun açılması) olasılığı her bağımsız denemede {success_prob:.2f} olarak ölçülmüştür.\n"
                f"Günlük kotasını doldurmaya çalışan temsilcinin, {success_target}. başarılı abone görüşmesini günün tam olarak {trial_number}. aramasında gerçekleştirme olasılığını hesaplayınız: P(X={trial_number})."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "success_target": success_target,
        "trial_number": trial_number,
        "success_prob": success_prob,
    }


def _build_q_poisson_courier_calls(r: random.Random, is_en: bool, tolerance: float) -> Question:
    rate_per_hour, hours, event_count = r.choice(
        [
            (2.0, 0.5, 1),
            (2.4, 0.5, 1),
            (3.0, 1.0, 2),
            (3.6, 0.5, 2),
        ]
    )
    mean = rate_per_hour * hours
    answer = math.exp(-mean) * (mean**event_count) / math.factorial(event_count)
    minutes = int(hours * 60)
    return {
        "title": "Courier Call Count - Poisson" if is_en else "Kurye Araması Sayısı - Poisson",
        "text": (
            (
                f"The reception desk of a busy corporate building receives notification calls from delivery couriers at an average rate of {rate_per_hour:.1f} per hour.\n"
                f"Assuming these incoming calls follow a Poisson model, find the probability that the receptionist will receive exactly {event_count} courier calls during a {minutes}-minute timeframe."
            )
            if is_en
            else (
                f"Kalabalık bir iş merkezinin resepsiyonuna, teslimat için gelen kuryelerden saatte ortalama {rate_per_hour:.1f} kez bildirim telefonu gelmektedir.\n"
                f"Telefon geliş senaryosunun bir Poisson sürecine uyduğu varsayılırsa, banko görevlisi moladayken ({minutes} dakika) tam olarak {event_count} adet kurye telefonu gelme olasılığını hesaplayınız."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "rate_per_hour": rate_per_hour,
        "hours": hours,
        "event_count": event_count,
    }


def _build_q_continuous_uniform_paint_dry(r: random.Random, is_en: bool, tolerance: float) -> Question:
    interval_start, interval_end, lower_bound, upper_bound = r.choice(
        [
            (30, 50, 34, 42),
            (25, 45, 30, 38),
            (20, 40, 26, 33),
            (35, 55, 41, 49),
        ]
    )
    answer = (upper_bound - lower_bound) / (interval_end - interval_start)
    return {
        "title": "Paint Drying Time - Continuous Uniform" if is_en else "Boya Kuruma Süresi - Sürekli Uniform",
        "text": (
            (
                f"In a woodworking workshop, the time it takes for a special varnish applied to wooden furniture to completely dry (X minutes) follows a Continuous Uniform distribution on the interval ({interval_start}, {interval_end}).\n"
                f"Calculate the probability that a freshly varnished coffee table takes between {lower_bound} and {upper_bound} minutes to dry: P({lower_bound}<X<{upper_bound})."
            )
            if is_en
            else (
                f"Bir marangoz atölyesinde ahşap mobilyalar için kullanılan özel bir cilanın tamamen kuruma ve dokunulabilir hale gelme süresi (X dakika), ({interval_start}, {interval_end}) aralığında Sürekli Uniform (Düzgün) dağılıma uymaktadır.\n"
                f"Yeni cilalanmış bir sehpanın kurumasının {lower_bound} dakika ile {upper_bound} dakika arasında sürme olasılığını hesaplayınız: P({lower_bound}<X<{upper_bound})."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "interval_start": interval_start,
        "interval_end": interval_end,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


def _build_q_normal_shaft_acceptance(r: random.Random, is_en: bool, tolerance: float) -> Question:
    mean, std_dev, lower_bound, upper_bound = r.choice(
        [
            (20.00, 0.04, 19.92, 20.06),
            (18.00, 0.05, 17.95, 18.08),
            (25.00, 0.10, 24.90, 25.15),
            (12.00, 0.20, 11.80, 12.10),
        ]
    )
    answer = _normal_cdf(upper_bound, mean, std_dev) - _normal_cdf(lower_bound, mean, std_dev)
    return {
        "title": "Shaft Acceptance - Normal" if is_en else "Şaft Kabul Olasılığı - Normal",
        "text": (
            (
                f"Precision steel shafts produced in a factory have diameters (X mm) that are Normally distributed with a mean of {mean:.2f} mm and a standard deviation of {std_dev:.2f} mm.\n"
                f"Quality control standards require a shaft diameter to be strictly between {lower_bound:.2f} mm and {upper_bound:.2f} mm to fit seamlessly into the engine block. Find the probability that a randomly selected shaft passes the inspection."
            )
            if is_en
            else (
                f"Bir fabrikada üretilen hassas çelik şaftların çapı (X mm), ortalaması {mean:.2f} mm ve standart sapması {std_dev:.2f} mm olan Normal dağılıma uymaktadır.\n"
                f"Kalite kontrol standartlarına göre, bir şaftın motora sorunsuz monte edilebilmesi için çapının {lower_bound:.2f} mm ile {upper_bound:.2f} mm arasında olması şarttır. Rastgele seçilmiş bir şaftın kalite testini geçme (kabul edilme) olasılığını hesaplayınız."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "mean": mean,
        "std_dev": std_dev,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


def _build_q_binomial_normal_approx_quality(r: random.Random, is_en: bool, tolerance: float) -> Question:
    trial_count, defect_prob, threshold_count = r.choice(
        [
            (200, 0.04, 10),
            (150, 0.08, 16),
            (120, 0.40, 55),
            (180, 0.30, 60),
        ]
    )
    mean = trial_count * defect_prob
    std_dev = math.sqrt(trial_count * defect_prob * (1 - defect_prob))
    answer = _standard_normal_cdf((threshold_count + 0.5 - mean) / std_dev)
    return {
        "title": "Quality Line - Normal Approximation to Binomial" if is_en else "Kalite Hattı - Binoma Normal Yaklaşım",
        "text": (
            (
                f"A factory produces {trial_count} smartwatch screens daily. Each screen has an independent {defect_prob:.2f} probability of containing a defective dead pixel.\n"
                f"The quality department tolerates up to {threshold_count} defective units per day. Using the normal approximation with continuity correction, find the probability that {threshold_count} or fewer defective screens are produced in a day."
            )
            if is_en
            else (
                f"Günde {trial_count} adet akıllı saat ekranı üreten bir fabrikada, her bir ürünün minik bir piksel kusuru içerme olasılığı birbirinden bağımsız olarak {defect_prob:.2f}'dir.\n"
                f"Günlük kusurlu üretim sayısı (X), Binom dağılımından gelmektedir. Kalite biriminin günlük toleransı maksimum {threshold_count} hatadır. Sürekli düzeltmeli normal yaklaşımı formülünü kullanarak, fabrikada o gün en fazla {threshold_count} adet kusurlu ekran üretilme olasılığını yaklaşık olarak hesaplayınız: P(X<={threshold_count})."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "trial_count": trial_count,
        "defect_prob": defect_prob,
        "threshold_count": threshold_count,
    }


def _build_q_exponential_server_fault_wait(r: random.Random, is_en: bool, tolerance: float) -> Question:
    rate_per_hour, threshold_hours = r.choice(
        [
            (2.0, 0.75),
            (1.5, 1.0),
            (3.0, 0.50),
            (2.4, 1.25),
        ]
    )
    answer = math.exp(-rate_per_hour * threshold_hours)
    minutes = int(round(threshold_hours * 60))
    return {
        "title": "First Fault Record Wait - Exponential" if is_en else "İlk Arıza Kaydı Bekleme - Üssel",
        "text": (
            (
                f"The cloud server center of a tech firm receives system warning alerts at an average rate of {rate_per_hour:.1f} per hour.\n"
                f"The waiting time (X hours) until the very first warning alert arrives is modeled by an Exponential distribution. After restarting the servers, calculate the probability that no warnings occur in the first {minutes} minutes, which means the waiting time exceeds {threshold_hours:.2f} hours: P(X>{threshold_hours:.2f})."
            )
            if is_en
            else (
                f"Büyük bir teknoloji firmasının bulut sunucu merkezine, saatte ortalama {rate_per_hour:.1f} adet sistemsel uyarı (arıza/kesinti logu) düşmektedir.\n"
                f"Sistemi yeni baştan başlatan bir yöneticinin, ilk uyarı kaydını alana kadar geçen bekleme süresi (X saat) Üstel dağılım ile modellenir. Servisler açıldıktan sonraki ilk {minutes} dakika içinde sunuculardan hiçbir arıza uyarısı gelmeme, yani bekleme süresinin {threshold_hours:.2f} saatten uzun sürme olasılığını hesaplayınız: P(X>{threshold_hours:.2f})."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "rate_per_hour": rate_per_hour,
        "threshold_hours": threshold_hours,
    }


def _build_q_gamma_second_call_wait(r: random.Random, is_en: bool, tolerance: float) -> Question:
    shape, rate_per_minute, threshold_minutes = r.choice(
        [
            (2, 4.0, 0.25),   # lambda*t=1.0, k=2 -> survival ~0.736 (15 s)
            (2, 3.0, 0.50),   # lambda*t=1.5, k=2 -> survival ~0.558 (30 s)
            (2, 6.0, 1 / 3),  # lambda*t=2.0, k=2 -> survival ~0.406 (20 s)
            (2, 2.0, 1.50),   # lambda*t=3.0, k=2 -> survival ~0.199 (90 s)
            (3, 2.0, 0.75),   # lambda*t=1.5, k=3 -> survival ~0.809 (45 s)
            (3, 3.0, 1.00),   # lambda*t=3.0, k=3 -> survival ~0.423 (60 s)
        ]
    )
    answer = _erlang_survival(shape, rate_per_minute, threshold_minutes)
    threshold_seconds = int(round(threshold_minutes * 60))
    return {
        "title": "Waiting for the kth Call - Gamma" if is_en else "k. Çağrıya Kadar Bekleme - Gamma",
        "text": (
            (
                f"During a busy lunch hour, an emergency dispatch center handles incoming calls at an average rate of {rate_per_minute:.1f} per minute.\n"
                f"Assuming arrivals follow a Poisson process, the waiting time (X minutes) to receive the {shape}th emergency case follows a Gamma (Erlang) distribution. Calculate the probability that the dispatcher waits more than {threshold_seconds} seconds to be assigned their {shape}th case: P(X>{threshold_minutes:.2f})."
            )
            if is_en
            else (
                f"Bir acil çağrı merkezine yoğunluğun çok olduğu saatlerde dakikada ortalama {rate_per_minute:.1f} adet çağrı düşmektedir.\n"
                f"Vardiyaya yeni başlayan bir operatörün sisteme düşen {shape}. vakayı bekleme süresinin (X dakika) Gamma dağılımına uyduğu bilinmektedir. Bu operatörün {shape}. vakasının, sisteme giriş yaptığı andan itibaren tam {threshold_seconds} saniyeden daha uzun bir sürenin ardından gelme olasılığını bulunuz: P(X>{threshold_minutes:.2f})."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "shape": shape,
        "rate_per_minute": rate_per_minute,
        "threshold_minutes": threshold_minutes,
    }


def _build_q_weibull_pump_lifetime(r: random.Random, is_en: bool, tolerance: float) -> Question:
    alpha_scale, beta_shape, threshold_hours = r.choice(
        [
            (0.002, 2.0, 15),
            (0.005, 2.0, 10),
            (0.001, 3.0, 8),
            (0.004, 1.5, 12),
        ]
    )
    answer = 1 - math.exp(-alpha_scale * (threshold_hours**beta_shape))
    return {
        "title": "Pump Seal Lifetime - Weibull" if is_en else "Pompa Keçesi Ömrü - Weibull",
        "text": (
            (
                f"The wear and failure lifetime X (in hours) of high-pressure water pump seals used in a hydroelectric dam is modeled by a Weibull distribution. Its Cumulative Distribution Function for failure is F(x)=1-exp(-{alpha_scale:.3f} x^{beta_shape:.1f}) for x>0.\n"
                f"Find the probability that a newly installed pump seal degrades and fails within the first {threshold_hours} hours of operation."
            )
            if is_en
            else (
                f"Bir hidroelektrik santralinde kullanılan yüksek basınçlı su pompası sızdırmazlık contalarının arızalanma ömrü X (saat), Weibull dağılımı ile modellenmektedir. Contaların bozulma birikimli dağılım fonksiyonu (CDF), x>0 için F(x)=1-exp(-{alpha_scale:.3f} x^{beta_shape:.1f}) formülüyle ifade edilir.\n"
                f"Yeni takılan bir sızdırmazlık contasının santral çalışmaya başladıktan sonraki ilk {threshold_hours} saat içinde dayanamayıp arıza verme olasılığını hesaplayınız."
            )
        ),
        "answer": answer,
        "answer_min": 0.0,
        "answer_max": 1.0,
        "tolerance": tolerance,
        "alpha_scale": alpha_scale,
        "beta_shape": beta_shape,
        "threshold_hours": threshold_hours,
    }


def _validate_q_backup_power_union(question: Question) -> None:
    question_id = "backup_power_union"
    answer = _validate_question_shell(question_id, question)
    visual = _require_visual_kind(question_id, question, "venn_prob")
    left = _ensure_probability(visual.get("left"), question_id, "visual.left")
    right = _ensure_probability(visual.get("right"), question_id, "visual.right")
    both = _ensure_probability(visual.get("both"), question_id, "visual.both")
    _ensure(_approx_equal(both, left * right), question_id, "visual.both must equal P(A)P(B) under independence")
    expected_answer = left + right - both
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal P(A union B)")


def _validate_q_hotel_chain_total_probability(question: Question) -> None:
    question_id = "hotel_chain_total_probability"
    answer, _, priors, cond_event, _ = _validate_probability_tree_visual(question_id, question)
    expected_answer = sum(prior * cond for prior, cond in zip(priors, cond_event))
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal total probability")


def _validate_q_medical_test_bayes(question: Question) -> None:
    question_id = "medical_test_bayes"
    answer, sources, priors, cond_event, _ = _validate_probability_tree_visual(question_id, question)
    _ensure(sources == ["H", "H'"], question_id, "sources must be ['H', \"H'\"]")
    _ensure(len(priors) == 2 and len(cond_event) == 2, question_id, "Bayes test must have two branches")
    _ensure(_approx_equal(priors[0] + priors[1], 1.0), question_id, "health priors must sum to 1")
    p_pos = priors[0] * cond_event[0] + priors[1] * cond_event[1]
    _ensure(p_pos > 0.0, question_id, "positive-test evidence must have positive probability")
    expected_answer = (priors[0] * cond_event[0]) / p_pos
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal Bayes posterior P(H|+)")


def _validate_q_tv_sets_p_ge_1(question: Question) -> None:
    question_id = "tv_sets_p_ge_1"
    answer, x_values, p_values, visual = _validate_pmf_visual(question_id, question)
    support = _validate_integer_support(question_id, x_values)
    _ensure(support[0] == 0, question_id, "support must start at 0 defective TVs")
    _ensure(support == list(range(support[0], support[0] + len(support))), question_id, "support must be consecutive integers")
    total = int(_as_float(visual.get("population_size"), question_id, "visual.population_size"))
    defective = int(_as_float(visual.get("defective_count"), question_id, "visual.defective_count"))
    draw = int(_as_float(visual.get("draw_count"), question_id, "visual.draw_count"))
    _ensure(0 <= defective <= total, question_id, "defective_count must stay within population_size")
    _ensure(0 < draw <= total, question_id, "draw_count must stay within population_size")
    expected_probs = [
        (math.comb(defective, x) * math.comb(total - defective, draw - x)) / math.comb(total, draw)
        for x in support
    ]
    for idx, value in enumerate(p_values):
        _ensure(_approx_equal(value, expected_probs[idx]), question_id, f"p_values[{idx}] must match hypergeometric PMF")
    _ensure(_approx_equal(answer, 1.0 - p_values[0]), question_id, "answer must equal 1 - P(X=0)")


def _validate_q_software_bug_variance(question: Question) -> None:
    question_id = "software_bug_variance"
    answer, x_values, p_values, _ = _validate_pmf_visual(question_id, question)
    support = _validate_integer_support(question_id, x_values)
    _ensure(support == list(range(support[0], support[0] + len(support))), question_id, "support must be consecutive integers")
    expected_value = sum(x * p for x, p in zip(x_values, p_values))
    expected_square = sum((x**2) * p for x, p in zip(x_values, p_values))
    expected_answer = expected_square - expected_value * expected_value
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal Var(X)")


def _validate_q_factory_total_probability(question: Question) -> None:
    question_id = "factory_total_probability"
    answer, sources, priors, cond_event, _ = _validate_probability_tree_visual(question_id, question)
    _ensure(sources == ["B1", "B2", "B3"], question_id, "sources must be ['B1', 'B2', 'B3']")
    expected_answer = sum(prior * cond for prior, cond in zip(priors, cond_event))
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal total probability P(A)")


def _validate_q_factory_bayes_machine3(question: Question) -> None:
    question_id = "factory_bayes_machine3"
    answer, sources, priors, cond_event, _ = _validate_probability_tree_visual(question_id, question)
    _ensure(sources == ["B1", "B2", "B3"], question_id, "sources must be ['B1', 'B2', 'B3']")
    evidence = sum(prior * cond for prior, cond in zip(priors, cond_event))
    _ensure(evidence > 0.0, question_id, "defect evidence must have positive probability")
    expected_answer = (priors[2] * cond_event[2]) / evidence
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal Bayes posterior P(B3|A)")


def _validate_q_airbag_pmf_exact_two(question: Question) -> None:
    question_id = "airbag_pmf_exact_two"
    answer, x_values, p_values, visual = _validate_pmf_visual(question_id, question)
    support = _validate_integer_support(question_id, x_values)
    _ensure(support == list(range(0, len(support))), question_id, "support must be 0..n")
    trial_count = int(_as_float(visual.get("trial_count"), question_id, "visual.trial_count"))
    success_prob = _ensure_probability(visual.get("success_prob"), question_id, "visual.success_prob")
    highlight_x = int(_as_float(visual.get("highlight_x"), question_id, "visual.highlight_x"))
    _ensure(trial_count == support[-1], question_id, "trial_count must match PMF support")
    _ensure(highlight_x in support, question_id, "highlight_x must belong to PMF support")
    expected_probs = [
        math.comb(trial_count, x) * (success_prob**x) * ((1 - success_prob) ** (trial_count - x))
        for x in support
    ]
    for idx, value in enumerate(p_values):
        _ensure(_approx_equal(value, expected_probs[idx]), question_id, f"p_values[{idx}] must match binomial PMF")
    _ensure(_approx_equal(answer, p_values[support.index(highlight_x)]), question_id, "answer must equal highlighted PMF value")


def _validate_q_vacuum_pdf_prob(question: Question) -> None:
    question_id = "vacuum_pdf_prob"
    answer = _validate_question_shell(question_id, question)
    visual = _require_visual_kind(question_id, question, "piecewise_pdf_vacuum")
    threshold = _as_float(visual.get("threshold"), question_id, "visual.threshold")
    _ensure(1.0 < threshold < 2.0, question_id, "threshold must stay in the 1<=x<2 branch")
    area = 0.5 + ((2 * 2) - (2**2) / 2) - ((2 * 1) - (1**2) / 2)
    _ensure(_approx_equal(area, 1.0), question_id, "piecewise PDF must integrate to 1")
    expected_answer = 2 * threshold - (threshold**2) / 2 - 1
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal integral P(X<threshold)")
    _ensure(0.0 <= answer <= 1.0, question_id, "probability answer must stay in [0, 1]")


def _validate_q_circuit_reliability(question: Question) -> None:
    question_id = "circuit_reliability"
    answer = _validate_question_shell(question_id, question)
    visual = _require_visual_kind(question_id, question, "circuit_c123")
    p_c1 = _ensure_probability(visual.get("p_c1"), question_id, "visual.p_c1")
    p_c2 = _ensure_probability(visual.get("p_c2"), question_id, "visual.p_c2")
    p_c3 = _ensure_probability(visual.get("p_c3"), question_id, "visual.p_c3")
    expected_answer = (1 - (1 - p_c1) * (1 - p_c2)) * p_c3
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal series-parallel reliability")


def _validate_q_fabric_expected_value(question: Question) -> None:
    question_id = "fabric_expected_value"
    answer, x_values, p_values, _ = _validate_pmf_visual(question_id, question)
    support = _validate_integer_support(question_id, x_values)
    _ensure(support == list(range(0, len(support))), question_id, "support must start at 0 and be consecutive")
    expected_answer = sum(x * p for x, p in zip(x_values, p_values))
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal expected value E(X)")


def _validate_q_discrete_uniform_helpdesk(question: Question) -> None:
    question_id = "discrete_uniform_helpdesk"
    answer, x_values, p_values, _ = _validate_pmf_visual(question_id, question)
    support = _validate_integer_support(question_id, x_values)
    _ensure(support == list(range(1, len(support) + 1)), question_id, "support must be 1..m")
    _ensure(all(_approx_equal(value, p_values[0]) for value in p_values), question_id, "all probabilities must be equal")
    threshold_raw = _as_float(question.get("threshold"), question_id, "threshold")
    threshold = int(round(threshold_raw))
    _ensure(_approx_equal(threshold_raw, threshold), question_id, "threshold must be an integer")
    expected_answer = sum(prob for x, prob in zip(support, p_values) if x >= threshold)
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal upper-tail discrete uniform probability")


def _validate_q_bernoulli_message_reply(question: Question) -> None:
    question_id = "bernoulli_message_reply"
    answer, x_values, p_values, _ = _validate_pmf_visual(question_id, question)
    support = _validate_integer_support(question_id, x_values)
    _ensure(support == [0, 1], question_id, "support must be [0, 1]")
    success_prob = p_values[1]
    expected_answer = success_prob * (1 - success_prob)
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal Bernoulli variance p(1-p)")


def _validate_q_binomial_assignment_uploads(question: Question) -> None:
    question_id = "binomial_assignment_uploads"
    answer, x_values, p_values, visual = _validate_pmf_visual(question_id, question)
    support = _validate_integer_support(question_id, x_values)
    trial_count = int(_as_float(visual.get("trial_count"), question_id, "visual.trial_count"))
    success_prob = _ensure_probability(visual.get("success_prob"), question_id, "visual.success_prob")
    highlight_x = int(_as_float(visual.get("highlight_x"), question_id, "visual.highlight_x"))
    _ensure(support == list(range(0, trial_count + 1)), question_id, "support must be 0..n")
    expected_probs = [
        math.comb(trial_count, x) * (success_prob**x) * ((1 - success_prob) ** (trial_count - x))
        for x in support
    ]
    for idx, value in enumerate(p_values):
        _ensure(_approx_equal(value, expected_probs[idx]), question_id, f"p_values[{idx}] must match binomial PMF")
    _ensure(highlight_x in support, question_id, "highlight_x must belong to support")
    _ensure(_approx_equal(answer, p_values[support.index(highlight_x)]), question_id, "answer must equal highlighted PMF value")


def _validate_q_multinomial_feedback_mix(question: Question) -> None:
    question_id = "multinomial_feedback_mix"
    answer = _validate_question_shell(question_id, question)
    trial_count_raw = _as_float(question.get("trial_count"), question_id, "trial_count")
    trial_count = int(round(trial_count_raw))
    _ensure(_approx_equal(trial_count_raw, trial_count), question_id, "trial_count must be an integer")
    category_probs = _require_numeric_sequence(question_id, question, "category_probs")
    category_counts_raw = _require_numeric_sequence(question_id, question, "category_counts")
    category_counts = [int(round(value)) for value in category_counts_raw]
    _ensure(len(category_probs) == len(category_counts), question_id, "category arrays must have same length")
    _ensure(len(category_probs) >= 2, question_id, "multinomial must have at least two categories")
    for idx, value in enumerate(category_probs):
        _ensure_probability(value, question_id, f"category_probs[{idx}]")
    for idx, value in enumerate(category_counts_raw):
        _ensure(_approx_equal(value, category_counts[idx]), question_id, f"category_counts[{idx}] must be integer")
        _ensure(category_counts[idx] >= 0, question_id, f"category_counts[{idx}] must be non-negative")
    _ensure(_approx_equal(sum(category_probs), 1.0), question_id, "category probabilities must sum to 1")
    _ensure(sum(category_counts) == trial_count, question_id, "category counts must sum to trial_count")
    expected_answer = math.factorial(trial_count)
    for count in category_counts:
        expected_answer /= math.factorial(count)
    for probability, count in zip(category_probs, category_counts):
        expected_answer *= probability**count
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal multinomial probability")


def _validate_q_hypergeom_cookie_box(question: Question) -> None:
    question_id = "hypergeom_cookie_box"
    answer, x_values, p_values, visual = _validate_pmf_visual(question_id, question)
    support = _validate_integer_support(question_id, x_values)
    population_size = int(_as_float(visual.get("population_size"), question_id, "visual.population_size"))
    success_count = int(_as_float(visual.get("success_count"), question_id, "visual.success_count"))
    draw_count = int(_as_float(visual.get("draw_count"), question_id, "visual.draw_count"))
    highlight_x = int(_as_float(visual.get("highlight_x"), question_id, "visual.highlight_x"))
    expected_support = list(range(max(0, draw_count - (population_size - success_count)), min(success_count, draw_count) + 1))
    _ensure(support == expected_support, question_id, "support must match valid hypergeometric outcomes")
    expected_probs = [
        (math.comb(success_count, x) * math.comb(population_size - success_count, draw_count - x))
        / math.comb(population_size, draw_count)
        for x in support
    ]
    for idx, value in enumerate(p_values):
        _ensure(_approx_equal(value, expected_probs[idx]), question_id, f"p_values[{idx}] must match hypergeometric PMF")
    _ensure(highlight_x in support, question_id, "highlight_x must belong to support")
    _ensure(_approx_equal(answer, p_values[support.index(highlight_x)]), question_id, "answer must equal highlighted PMF value")


def _validate_q_geometric_online_payment(question: Question) -> None:
    question_id = "geometric_online_payment"
    answer = _validate_question_shell(question_id, question)
    success_prob = _ensure_probability(question.get("success_prob"), question_id, "success_prob")
    trial_number_raw = _as_float(question.get("trial_number"), question_id, "trial_number")
    trial_number = int(round(trial_number_raw))
    _ensure(_approx_equal(trial_number_raw, trial_number), question_id, "trial_number must be an integer")
    _ensure(trial_number >= 1, question_id, "trial_number must be at least 1")
    expected_answer = ((1 - success_prob) ** (trial_number - 1)) * success_prob
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal geometric PMF")


def _validate_q_negative_binomial_call_reach(question: Question) -> None:
    question_id = "negative_binomial_call_reach"
    answer = _validate_question_shell(question_id, question)
    success_prob = _ensure_probability(question.get("success_prob"), question_id, "success_prob")
    success_target_raw = _as_float(question.get("success_target"), question_id, "success_target")
    trial_number_raw = _as_float(question.get("trial_number"), question_id, "trial_number")
    success_target = int(round(success_target_raw))
    trial_number = int(round(trial_number_raw))
    _ensure(_approx_equal(success_target_raw, success_target), question_id, "success_target must be an integer")
    _ensure(_approx_equal(trial_number_raw, trial_number), question_id, "trial_number must be an integer")
    _ensure(success_target >= 1, question_id, "success_target must be at least 1")
    _ensure(trial_number >= success_target, question_id, "trial_number must be at least success_target")
    expected_answer = math.comb(trial_number - 1, success_target - 1) * (success_prob**success_target) * (
        (1 - success_prob) ** (trial_number - success_target)
    )
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal negative binomial PMF")


def _validate_q_poisson_courier_calls(question: Question) -> None:
    question_id = "poisson_courier_calls"
    answer = _validate_question_shell(question_id, question)
    rate_per_hour = _as_float(question.get("rate_per_hour"), question_id, "rate_per_hour")
    hours = _as_float(question.get("hours"), question_id, "hours")
    event_count_raw = _as_float(question.get("event_count"), question_id, "event_count")
    event_count = int(round(event_count_raw))
    _ensure(rate_per_hour > 0.0, question_id, "rate_per_hour must be positive")
    _ensure(hours > 0.0, question_id, "hours must be positive")
    _ensure(_approx_equal(event_count_raw, event_count), question_id, "event_count must be an integer")
    _ensure(event_count >= 0, question_id, "event_count must be non-negative")
    mean = rate_per_hour * hours
    expected_answer = math.exp(-mean) * (mean**event_count) / math.factorial(event_count)
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal Poisson PMF")


def _validate_q_continuous_uniform_paint_dry(question: Question) -> None:
    question_id = "continuous_uniform_paint_dry"
    answer = _validate_question_shell(question_id, question)
    interval_start = _as_float(question.get("interval_start"), question_id, "interval_start")
    interval_end = _as_float(question.get("interval_end"), question_id, "interval_end")
    lower_bound = _as_float(question.get("lower_bound"), question_id, "lower_bound")
    upper_bound = _as_float(question.get("upper_bound"), question_id, "upper_bound")
    _ensure(interval_start < interval_end, question_id, "interval_start must be smaller than interval_end")
    _ensure(interval_start <= lower_bound < upper_bound <= interval_end, question_id, "bounds must stay within support")
    expected_answer = (upper_bound - lower_bound) / (interval_end - interval_start)
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal continuous uniform interval probability")


def _validate_q_normal_shaft_acceptance(question: Question) -> None:
    question_id = "normal_shaft_acceptance"
    answer = _validate_question_shell(question_id, question)
    mean = _as_float(question.get("mean"), question_id, "mean")
    std_dev = _as_float(question.get("std_dev"), question_id, "std_dev")
    lower_bound = _as_float(question.get("lower_bound"), question_id, "lower_bound")
    upper_bound = _as_float(question.get("upper_bound"), question_id, "upper_bound")
    _ensure(std_dev > 0.0, question_id, "std_dev must be positive")
    _ensure(lower_bound < upper_bound, question_id, "lower_bound must be smaller than upper_bound")
    expected_answer = _normal_cdf(upper_bound, mean, std_dev) - _normal_cdf(lower_bound, mean, std_dev)
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal normal acceptance probability")


def _validate_q_binomial_normal_approx_quality(question: Question) -> None:
    question_id = "binomial_normal_approx_quality"
    answer = _validate_question_shell(question_id, question)
    trial_count_raw = _as_float(question.get("trial_count"), question_id, "trial_count")
    trial_count = int(round(trial_count_raw))
    defect_prob = _ensure_probability(question.get("defect_prob"), question_id, "defect_prob")
    threshold_raw = _as_float(question.get("threshold_count"), question_id, "threshold_count")
    threshold_count = int(round(threshold_raw))
    _ensure(_approx_equal(trial_count_raw, trial_count), question_id, "trial_count must be an integer")
    _ensure(_approx_equal(threshold_raw, threshold_count), question_id, "threshold_count must be an integer")
    _ensure(trial_count > 0, question_id, "trial_count must be positive")
    _ensure(0 <= threshold_count <= trial_count, question_id, "threshold_count must stay within [0, n]")
    variance = trial_count * defect_prob * (1 - defect_prob)
    _ensure(variance > 0.0, question_id, "binomial variance must be positive")
    expected_answer = _standard_normal_cdf((threshold_count + 0.5 - (trial_count * defect_prob)) / math.sqrt(variance))
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal continuity-corrected normal approximation")


def _validate_q_exponential_server_fault_wait(question: Question) -> None:
    question_id = "exponential_server_fault_wait"
    answer = _validate_question_shell(question_id, question)
    rate_per_hour = _as_float(question.get("rate_per_hour"), question_id, "rate_per_hour")
    threshold_hours = _as_float(question.get("threshold_hours"), question_id, "threshold_hours")
    _ensure(rate_per_hour > 0.0, question_id, "rate_per_hour must be positive")
    _ensure(threshold_hours > 0.0, question_id, "threshold_hours must be positive")
    expected_answer = math.exp(-rate_per_hour * threshold_hours)
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal exponential survival probability")


def _validate_q_gamma_second_call_wait(question: Question) -> None:
    question_id = "gamma_second_call_wait"
    answer = _validate_question_shell(question_id, question)
    shape_raw = _as_float(question.get("shape"), question_id, "shape")
    shape = int(round(shape_raw))
    rate_per_minute = _as_float(question.get("rate_per_minute"), question_id, "rate_per_minute")
    threshold_minutes = _as_float(question.get("threshold_minutes"), question_id, "threshold_minutes")
    _ensure(_approx_equal(shape_raw, shape), question_id, "shape must be an integer")
    _ensure(shape >= 1, question_id, "shape must be at least 1")
    _ensure(rate_per_minute > 0.0, question_id, "rate_per_minute must be positive")
    _ensure(threshold_minutes > 0.0, question_id, "threshold_minutes must be positive")
    expected_answer = _erlang_survival(shape, rate_per_minute, threshold_minutes)
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal gamma survival probability")


def _validate_q_weibull_pump_lifetime(question: Question) -> None:
    question_id = "weibull_pump_lifetime"
    answer = _validate_question_shell(question_id, question)
    alpha_scale = _as_float(question.get("alpha_scale"), question_id, "alpha_scale")
    beta_shape = _as_float(question.get("beta_shape"), question_id, "beta_shape")
    threshold_hours = _as_float(question.get("threshold_hours"), question_id, "threshold_hours")
    _ensure(alpha_scale > 0.0, question_id, "alpha_scale must be positive")
    _ensure(beta_shape > 0.0, question_id, "beta_shape must be positive")
    _ensure(threshold_hours > 0.0, question_id, "threshold_hours must be positive")
    expected_answer = 1 - math.exp(-alpha_scale * (threshold_hours**beta_shape))
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal Weibull CDF probability")


QUESTION_DEFINITIONS: tuple[QuestionDefinition, ...] = (
    _question_definition(
        "backup_power_union",
        "Yedek Güç Ünitesi (Bağımsızlık)",
        "Backup Power (Independence)",
        _build_q_backup_power_union,
        _validate_q_backup_power_union,
    ),
    _question_definition(
        "hotel_chain_total_probability",
        "Otel Zinciri (Toplam Olasılık)",
        "Hotel Chain (Total Probability)",
        _build_q_hotel_chain_total_probability,
        _validate_q_hotel_chain_total_probability,
    ),
    _question_definition(
        "medical_test_bayes",
        "Tıbbi Test (Bayes)",
        "Medical Test (Bayes)",
        _build_q_medical_test_bayes,
        _validate_q_medical_test_bayes,
    ),
    _question_definition(
        "tv_sets_p_ge_1",
        "Televizyon Setleri (PMF/CDF)",
        "TV Sets (PMF/CDF)",
        _build_q_tv_sets_p_ge_1,
        _validate_q_tv_sets_p_ge_1,
    ),
    _question_definition(
        "software_bug_variance",
        "Yazılım Hata Analizi (Varyans)",
        "Software Error Analysis (Variance)",
        _build_q_software_bug_variance,
        _validate_q_software_bug_variance,
    ),
    _question_definition(
        "factory_total_probability",
        "Montaj Fabrikası (Toplam Olasılık)",
        "Assembly Factory (Total Probability)",
        _build_q_factory_total_probability,
        _validate_q_factory_total_probability,
    ),
    _question_definition(
        "factory_bayes_machine3",
        "Hata Kaynağı (Bayes)",
        "Defect Source (Bayes)",
        _build_q_factory_bayes_machine3,
        _validate_q_factory_bayes_machine3,
    ),
    _question_definition(
        "airbag_pmf_exact_two",
        "Hava Yastığı Satışı (PMF)",
        "Airbag Sales (PMF)",
        _build_q_airbag_pmf_exact_two,
        _validate_q_airbag_pmf_exact_two,
    ),
    _question_definition(
        "vacuum_pdf_prob",
        "Elektrik Süpürgesi (Parçalı PDF)",
        "Vacuum Usage (Piecewise PDF)",
        _build_q_vacuum_pdf_prob,
        _validate_q_vacuum_pdf_prob,
    ),
    _question_definition(
        "circuit_reliability",
        "Seri-Paralel Devre",
        "Series-Parallel Circuit",
        _build_q_circuit_reliability,
        _validate_q_circuit_reliability,
    ),
    _question_definition(
        "fabric_expected_value",
        "Kumaş Kusur Sayısı (Beklenen Değer)",
        "Fabric Defect Count (Expected Value)",
        _build_q_fabric_expected_value,
        _validate_q_fabric_expected_value,
    ),
    _question_definition(
        "discrete_uniform_helpdesk",
        "Danışma Yönlendirmesi (Ayrık Uniform)",
        "Help Desk Direction (Discrete Uniform)",
        _build_q_discrete_uniform_helpdesk,
        _validate_q_discrete_uniform_helpdesk,
    ),
    _question_definition(
        "bernoulli_message_reply",
        "Aynı Gün Dönüş (Bernoulli)",
        "Same-Day Reply (Bernoulli)",
        _build_q_bernoulli_message_reply,
        _validate_q_bernoulli_message_reply,
    ),
    _question_definition(
        "binomial_assignment_uploads",
        "Ödev Yüklemeleri (Binom)",
        "Assignment Uploads (Binomial)",
        _build_q_binomial_assignment_uploads,
        _validate_q_binomial_assignment_uploads,
    ),
    _question_definition(
        "multinomial_feedback_mix",
        "Günlük Geri Bildirim Dağılımı (Multinomial)",
        "Daily Feedback Mix (Multinomial)",
        _build_q_multinomial_feedback_mix,
        _validate_q_multinomial_feedback_mix,
    ),
    _question_definition(
        "hypergeom_cookie_box",
        "Kurabiye Kutusu Seçimi (Hipergeometrik)",
        "Cookie Box Choice (Hypergeometric)",
        _build_q_hypergeom_cookie_box,
        _validate_q_hypergeom_cookie_box,
    ),
    _question_definition(
        "geometric_online_payment",
        "Online Ödeme Denemesi (Geometrik)",
        "Online Payment Attempt (Geometric)",
        _build_q_geometric_online_payment,
        _validate_q_geometric_online_payment,
    ),
    _question_definition(
        "negative_binomial_call_reach",
        "Müşteriye Ulaşma (Negatif Binom)",
        "Reaching a Customer (Negative Binomial)",
        _build_q_negative_binomial_call_reach,
        _validate_q_negative_binomial_call_reach,
    ),
    _question_definition(
        "poisson_courier_calls",
        "Kurye Araması Sayısı (Poisson)",
        "Courier Call Count (Poisson)",
        _build_q_poisson_courier_calls,
        _validate_q_poisson_courier_calls,
    ),
    _question_definition(
        "continuous_uniform_paint_dry",
        "Boya Kuruma Süresi (Sürekli Uniform)",
        "Paint Drying Time (Continuous Uniform)",
        _build_q_continuous_uniform_paint_dry,
        _validate_q_continuous_uniform_paint_dry,
    ),
    _question_definition(
        "normal_shaft_acceptance",
        "Şaft Kabul Olasılığı (Normal)",
        "Shaft Acceptance (Normal)",
        _build_q_normal_shaft_acceptance,
        _validate_q_normal_shaft_acceptance,
    ),
    _question_definition(
        "binomial_normal_approx_quality",
        "Kalite Hattı (Binoma Normal Yaklaşım)",
        "Quality Line (Normal Approximation to Binomial)",
        _build_q_binomial_normal_approx_quality,
        _validate_q_binomial_normal_approx_quality,
    ),
    _question_definition(
        "exponential_server_fault_wait",
        "İlk Arıza Kaydı Bekleme (Üssel)",
        "First Fault Record Wait (Exponential)",
        _build_q_exponential_server_fault_wait,
        _validate_q_exponential_server_fault_wait,
    ),
    _question_definition(
        "gamma_second_call_wait",
        "k. Çağrıya Kadar Bekleme (Gamma)",
        "Waiting for the kth Call (Gamma)",
        _build_q_gamma_second_call_wait,
        _validate_q_gamma_second_call_wait,
    ),
    _question_definition(
        "weibull_pump_lifetime",
        "Pompa Keçesi Ömrü (Weibull)",
        "Pump Seal Lifetime (Weibull)",
        _build_q_weibull_pump_lifetime,
        _validate_q_weibull_pump_lifetime,
    ),
)


QUESTION_DEFINITION_BY_ID: dict[str, QuestionDefinition] = {item["id"]: item for item in QUESTION_DEFINITIONS}


def validate_question_definitions() -> None:
    """Fail fast if a question definition is incomplete or duplicated."""
    _ensure(len(QUESTION_DEFINITIONS) > 0, "question_registry", "QUESTION_DEFINITIONS must not be empty")
    _ensure(
        len(QUESTION_DEFINITION_BY_ID) == len(QUESTION_DEFINITIONS),
        "question_registry",
        "question ids must be unique",
    )
    for item in QUESTION_DEFINITIONS:
        question_id = str(item.get("id") or "").strip()
        _ensure(question_id != "", "question_registry", "each question must have a non-empty id")
        _ensure(callable(item.get("builder")), question_id, "each question definition must declare a callable builder")
        _ensure(callable(item.get("validator")), question_id, "each question definition must declare a callable validator")
        _ensure(str(item.get("name_tr") or "").strip() != "", question_id, "name_tr must be non-empty")
        _ensure(str(item.get("name_en") or "").strip() != "", question_id, "name_en must be non-empty")


def validate_question(question_id: str, question: Question) -> None:
    """Validate a generated question against its own mathematical invariants."""
    target = str(question_id or "").strip()
    definition = QUESTION_DEFINITION_BY_ID.get(target)
    _ensure(definition is not None, "question_registry", f"unknown question id {target!r}")
    validator = definition["validator"]
    validator(question)


validate_question_definitions()


def get_question_catalog(language: str = DEFAULT_UI_LANGUAGE) -> list[dict[str, str]]:
    """Question bank list used in teacher panel."""
    is_en = _normalize_language(language) == "en"
    return [
        {"id": item["id"], "name": item["name_en"] if is_en else item["name_tr"]}
        for item in QUESTION_DEFINITIONS
    ]


def default_question_slot_ids(slot_count: int = QUIZ_SLOT_COUNT) -> list[str]:
    """Build default slot sequence."""
    valid_ids = [item["id"] for item in QUESTION_DEFINITIONS]
    if not valid_ids:
        return []
    count = max(1, int(slot_count))
    return [valid_ids[idx % len(valid_ids)] for idx in range(count)]


def sanitize_question_slot_ids(slot_ids: Any, slot_count: int = QUIZ_SLOT_COUNT) -> list[str]:
    """Normalize incoming slot ids against valid question ids."""
    defaults = default_question_slot_ids(slot_count=slot_count)
    valid_ids = {item["id"] for item in QUESTION_DEFINITIONS}

    raw_values: list[str] = []
    if isinstance(slot_ids, (list, tuple)):
        raw_values = [str(value).strip() for value in slot_ids]

    normalized: list[str] = []
    for idx in range(len(defaults)):
        candidate = raw_values[idx] if idx < len(raw_values) else ""
        normalized.append(candidate if candidate in valid_ids else defaults[idx])
    return normalized


def question_name_by_id(question_id: str, language: str = DEFAULT_UI_LANGUAGE) -> str:
    """Resolve user-facing name from question id."""
    is_en = _normalize_language(language) == "en"
    target = str(question_id or "").strip()
    for item in QUESTION_DEFINITIONS:
        if item["id"] == target:
            return item["name_en"] if is_en else item["name_tr"]
    return target or "-"


def question_bank(
    student_id: str,
    quiz_session: str,
    answer_tolerance: float = DEFAULT_ANSWER_ABS_TOL,
    language: str = DEFAULT_UI_LANGUAGE,
    slot_ids: list[str] | tuple[str, ...] | None = None,
) -> list[Question]:
    """Generate deterministic question list from selected slot ids."""
    r = rng_from_student(student_id, quiz_session)
    is_en = _normalize_language(language) == "en"
    tolerance = _sanitize_answer_tolerance(answer_tolerance)
    selected_slot_ids = sanitize_question_slot_ids(slot_ids, slot_count=QUIZ_SLOT_COUNT)
    questions: list[Question] = []
    for slot_id in selected_slot_ids:
        definition = QUESTION_DEFINITION_BY_ID[slot_id]
        question = definition["builder"](r, is_en, tolerance)
        validate_question(slot_id, question)
        questions.append(question)
    return questions
