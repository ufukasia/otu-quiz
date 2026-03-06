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
    fabric_length = r.choice([8, 10, 12, 15])
    weights = r.choice(
        [
            (44, 32, 15, 6, 3),
            (38, 35, 17, 7, 3),
            (41, 30, 18, 8, 3),
            (36, 37, 17, 7, 3),
            (42, 29, 20, 6, 3),
            (39, 34, 18, 6, 3),
        ]
    )
    x_values = [0, 1, 2, 3, 4]
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
                f"In a shipment of {total} TVs, {defective} are defective. A hotel randomly buys {draw} TVs.\n"
                "Let X be number of defective TVs selected. Find P(X>=1)."
            )
            if is_en
            else (
                f"{total} televizyonluk sevkiyatta {defective} tanesi arızalıdır. Bir otel rastgele {draw} televizyon alıyor.\n"
                "X seçilen arızalı TV sayısı olsun. P(X>=1) değerini bulunuz."
            )
        ),
        "answer": answer,
        "tolerance": tolerance,
        "visual": {
            "kind": "pmf_table",
            "x_values": x_values,
            "p_values": probs,
            "population_size": total,
            "defective_count": defective,
            "draw_count": draw,
            "caption": "PMF: X=arızalı TV sayısı" if not is_en else "PMF: X=number of defective TVs",
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
                f"Error count X in {code_lines} lines of code has the PMF shown in the table.\n"
                "Using Var(X)=E(X^2)-[E(X)]^2, find Var(X)."
            )
            if is_en
            else (
                f"{code_lines} satırlık kod bloğundaki hata sayısı X'in PMF tablosu aşağıdadır.\n"
                "Var(X)=E(X^2)-[E(X)]^2 formülüyle Var(X) değerini bulunuz."
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
                f"In {n} independent sales, each sale has airbag probability 0.50.\n"
                "Let X be number of airbag-equipped sales. Find P(X=2)."
            )
            if is_en
            else (
                f"{n} bağımsız satışta her satışın hava yastıklı olma olasılığı 0.50 olsun.\n"
                "X hava yastıklı satış sayısı olsun. P(X=2) değerini bulunuz."
            )
        ),
        "answer": answer,
        "tolerance": tolerance,
        "visual": {
            "kind": "pmf_bar",
            "x_values": x_values,
            "p_values": probs,
            "highlight_x": k,
            "trial_count": n,
            "success_prob": p,
            "caption": "Binom PMF" if is_en else "Binom PMF",
        },
    }


def _build_q_vacuum_pdf_prob(r: random.Random, is_en: bool, tolerance: float) -> Question:
    threshold = r.choice([1.10, 1.20, 1.30, 1.40])
    answer = 2 * threshold - (threshold**2) / 2 - 1
    return {
        "title": "Vacuum Usage - Piecewise PDF" if is_en else "Elektrik Süpürgesi - Parçalı PDF",
        "text": (
            (
                "Density is piecewise:\n"
                "f(x)=x for 0<x<1, f(x)=2-x for 1<=x<2, 0 otherwise.\n"
                f"Find P(X<{threshold:.2f})."
            )
            if is_en
            else (
                "Yoğunluk parçalı:\n"
                "0<x<1 için f(x)=x, 1<=x<2 için f(x)=2-x, diğer yerde 0.\n"
                f"P(X<{threshold:.2f}) değerini bulunuz."
            )
        ),
        "answer": answer,
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
                f"For defect count X in {fabric_length}m synthetic fabric, the PMF is shown in the table.\n"
                "Find expected value E(X)."
            )
            if is_en
            else (
                f"{fabric_length}m sentetik kumaştaki kusur sayısı X'in PMF tablosu aşağıdadır.\n"
                "Beklenen değeri E(X) hesaplayınız."
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
    _ensure(support == list(range(0, len(support))), question_id, "support must be 0..4 defect counts")
    expected_answer = sum(x * p for x, p in zip(x_values, p_values))
    _ensure(_approx_equal(answer, expected_answer), question_id, "answer must equal expected value E(X)")


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
