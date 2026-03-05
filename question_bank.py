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
            "caption": "PMF: X=arızalı TV sayısı" if not is_en else "PMF: X=number of defective TVs",
        },
    }


def _build_q_software_bug_variance(r: random.Random, is_en: bool, tolerance: float) -> Question:
    probs = [(2, 0.01), (3, 0.25), (4, 0.40), (5, 0.30), (6, 0.04)]
    ex = sum(x * p for x, p in probs)
    ex2 = sum((x**2) * p for x, p in probs)
    answer = ex2 - ex * ex
    return {
        "title": "Software Error Analysis - Variance" if is_en else "Yazılım Hata Analizi - Varyans",
        "text": (
            (
                "Error count X in 100 lines has PMF shown in table.\n"
                "Using Var(X)=E(X^2)-[E(X)]^2, find Var(X)."
            )
            if is_en
            else (
                "100 satır koddaki hata sayısı X'in PMF tablosu aşağıdadır.\n"
                "Var(X)=E(X^2)-[E(X)]^2 formülüyle Var(X) değerini bulunuz."
            )
        ),
        "answer": answer,
        "tolerance": tolerance,
        "visual": {
            "kind": "pmf_table",
            "x_values": [2, 3, 4, 5, 6],
            "p_values": [0.01, 0.25, 0.40, 0.30, 0.04],
            "caption": "PMF: Yazılım hatası sayısı" if not is_en else "PMF: Software error count",
        },
    }


def _build_q_factory_total_probability(r: random.Random, is_en: bool, tolerance: float) -> Question:
    p_b1, p_b2, p_b3 = (0.30, 0.45, 0.25)
    p_a_b1, p_a_b2, p_a_b3 = (0.02, 0.03, 0.02)
    answer = p_b1 * p_a_b1 + p_b2 * p_a_b2 + p_b3 * p_a_b3
    return {
        "title": "Assembly Factory - Total Probability" if is_en else "Montaj Fabrikası - Toplam Olasılık",
        "text": (
            (
                "Machine shares: P(B1)=0.30, P(B2)=0.45, P(B3)=0.25.\n"
                "Defect rates: P(A|B1)=0.02, P(A|B2)=0.03, P(A|B3)=0.02.\n"
                "Find P(A) with Total Probability."
            )
            if is_en
            else (
                "Makine payları: P(B1)=0.30, P(B2)=0.45, P(B3)=0.25.\n"
                "Hata oranları: P(A|B1)=0.02, P(A|B2)=0.03, P(A|B3)=0.02.\n"
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
    p_b1, p_b2, p_b3 = (0.30, 0.45, 0.25)
    p_a_b1, p_a_b2, p_a_b3 = (0.02, 0.03, 0.02)
    p_a = p_b1 * p_a_b1 + p_b2 * p_a_b2 + p_b3 * p_a_b3
    answer = (p_b3 * p_a_b3) / p_a
    return {
        "title": "Defect Source - Bayes" if is_en else "Hata Kaynağı - Bayes",
        "text": (
            (
                "Assembly factory data: P(B1)=0.30, P(B2)=0.45, P(B3)=0.25 and P(A|Bi)=(0.02,0.03,0.02).\n"
                "Given product is defective (A), find P(B3|A)."
            )
            if is_en
            else (
                "Montaj fabrikası verisi: P(B1)=0.30, P(B2)=0.45, P(B3)=0.25 ve P(A|Bi)=(0.02,0.03,0.02).\n"
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
    x_values = [0, 1, 2, 3, 4]
    p_values = [0.41, 0.37, 0.16, 0.05, 0.01]
    answer = sum(x * p for x, p in zip(x_values, p_values))
    return {
        "title": "Fabric Defect Count - Expected Value" if is_en else "Kumaş Kusur Sayısı - Beklenen Değer",
        "text": (
            (
                "For defect count X in 10m synthetic fabric, PMF is in the table.\n"
                "Find expected value E(X)."
            )
            if is_en
            else (
                "10m sentetik kumaştaki kusur sayısı X'in PMF tablosu aşağıdadır.\n"
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


QUESTION_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "backup_power_union",
        "name_tr": "Yedek Güç Ünitesi (Bağımsızlık)",
        "name_en": "Backup Power (Independence)",
        "builder": _build_q_backup_power_union,
    },
    {
        "id": "hotel_chain_total_probability",
        "name_tr": "Otel Zinciri (Toplam Olasılık)",
        "name_en": "Hotel Chain (Total Probability)",
        "builder": _build_q_hotel_chain_total_probability,
    },
    {
        "id": "medical_test_bayes",
        "name_tr": "Tıbbi Test (Bayes)",
        "name_en": "Medical Test (Bayes)",
        "builder": _build_q_medical_test_bayes,
    },
    {
        "id": "tv_sets_p_ge_1",
        "name_tr": "Televizyon Setleri (PMF/CDF)",
        "name_en": "TV Sets (PMF/CDF)",
        "builder": _build_q_tv_sets_p_ge_1,
    },
    {
        "id": "software_bug_variance",
        "name_tr": "Yazılım Hata Analizi (Varyans)",
        "name_en": "Software Error Analysis (Variance)",
        "builder": _build_q_software_bug_variance,
    },
    {
        "id": "factory_total_probability",
        "name_tr": "Montaj Fabrikası (Toplam Olasılık)",
        "name_en": "Assembly Factory (Total Probability)",
        "builder": _build_q_factory_total_probability,
    },
    {
        "id": "factory_bayes_machine3",
        "name_tr": "Hata Kaynağı (Bayes)",
        "name_en": "Defect Source (Bayes)",
        "builder": _build_q_factory_bayes_machine3,
    },
    {
        "id": "airbag_pmf_exact_two",
        "name_tr": "Hava Yastığı Satışı (PMF)",
        "name_en": "Airbag Sales (PMF)",
        "builder": _build_q_airbag_pmf_exact_two,
    },
    {
        "id": "vacuum_pdf_prob",
        "name_tr": "Elektrik Süpürgesi (Parçalı PDF)",
        "name_en": "Vacuum Usage (Piecewise PDF)",
        "builder": _build_q_vacuum_pdf_prob,
    },
    {
        "id": "circuit_reliability",
        "name_tr": "Seri-Paralel Devre",
        "name_en": "Series-Parallel Circuit",
        "builder": _build_q_circuit_reliability,
    },
    {
        "id": "fabric_expected_value",
        "name_tr": "Kumaş Kusur Sayısı (Beklenen Değer)",
        "name_en": "Fabric Defect Count (Expected Value)",
        "builder": _build_q_fabric_expected_value,
    },
)


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
    builders = {item["id"]: item["builder"] for item in QUESTION_DEFINITIONS}
    return [builders[slot_id](r, is_en, tolerance) for slot_id in selected_slot_ids]
