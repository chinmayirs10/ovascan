"""Lifestyle guidance, based on the 2023 International Evidence-based
Guideline for the Assessment and Management of PCOS (Teede et al., 2023).

This module produces general, non-prescriptive guidance. It never suggests
medication or doses, and it always points people to a clinician.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GUIDELINE_REF = (
    "Teede HJ et al. Recommendations from the 2023 International Evidence-based "
    "Guideline for the Assessment and Management of Polycystic Ovary Syndrome. "
    "J Clin Endocrinol Metab. 2023;108(10):2447-2469."
)


@dataclass
class Profile:
    """Optional self-reported information. Every field may be left as None."""
    activity_min_per_week: int | None = None   # moderate-intensity equivalent
    strength_days_per_week: int | None = None
    sleep_hours: float | None = None
    irregular_cycles: bool | None = None
    acne_or_excess_hair: bool | None = None
    low_mood_or_anxiety: bool | None = None


@dataclass
class Section:
    title: str
    items: list[str] = field(default_factory=list)


@dataclass
class Plan:
    headline: str
    see_doctor: str
    sections: list[Section]
    reference: str = GUIDELINE_REF


def _see_doctor_text(band_key: str, p: Profile) -> str:
    symptoms = bool(p.irregular_cycles or p.acne_or_excess_hair)
    if band_key == "pcos_like" or (band_key == "inconclusive" and symptoms):
        return (
            "Please book an appointment with a gynaecologist or GP. PCOS is diagnosed "
            "with the Rotterdam criteria: at least two of irregular cycles, signs of high "
            "androgens (clinical or blood test) and polycystic ovaries on ultrasound "
            "(or high AMH in adults). One image cannot establish this."
        )
    if band_key == "inconclusive":
        return (
            "The result is unclear. If you have irregular periods, acne, excess hair "
            "growth or difficulty getting pregnant, please talk to a doctor."
        )
    if symptoms:
        return (
            "The image showed no PCOS-like pattern, but you reported symptoms that are "
            "worth discussing with a doctor; PCOS can be present without polycystic-looking ovaries."
        )
    return "No action needed based on this image. See a doctor if you notice cycle changes or new symptoms."


def build_plan(band_key: str, profile: Profile | None = None) -> Plan:
    if band_key not in {"normal", "inconclusive", "pcos_like"}:
        raise ValueError(f"unknown band: {band_key}")
    p = profile or Profile()

    move = Section("Move more", [
        "Aim for 150-300 minutes a week of moderate activity (e.g. brisk walking, cycling) "
        "or 75-150 minutes of vigorous activity, or a mix.",
        "Add muscle-strengthening exercise on 2 non-consecutive days a week.",
        "Break up long periods of sitting; any activity counts.",
    ])
    if p.activity_min_per_week is not None:
        if p.activity_min_per_week < 150:
            gap = 150 - p.activity_min_per_week
            move.items.insert(0, f"You reported about {p.activity_min_per_week} min/week. "
                                 f"Adding ~{gap} min/week (e.g. {max(1, round(gap / 5))} min on 5 days) "
                                 "would reach the minimum target.")
        else:
            move.items.insert(0, f"You already reach the activity target "
                                 f"({p.activity_min_per_week} min/week). Keep it up.")
    if p.strength_days_per_week is not None and p.strength_days_per_week < 2:
        move.items.append("You reported fewer than 2 strength days; bodyweight exercises at home are a simple start.")

    eat = Section("Eat well", [
        "No single diet is proven best for PCOS. Choose a balanced pattern you can keep up: "
        "vegetables, fruit, whole grains, legumes, nuts and lean proteins.",
        "Limit highly processed foods and sugary drinks.",
        "Avoid very restrictive diets unless supervised by a professional.",
    ])
    if band_key == "pcos_like":
        eat.items.append(
            "If you are carrying extra weight, a modest loss of 5-10% over about 6 months "
            "can noticeably improve cycles and metabolic health, but health gains from "
            "activity and diet happen regardless of weight change."
        )

    rest = Section("Sleep and wellbeing", [
        "Most adults need 7-9 hours of sleep. Keep a regular sleep schedule.",
        "Stress, low mood and anxiety are more common with PCOS. Talking to someone helps, "
        "and it is fine to ask a doctor about it.",
    ])
    if p.sleep_hours is not None and p.sleep_hours < 7:
        rest.items.insert(0, f"You reported about {p.sleep_hours:g} h of sleep. "
                             "Try moving bedtime earlier in 15-minute steps.")
    if p.low_mood_or_anxiety:
        rest.items.insert(0, "You mentioned low mood or anxiety. Please raise this with your "
                             "doctor; the PCOS guideline recommends screening for it.")

    sections = [move, eat, rest]

    if band_key in {"pcos_like", "inconclusive"}:
        sections.append(Section("Questions to ask your doctor", [
            "Do I meet the Rotterdam criteria for PCOS?",
            "Should I have blood tests for androgens, glucose (OGTT or HbA1c) and cholesterol?",
            "How often should my blood pressure and weight be checked?",
            "What are my options if I want to regulate my cycle or plan a pregnancy?",
        ]))

    headline = {
        "normal": "Healthy-habits plan",
        "inconclusive": "Healthy-habits plan while you get a clearer answer",
        "pcos_like": "Lifestyle plan to discuss with your doctor",
    }[band_key]

    return Plan(headline, _see_doctor_text(band_key, p), sections)
