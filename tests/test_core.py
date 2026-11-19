import numpy as np
import pytest

from pcos_screen.data import group_split
from pcos_screen.lifestyle import Profile, build_plan
from pcos_screen.model import band_for


@pytest.mark.parametrize("p,key", [(0.0, "normal"), (0.29, "normal"), (0.3, "inconclusive"),
                                   (0.5, "inconclusive"), (0.7, "inconclusive"), (0.71, "pcos_like"),
                                   (1.0, "pcos_like")])
def test_band_thresholds(p, key):
    assert band_for(p).key == key


def test_band_rejects_bad_probability():
    with pytest.raises(ValueError):
        band_for(1.5)


def test_group_split_never_leaks_a_family():
    rng = np.random.default_rng(0)
    families = np.repeat(np.arange(40), 10)
    rng.shuffle(families)  # order shouldn't matter
    labels = (families % 2).astype(int)
    split = group_split(families, labels, seed=1)
    for f in np.unique(families):
        assert len(set(split[families == f])) == 1
    assert set(split) == {"train", "val", "test"}


def test_plan_always_points_to_doctor_for_pcos_like():
    plan = build_plan("pcos_like")
    assert "Rotterdam" in plan.see_doctor
    assert any(s.title.startswith("Questions") for s in plan.sections)


def test_plan_personalises_activity_gap():
    plan = build_plan("normal", Profile(activity_min_per_week=60))
    move = plan.sections[0]
    assert "60 min/week" in move.items[0]


def test_symptoms_escalate_normal_result():
    plan = build_plan("normal", Profile(irregular_cycles=True))
    assert "doctor" in plan.see_doctor.lower()


def test_plan_never_mentions_medication():
    for key in ("normal", "inconclusive", "pcos_like"):
        text = str(build_plan(key, Profile(low_mood_or_anxiety=True))).lower()
        for word in ("metformin", "mg", "dose", "pill"):
            assert word not in text
