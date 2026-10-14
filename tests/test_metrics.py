from driveact.environments.drive_env import EpisodeRecord
from driveact.evaluation.metrics import summarize, wilson


def _rec(outcome, curb=False, groups=2, passed=2):
    return EpisodeRecord(0, 0, 1, 0, outcome, outcome == "success", outcome == "success", curb, 100, 10.0, 100.0, 1.0,
                         0.1, 0.3, False, 0.0, 0.01, 0.02, 0.001, 1.0, groups, passed, 8.0)


def test_wilson_bounds():
    lo, hi = wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.98
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # nan


def test_summary_counts():
    s = summarize([_rec("success"), _rec("collision_cone", passed=1), _rec("success", curb=True)])
    assert s["goal_completion"]["count"] == 2 and s["collision"]["count"] == 1
    assert s["cone_collision"]["count"] == 1 and s["curb_strike"]["count"] == 1
    assert s["obstacle_avoidance"]["passed"] == 5 and s["obstacle_avoidance"]["total"] == 6
