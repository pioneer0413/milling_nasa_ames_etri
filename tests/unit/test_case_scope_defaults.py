from milling_experiment_framework.experiments import domain_shift_execution
from milling_experiment_framework.experiments import s1_segment_execution


EXPECTED_CASE_SCOPE = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def test_h2_leave_one_case_scope_excludes_case_6_only():
    assert s1_segment_execution.CASE_SCOPE == EXPECTED_CASE_SCOPE
    assert 6 not in s1_segment_execution.CASE_SCOPE
    assert len(s1_segment_execution.SHIFT_SCENARIOS) == len(EXPECTED_CASE_SCOPE)
    assert ("train_without_case_11", "case_11") in s1_segment_execution.SHIFT_SCENARIOS


def test_domain_shift_leave_one_case_scope_matches_h2_default():
    assert domain_shift_execution.CASE_SCOPE == EXPECTED_CASE_SCOPE
    assert domain_shift_execution.SHIFT_SCENARIOS == s1_segment_execution.SHIFT_SCENARIOS
