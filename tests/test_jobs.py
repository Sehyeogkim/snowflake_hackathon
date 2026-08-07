from mavis.jobs import plan_seed_jobs


def test_job_planner_is_importable() -> None:
    assert callable(plan_seed_jobs)
