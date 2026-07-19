from app.agent.planner_agent import PlannerAgent


def test_build_user_prompt_without_memory() -> None:
    prompt = PlannerAgent._build_user_prompt(
        user_message="Find a dermatologist in Bellevue",
    )

    assert prompt == (
        "Current user request:\n"
        "Find a dermatologist in Bellevue"
    )


def test_build_user_prompt_with_memory() -> None:
    prompt = PlannerAgent._build_user_prompt(
        user_message="Find a dermatologist in Bellevue",
        memory_context=(
            "Previous request:\n"
            "Find a dermatologist in Seattle"
        ),
    )

    assert "Current user request:" in prompt
    assert "Find a dermatologist in Bellevue" in prompt
    assert "Relevant successful past executions:" in prompt
    assert "Find a dermatologist in Seattle" in prompt


def test_build_user_prompt_ignores_empty_memory() -> None:
    prompt = PlannerAgent._build_user_prompt(
        user_message="Find a dermatologist",
        memory_context="",
    )

    assert "Relevant successful past executions:" not in prompt