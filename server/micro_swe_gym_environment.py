"""
Micro-SWE Gym — Core OpenEnv environment.

Implements the three required methods:
    reset()  → SWEObservation
    step()   → (SWEObservation, float, bool, dict)
    state()  → dict

Reward scheme:
    0.0  — code does not compile
    0.2  — compiles but fails tests
    1.0  — all tests pass
"""
from __future__ import annotations

import traceback
import textwrap
from typing import Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MicroSweGymObservation
# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

TASKS: list[dict[str, Any]] = [
    # ── Task 0 · EASY ── fix a sign error ──────────────────────────────────
    {
        "difficulty": "easy",
        "broken_code": textwrap.dedent("""\
            def subtract(a, b):
                \"\"\"Return a minus b.\"\"\"
                return a + b          # BUG: should be a - b
        """),
        "solution_code": textwrap.dedent("""\
            def subtract(a, b):
                \"\"\"Return a minus b.\"\"\"
                return a - b
        """),
        # Each test is a (args, expected_return) tuple.
        "tests": [
            {"call": "subtract(10, 3)", "expected": 7},
            {"call": "subtract(0, 5)",  "expected": -5},
            {"call": "subtract(-1, -1)", "expected": 0},
        ],
    },

    # ── Task 1 · MEDIUM ── handle empty-list edge case ─────────────────────
    {
        "difficulty": "medium",
        "broken_code": textwrap.dedent("""\
            def average(nums):
                \"\"\"Return the arithmetic mean of a list of numbers.\"\"\"
                return sum(nums) / len(nums)   # BUG: crashes on empty list
        """),
        "solution_code": textwrap.dedent("""\
            def average(nums):
                \"\"\"Return the arithmetic mean of a list of numbers.\"\"\"
                if not nums:
                    return 0.0
                return sum(nums) / len(nums)
        """),
        "tests": [
            {"call": "average([1, 2, 3])",   "expected": 2.0},
            {"call": "average([10])",         "expected": 10.0},
            {"call": "average([])",           "expected": 0.0},
            {"call": "average([0, 0, 0])",    "expected": 0.0},
        ],
    },

    # ── Task 2 · HARD ── O(n²) → O(n) two-sum ─────────────────────────────
    {
        "difficulty": "hard",
        "broken_code": textwrap.dedent("""\
            def two_sum(nums, target):
                \"\"\"Return indices of the two numbers that add up to target.
                Guaranteed exactly one solution exists.
                Current implementation is O(n^2) — must be improved to O(n).
                \"\"\"
                n = len(nums)
                for i in range(n):
                    for j in range(i + 1, n):          # BUG: O(n^2)
                        if nums[i] + nums[j] == target:
                            return [i, j]
                return []
        """),
        "solution_code": textwrap.dedent("""\
            def two_sum(nums, target):
                \"\"\"Return indices of the two numbers that add up to target.
                O(n) single-pass hash-map solution.
                \"\"\"
                seen = {}
                for i, num in enumerate(nums):
                    complement = target - num
                    if complement in seen:
                        return [seen[complement], i]
                    seen[num] = i
                return []
        """),
        "tests": [
            {"call": "two_sum([2, 7, 11, 15], 9)",  "expected": [0, 1]},
            {"call": "two_sum([3, 2, 4], 6)",        "expected": [1, 2]},
            {"call": "two_sum([3, 3], 6)",            "expected": [0, 1]},
            {"call": "two_sum([1, 4, 8, 2], 6)",     "expected": [1, 3]},
        ],
    },
]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MicroSweGymEnvironment:
    """OpenEnv-compliant environment for automated PR review / code fixing."""

    def __init__(self, task_id: int = 0) -> None:
        assert 0 <= task_id < len(TASKS), f"task_id must be 0–{len(TASKS)-1}"
        self._task_id: int = task_id
        self._last_error: str = ""
        self._done: bool = False
        self._steps: int = 0
        self._last_reward: float = 0.0

    # ------------------------------------------------------------------
    # Public OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> MicroSweGymObservation:
        """Reset the environment and return the initial observation."""
        self._last_error = ""
        self._done = False
        self._steps = 0
        self._last_reward = 0.0
        obs_dict = self._observation()
        return MicroSweGymObservation(**obs_dict)

    def step(self, action: dict) -> tuple[MicroSweGymObservation, float, bool, dict]:
        """
        Execute one agent step.

        Parameters
        ----------
        action : dict
            Must contain key ``fixed_code`` (str).

        Returns
        -------
        observation : MicroSweGymObservation
        reward      : float  (0.0 / 0.2 / 1.0)
        done        : bool
        info        : dict
        """
        if self._done:
            raise RuntimeError("Environment is done — call reset() first.")

        fixed_code: str = action.get("fixed_code", "")
        self._steps += 1

        reward, error = self._evaluate(fixed_code)
        self._last_reward = reward
        self._last_error = error
        self._done = reward == 1.0  # succeed → terminal

        obs_dict = self._observation()
        obs = MicroSweGymObservation(**obs_dict)
        info = {
            "steps": self._steps,
            "task_id": self._task_id,
            "difficulty": TASKS[self._task_id]["difficulty"],
        }
        return obs, reward, self._done, info

    def state(self) -> dict:
        """Return a serialisable snapshot of the environment state."""
        task = TASKS[self._task_id]
        return {
            "task_id": self._task_id,
            "difficulty": task["difficulty"],
            "done": self._done,
            "steps": self._steps,
            "last_reward": self._last_reward,
            "last_error": self._last_error,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observation(self) -> dict:
        task = TASKS[self._task_id]
        return {
            "broken_code": task["broken_code"],
            "error_message": self._last_error,
            "difficulty": task["difficulty"],
            "task_id": self._task_id,
        }

    def _evaluate(self, code: str) -> tuple[float, str]:
        """
        Execute agent code in an isolated namespace and run all unit tests.

        Returns (reward, error_message).
        """
        namespace: dict[str, Any] = {}

        # ── Phase 1: compilation / exec ────────────────────────────────────
        try:
            exec(compile(code, "<agent_code>", "exec"), namespace)  # noqa: S102
        except SyntaxError as exc:
            return 0.0, f"SyntaxError: {exc}"
        except Exception:
            return 0.0, f"ExecError:\n{traceback.format_exc()}"

        # ── Phase 2: run unit tests ────────────────────────────────────────
        failures: list[str] = []
        for test in TASKS[self._task_id]["tests"]:
            call_expr: str = test["call"]
            expected = test["expected"]
            try:
                result = eval(call_expr, namespace)  # noqa: S307
                if result != expected:
                    failures.append(
                        f"  FAIL  {call_expr} → got {result!r}, expected {expected!r}"
                    )
            except Exception:
                failures.append(
                    f"  ERROR {call_expr}:\n{traceback.format_exc()}"
                )

        if failures:
            return 0.2, "Tests failed:\n" + "\n".join(failures)

        return 1.0, ""
