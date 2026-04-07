"""
Micro-SWE Gym — Pydantic data models.
Compliant with OpenEnv spec v1.
"""
from typing import Literal
from pydantic import BaseModel, Field


class MicroSweGymAction(BaseModel):
    """Action submitted by the agent: a string containing the fixed Python code."""

    fixed_code: str = Field(
        ...,
        description="The complete, corrected Python source code proposed by the agent.",
    )


class MicroSweGymObservation(BaseModel):
    """Observation returned by the environment after reset or step."""

    broken_code: str = Field(
        ...,
        description="The broken Python function the agent must repair.",
    )
    error_message: str = Field(
        default="",
        description="Compiler / runtime error from the last step, empty on reset.",
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ...,
        description="Difficulty tier of the current task.",
    )
    task_id: int = Field(
        ...,
        description="Index of the active task (0, 1, or 2).",
    )