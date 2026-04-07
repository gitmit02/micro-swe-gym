"""
Micro-SWE Gym — Baseline inference script.

Reads configuration from environment variables:
    API_BASE_URL   Base URL of the OpenAI-compatible endpoint (required)
    MODEL_NAME     Model identifier                              (required)
    HF_TOKEN       Hugging Face / bearer token                  (required)

Emits structured logs:
    [START] — once at the beginning
    [STEP]  — after every environment step
    [END]   — once at termination with final reward

Usage:
    export API_BASE_URL="https://api-inference.huggingface.co/models/<org>/<model>/v1"
    export MODEL_NAME="<org>/<model>"
    export HF_TOKEN="hf_..."
    python inference.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("inference")


def _log(tag: str, payload: dict) -> None:
    """Emit a structured JSON log line with a required tag."""
    log.info("%s %s", tag, json.dumps(payload))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "").rstrip("/")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
SERVER_URL: str = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")
MAX_STEPS: int = int(os.environ.get("MAX_STEPS", "5"))


def _validate_env() -> None:
    missing = [v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN") if not os.environ.get(v)]
    if missing:
        log.error("Missing required environment variables: %s", ", ".join(missing))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Environment client helpers
# ---------------------------------------------------------------------------

def _reset(task_id: int = 0) -> dict:
    r = requests.post(f"{SERVER_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()["observation"]


def _step(fixed_code: str, task_id: int = 0) -> tuple[dict, float, bool, dict]:
    r = requests.post(
        f"{SERVER_URL}/step",
        json={"fixed_code": fixed_code},
        params={"task_id": task_id},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return data["observation"], data["reward"], data["done"], data["info"]


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Python software engineer.
You will be given a broken Python function.
Your task: return ONLY the corrected Python source code — no explanation, no markdown fences.
The code must be syntactically valid and pass all unit tests."""


def _ask_llm(client: OpenAI, broken_code: str, error_message: str, difficulty: str) -> str:
    """Call the model and extract the fixed code string."""
    user_content = (
        f"Difficulty: {difficulty}\n\n"
        f"Broken code:\n```python\n{broken_code}\n```\n"
    )
    if error_message:
        user_content += f"\nPrevious error:\n{error_message}\n"
    user_content += "\nReturn ONLY the corrected Python code."

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    raw: str = response.choices[0].message.content or ""
    # Strip any accidental markdown fences the model might emit
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        # drop first and last fence lines
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        raw = "\n".join(inner)
    return raw.strip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(task_id: int = 0) -> float:
    """Run the agent on one task. Returns the final reward."""
    _validate_env()

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # ── [START] ─────────────────────────────────────────────────────────────
    _log("[START]", {
        "task_id": task_id,
        "model": MODEL_NAME,
        "server": SERVER_URL,
        "max_steps": MAX_STEPS,
    })

    obs = _reset(task_id)
    _log("[STEP]", {"event": "reset", "difficulty": obs["difficulty"], "task_id": task_id})

    reward: float = 0.0
    done: bool = False
    step_num: int = 0

    while not done and step_num < MAX_STEPS:
        step_num += 1
        t0 = time.monotonic()

        fixed_code = _ask_llm(
            client,
            broken_code=obs["broken_code"],
            error_message=obs.get("error_message", ""),
            difficulty=obs["difficulty"],
        )

        obs, reward, done, info = _step(fixed_code, task_id)
        elapsed = round(time.monotonic() - t0, 3)

        _log("[STEP]", {
            "step": step_num,
            "reward": reward,
            "done": done,
            "elapsed_s": elapsed,
            "task_id": task_id,
            "difficulty": obs["difficulty"],
            "error": obs.get("error_message", "")[:200] or None,
        })

    # ── [END] ───────────────────────────────────────────────────────────────
    _log("[END]", {
        "task_id": task_id,
        "total_steps": step_num,
        "final_reward": reward,
        "solved": done,
    })

    return reward


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Micro-SWE Gym — baseline agent")
    parser.add_argument("--task-id", type=int, default=0, choices=[0, 1, 2],
                        help="Task to solve: 0=easy, 1=medium, 2=hard")
    args = parser.parse_args()

    final_reward = run(task_id=args.task_id)
    sys.exit(0 if final_reward == 1.0 else 1)