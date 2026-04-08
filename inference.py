"""
Micro-SWE Gym — Baseline inference script.
Aligned with Meta OpenEnv Hackathon requirements.
"""
from __future__ import annotations

import os
import sys
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration with MANDATORY defaults & Fixes
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN: str = os.getenv("HF_TOKEN")

# FIX: Use your direct Space URL instead of localhost
DEFAULT_ENV_URL = "https://coderbug-micro-swe-gym.hf.space"
SERVER_URL: str = os.getenv("ENV_SERVER_URL", DEFAULT_ENV_URL).rstrip("/")
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "5"))

if not HF_TOKEN:
    print("Warning: HF_TOKEN not found in environment")

# ---------------------------------------------------------------------------
# Environment client helpers with RETRY LOGIC
# ---------------------------------------------------------------------------

def _reset(task_id: int = 0) -> dict:
    # Retry logic ensures we don't crash if the Space is waking up
    for i in range(10):
        try:
            r = requests.post(f"{SERVER_URL}/reset", json={"task_id": task_id}, timeout=30)
            r.raise_for_status()
            return r.json()["observation"]
        except Exception as e:
            print(f"Waiting for server at {SERVER_URL} (Attempt {i+1}/10)...")
            time.sleep(10)
    raise ConnectionError(f"Could not connect to environment at {SERVER_URL}")

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

SYSTEM_PROMPT = "You are an expert Python engineer. Return ONLY corrected Python code — no explanation, no markdown."

def _ask_llm(client: OpenAI, broken_code: str, error_message: str) -> str:
    user_content = f"Broken code:\n{broken_code}\n"
    if error_message:
        user_content += f"\nError: {error_message}\n"
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content or ""
    return raw.replace("```python", "").replace("```", "").strip()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(task_id: int = 0):
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    rewards_history = []
    
    print(f"[START] task={task_id} env=micro_swe_gym model={MODEL_NAME}")

    step_num = 0
    solved = False
    try:
        obs = _reset(task_id)
        done = False

        while not done and step_num < MAX_STEPS:
            step_num += 1
            fixed_code = _ask_llm(client, obs["broken_code"], obs.get("error_message", ""))
            action_snippet = fixed_code.replace("\n", "\\n")[:50]
            
            obs, reward, done, info = _step(fixed_code, task_id)
            rewards_history.append(reward)
            
            error_msg = obs.get("error_message", "null")
            if not error_msg: error_msg = "null"

            print(f"[STEP] step={step_num} action={action_snippet}... reward={reward:.2f} done={str(done).lower()} error={error_msg}")
            
            if reward == 1.0:
                solved = True

    except Exception as e:
        print(f"[END] success=false steps={step_num} rewards=0.00")
        print(f"DEBUG: {e}")
        return

    rewards_str = ",".join([f"{r:.2f}" for r in rewards_history])
    print(f"[END] success={str(solved).lower()} steps={step_num} rewards={rewards_str}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()
    run(task_id=args.task_id)
