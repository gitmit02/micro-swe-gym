import os
import sys
import time
import requests
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
HF_TOKEN: str = os.getenv("HF_TOKEN")
SERVER_URL = os.getenv("ENV_SERVER_URL", "http://127.0.0.1:7860").rstrip("/")
MAX_STEPS: int = 5

def _reset(task_id: int):
    # Retry loop to wait for the Hugging Face container to wake up
    for i in range(10):
        try:
            r = requests.post(f"{SERVER_URL}/reset", params={"task_id": task_id}, timeout=30)
            r.raise_for_status()
            return r.json()["observation"]
        except Exception:
            print(f"Waiting for server (Attempt {i+1}/10)...")
            time.sleep(10)
    raise ConnectionError("Server unreachable")

def _step(fixed_code: str, task_id: int):
    r = requests.post(f"{SERVER_URL}/step", json={"fixed_code": fixed_code}, params={"task_id": task_id}, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["observation"], data["reward"], data["done"], data["info"]

def _ask_llm(client, broken_code, error_message):
    prompt = f"Fix the following Python code.\n\nBroken Code:\n{broken_code}\n\nError Message:\n{error_message}\n\nReturn ONLY the corrected code without any explanation or markdown backticks."
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are a senior Python engineer."},
                  {"role": "user", "content": prompt}],
        temperature=0.1
    )
    raw = response.choices[0].message.content or ""
    # Clean up any accidental markdown backticks the model might add
    return raw.replace("```python", "").replace("```", "").strip()

def run(task_id: int):
    try:
        if not HF_TOKEN: raise ValueError("Missing HF_TOKEN")
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        
        print(f"[START] task={task_id} env=micro_swe_gym model={MODEL_NAME}")
        obs = _reset(task_id)
        
        step_num = 0
        done = False
        rewards_history = []

        while not done and step_num < MAX_STEPS:
            step_num += 1
            fixed_code = _ask_llm(client, obs["broken_code"], obs.get("error_message", ""))
            obs, reward, done, info = _step(fixed_code, task_id)
            
            # The rewards are now 0.05, 0.2, or 0.95 (Perfect range!)
            rewards_history.append(reward)
            print(f"[STEP] step={step_num} reward={reward:.2f} done={str(done).lower()}")

        # Final Summary for the validator
        rewards_str = ",".join([f"{r:.2f}" for r in rewards_history])
        success_str = "true" if any(r >= 0.9 for r in rewards_history) else "false"
        print(f"[END] success={success_str} steps={step_num} rewards={rewards_str}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        # Send a safe default reward so the validator doesn't crash
        print(f"[END] success=false steps=0 rewards=0.05")

if __name__ == "__main__":
    # REQUIRED: Run all three tasks defined in Codex's new environment
    for tid in [0, 1, 2]:
        run(task_id=tid)
