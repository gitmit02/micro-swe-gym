import os
import sys
import time
import requests
from openai import OpenAI

# Configuration
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
HF_TOKEN: str = os.getenv("HF_TOKEN")
# Default to local evaluator container/server. Override with ENV_SERVER_URL if needed.
SERVER_URL = os.getenv("ENV_SERVER_URL", "http://127.0.0.1:7860").rstrip("/")
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "5"))

def _reset(task_id: int = 0) -> dict:
    for i in range(10): # 100-second retry window
        try:
            r = requests.post(f"{SERVER_URL}/reset", json={"task_id": task_id}, timeout=30)
            r.raise_for_status()
            return r.json()["observation"]
        except:
            print(f"Waiting for server (Attempt {i+1}/10)...")
            time.sleep(10)
    raise ConnectionError("Server unreachable")

def _step(fixed_code: str, task_id: int = 0) -> tuple[dict, float, bool, dict]:
    r = requests.post(f"{SERVER_URL}/step", json={"fixed_code": fixed_code}, params={"task_id": task_id}, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["observation"], data["reward"], data["done"], data["info"]

def _ask_llm(client, broken_code, error_message):
    user_content = f"Broken code:\n{broken_code}\nError: {error_message}"
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "Return ONLY corrected Python code."},
                  {"role": "user", "content": user_content}],
        temperature=0.1,
    )
    raw = response.choices[0].message.content or ""
    return raw.replace("```python", "").replace("```", "").strip()

def run(task_id: int = 0):
    step_num = 0
    solved = False
    rewards_history = []
    try:
        if not HF_TOKEN: raise ValueError("Missing HF_TOKEN")
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        print(f"[START] task={task_id} env=micro_swe_gym model={MODEL_NAME}")
        
        time.sleep(5) # Grace period
        obs = _reset(task_id)
        done = False

        while not done and step_num < MAX_STEPS:
            step_num += 1
            fixed_code = _ask_llm(client, obs["broken_code"], obs.get("error_message", ""))
            obs, reward, done, info = _step(fixed_code, task_id)
            rewards_history.append(reward)
            print(f"[STEP] step={step_num} reward={reward:.2f} done={str(done).lower()}")
            if reward == 1.0: solved = True

        rewards_str = ",".join([f"{r:.2f}" for r in rewards_history])
        print(f"[END] success={str(solved).lower()} steps={step_num} rewards={rewards_str}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        print(f"[END] success=false steps={step_num} rewards=0.00")
        sys.exit(0) # IMPORTANT: Prevents the "non-zero status code" error

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()
    run(task_id=args.task_id)
