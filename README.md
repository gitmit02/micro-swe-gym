# Micro-SWE Gym 🚀

OpenEnv environment for the **Meta × SST Hackathon Round 1**.  
**Micro-SWE Gym** acts as an Automated PR Reviewer/Fixer. Given a broken Python function, an AI agent must return corrected code that passes a hidden suite of unit tests to earn a reward.

---

## 📂 File Structure

```text
micro_swe_gym/
├── models.py                # MicroSweGymAction + MicroSweGymObservation models
├── inference.py             # Baseline agent (OpenAI-compatible client)
├── openenv.yaml             # OpenEnv spec v1 metadata
├── pyproject.toml           # Project dependencies and entry points
├── README.md                # Project documentation
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI entry point with main()
│   ├── Dockerfile           # Deployment container config
│   ├── requirements.txt     # Backend dependencies
│   └── micro_swe_gym_environment.py  # Core env logic (reset/step/state)