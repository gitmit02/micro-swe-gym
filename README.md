# Micro-SWE Gym 🚀

OpenEnv environment for the Meta × SST Hackathon Round 1.

**Micro-SWE Gym** acts as an Automated PR Reviewer/Fixer. Given a broken Python function, an AI agent must return corrected code that passes a hidden suite of unit tests to earn a reward.

## 📂 Project Structure
micro_swe_gym/
├── Dockerfile               # Deployment container config (Moved to Root)
├── requirements.txt         # Backend dependencies (Moved to Root)
├── models.py                # MicroSweGymAction + MicroSweGymObservation models
├── inference.py             # Baseline agent (Meta OpenEnv Compliant)
├── openenv.yaml             # OpenEnv spec v1 metadata
├── pyproject.toml           # Project dependencies and entry points
├── README.md                # Project documentation
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI entry point
│   └── micro_swe_gym_environment.py  # Core env logic (reset/step/state)

## 🚀 Baseline Performance
- **Task 0 (Easy):** 1.0 Reward (Deterministic fix)
- **Task 1 (Medium):** Variable Reward based on logic correction
- **Task 2 (Hard):** Variable Reward based on complexity
