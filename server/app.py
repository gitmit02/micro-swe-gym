from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import sys
import os
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MicroSweGymAction, MicroSweGymObservation
from server.micro_swe_gym_environment import MicroSweGymEnvironment

app = FastAPI(title="Micro-SWE Gym")

# Health Check Routes
@app.get("/")
async def root():
    return {"status": "healthy", "message": "Micro-SWE Gym is Live!"}

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

# Global environment instances
_envs: dict[int, MicroSweGymEnvironment] = {}

def _get_or_create_env(task_id: int) -> MicroSweGymEnvironment:
    if task_id not in _envs:
        if not (0 <= task_id <= 2):
            raise ValueError(f"task_id must be in [0, 1, 2], got {task_id}")
        _envs[task_id] = MicroSweGymEnvironment(task_id=task_id)
    return _envs[task_id]

@app.post("/reset")
def reset(task_id: int = 0) -> JSONResponse:
    try:
        env = _get_or_create_env(task_id)
        obs = env.reset()
        return JSONResponse({"observation": obs.model_dump()})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(task_id: int = 0, fixed_code: str = "") -> JSONResponse:
    if task_id not in _envs:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    try:
        env = _envs[task_id]
        obs, reward, done, info = env.step({"fixed_code": fixed_code})
        return JSONResponse({
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def main():
    # MANDATORY: Hugging Face must use port 7860
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
