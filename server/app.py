"""
Micro-SWE Gym — FastAPI server exposing the environment as REST API.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MicroSweGymAction, MicroSweGymObservation
from server.micro_swe_gym_environment import MicroSweGymEnvironment

app = FastAPI(
    title="Micro-SWE Gym",
    version="1.0.0",
    description="OpenEnv RESTful environment for automated PR review / code fixing",
)

# Global environment instances (task_id -> MicroSweGymEnvironment)
_envs: dict[int, MicroSweGymEnvironment] = {}


def _get_or_create_env(task_id: int) -> MicroSweGymEnvironment:
    """Get or create environment for task."""
    if task_id not in _envs:
        if not (0 <= task_id <= 2):
            raise ValueError(f"task_id must be in [0, 1, 2], got {task_id}")
        _envs[task_id] = MicroSweGymEnvironment(task_id=task_id)
    return _envs[task_id]

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Micro-SWE Gym is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/reset")
def reset(task_id: int = 0) -> JSONResponse:
    """Reset environment and return initial observation."""
    try:
        env = _get_or_create_env(task_id)
        obs = env.reset()
        return JSONResponse({
            "observation": obs.model_dump(),
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(task_id: int = 0, fixed_code: str = "") -> JSONResponse:
    """Execute one agent step."""
    if task_id not in _envs:
        raise HTTPException(
            status_code=400,
            detail=f"Environment for task_id={task_id} not initialized. Call /reset first.",
        )
    
    try:
        env = _envs[task_id]
        action_dict = {"fixed_code": fixed_code}
        obs, reward, done, info = env.step(action_dict)
        
        return JSONResponse({
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        })
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(task_id: int = 0) -> JSONResponse:
    """Get current environment state."""
    if task_id not in _envs:
        raise HTTPException(
            status_code=400,
            detail=f"Environment for task_id={task_id} not initialized. Call /reset first.",
        )
    
    
    return JSONResponse(_envs[task_id].state())

import uvicorn

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
