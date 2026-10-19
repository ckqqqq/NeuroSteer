"""Serve real NeuroSteer inference, with a useful status when assets are missing."""
from contextlib import asynccontextmanager
import logging
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from demo.backend.cache_config import discover_caches

TASK_DIRECTIONS = Literal[
    "sentiment_pos-neg", "sentiment_neg-pos", "toxicity_pos-neg", "toxicity_neg-pos",
    "debate_pos-neg", "debate_neg-pos", "polite_pos-neg", "polite_neg-pos",
]


class RequestData(BaseModel):
    input_dict: dict[TASK_DIRECTIONS, float] = Field(default_factory=dict)
    input_str: str = Field(min_length=1, max_length=4000)

    @field_validator("input_str")
    @classmethod
    def validate_prompt(cls, prompt):
        if not prompt.strip():
            raise ValueError("The prompt cannot be blank.")
        return prompt

    @field_validator("input_dict")
    @classmethod
    def validate_strengths(cls, strengths):
        # The frontend sends nonnegative magnitudes; direction is encoded in the key.
        if any(not 0 <= strength <= 300 for strength in strengths.values()):
            raise ValueError("Steering magnitudes must be finite and between 0 and 300.")
        return strengths


@asynccontextmanager
async def lifespan(application):
    application.state.engine = None
    application.state.load_error = None
    application.state.inference_lock = Lock()
    try:
        # Check local assets before importing heavy libraries or downloading models.
        discover_caches()
        from demo.backend.neurosteer_backend_utils import (
            model_preprocess, delta_h, neuron_steer_generate,
        )
        model, sae, tokenizer = model_preprocess("gpt2-small", 6)
        feature_vectors = delta_h(sae, "gpt2-small", 6)
        application.state.engine = (
            neuron_steer_generate, model, sae, tokenizer, feature_vectors,
        )
        logging.info("NeuroSteer model and feature vectors are ready.")
    except Exception as error:
        logging.exception("NeuroSteer inference is unavailable")
        application.state.load_error = str(error)
    yield
    application.state.engine = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    ready = app.state.engine is not None
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "unavailable",
                 "detail": app.state.load_error},
    )


@app.post("/process")
def process_data(data: RequestData):
    if app.state.engine is None:
        raise HTTPException(status_code=503, detail=app.state.load_error or "Model not ready.")
    generate, model, sae, tokenizer, feature_vectors = app.state.engine
    # Hooks mutate shared model state; serialize requests to prevent cross-user mixing.
    with app.state.inference_lock:
        result = generate(data.input_dict, data.input_str, model, sae, tokenizer, feature_vectors)
    return {"result": result}
