"""Serve one-hour demand forecasts from an existing training artifact."""

import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Annotated
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field, NaiveDatetime, field_validator
from threadpoolctl import threadpool_limits

from bike_demand.features import make_features
from bike_demand.request_logging import jsonl_logger, request_details

HISTORY_HOURS = 168
Count = Annotated[int, Field(strict=True, ge=0, le=2**53 - 1)]
EXAMPLE_REQUEST = {
    "target_hour": "2012-10-01T00:00:00",
    "recent_counts": [25] * HISTORY_HOURS,
}


class ForecastRequest(BaseModel):
    """Counts occupy fixed clock hours; a missing observation stays null."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"example": EXAMPLE_REQUEST}
    )
    target_hour: NaiveDatetime = Field(
        description="Hour to predict, using the UCI dataset's timezone-naive clock."
    )
    recent_counts: list[Count | None] = Field(
        min_length=HISTORY_HOURS,
        max_length=HISTORY_HOURS,
        description=(
            "168 hourly counts, oldest first, from target_hour minus 168 hours "
            "through target_hour minus 1 hour. Use null for missing hours; "
            "the first and last counts must be known."
        ),
    )

    @field_validator("target_hour")
    @classmethod
    def validate_target_hour(cls, value: datetime) -> datetime:
        """Require an exact clock hour with a representable history window."""
        if value.minute or value.second or value.microsecond:
            raise ValueError("target_hour must be aligned to a whole hour.")
        try:
            pd.date_range(end=value, periods=HISTORY_HOURS + 1, freq="h")
        except (pd.errors.OutOfBoundsDatetime, OverflowError):
            raise ValueError("target_hour is outside the supported calendar.") from None
        return value

    @field_validator("recent_counts")
    @classmethod
    def validate_required_counts(cls, values: list[int | None]) -> list[int | None]:
        """Use the same required lag observations as training."""
        if values[0] is None or values[-1] is None:
            raise ValueError(
                "Counts at target_hour minus 168h and minus 1h are required."
            )
        return values


class ForecastResponse(BaseModel):
    """Identify both the target hour and the exact model artifact used."""

    target_hour: NaiveDatetime
    prediction: float = Field(ge=0, allow_inf_nan=False)
    model_version: str = Field(description="SHA-256 of the loaded model.joblib bytes.")


def forecast_features(request: ForecastRequest) -> pd.DataFrame:
    """Append the unknown target hour and reuse the training feature builder."""
    index = pd.date_range(
        end=request.target_hour, periods=HISTORY_HOURS + 1, freq="h", name="timestamp"
    )
    history = pd.DataFrame({"cnt": request.recent_counts + [None]}, index=index)
    return make_features(history).iloc[[-1]]


def create_app(model_path: Path | None = None, log_path: Path | None = None) -> FastAPI:
    """Load the chosen model once, before the service accepts requests."""
    if model_path is None:
        model_path = Path(
            os.environ.get("BIKE_DEMAND_MODEL_PATH", "artifacts/run_01/model.joblib")
        )
    if log_path is None:
        log_path = Path(
            os.environ.get("BIKE_DEMAND_LOG_PATH", "artifacts/api_requests.jsonl")
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model_bytes = model_path.read_bytes()
        bundle = joblib.load(BytesIO(model_bytes))
        expected_columns = forecast_features(
            ForecastRequest.model_validate(EXAMPLE_REQUEST)
        ).columns.tolist()
        if bundle["feature_columns"] != expected_columns:
            raise ValueError("Model features do not match the current feature builder.")
        app.state.estimator = bundle["estimator"]
        app.state.feature_columns = bundle["feature_columns"]
        app.state.model_version = "sha256:" + hashlib.sha256(model_bytes).hexdigest()
        with jsonl_logger(log_path) as request_logger, threadpool_limits(limits=2):
            app.state.request_logger = request_logger
            yield

    app = FastAPI(title="Bike demand forecasting", lifespan=lifespan)

    @app.middleware("http")
    async def log_prediction_request(request: Request, call_next) -> Response:
        """Record one event for every prediction attempt, including failures."""
        if request.method != "POST" or request.url.path != "/predict":
            return await call_next(request)
        started = perf_counter()
        event = {
            "request_id": uuid4().hex,
            "received_at": datetime.now(UTC).isoformat(),
            "target_hour": None,
            "model_version": app.state.model_version,
            "status_code": 500,
            "latency_ms": None,
            "prediction": None,
            "history_hours": None,
            "missing_count": None,
        }
        request.state.prediction_log = event
        try:
            response = await call_next(request)
            event["status_code"] = response.status_code
            response.headers["X-Request-ID"] = event["request_id"]
            return response
        finally:
            event["latency_ms"] = round((perf_counter() - started) * 1000, 3)
            level = logging.ERROR if event["status_code"] >= 500 else logging.INFO
            app.state.request_logger.log(level, json.dumps(event, allow_nan=False))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        """Retain useful summaries while preserving FastAPI's validation response."""
        event = getattr(request.state, "prediction_log", None)
        if event is not None:
            event.update(request_details(exc.body))
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(Exception)
    async def server_error(request: Request, exc: Exception) -> PlainTextResponse:
        """Include the request ID in the standard server-error response."""
        event = getattr(request.state, "prediction_log", None)
        headers = {"X-Request-ID": event["request_id"]} if event else None
        return PlainTextResponse(
            "Internal Server Error", status_code=500, headers=headers
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report readiness after the model has loaded successfully."""
        return {"status": "ok", "model_version": app.state.model_version}

    @app.post("/predict")
    def predict(forecast: ForecastRequest, request: Request) -> ForecastResponse:
        """Predict the target hour from the previous week's observations."""
        event = request.state.prediction_log
        event.update(
            target_hour=forecast.target_hour.isoformat(),
            history_hours=len(forecast.recent_counts),
            missing_count=sum(value is None for value in forecast.recent_counts),
        )
        features = forecast_features(forecast)[app.state.feature_columns]
        prediction = float(app.state.estimator.predict(features)[0])
        response = ForecastResponse(
            target_hour=forecast.target_hour,
            prediction=prediction,
            model_version=app.state.model_version,
        )
        event["prediction"] = response.prediction
        return response

    return app
