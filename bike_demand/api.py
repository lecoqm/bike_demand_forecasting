"""Serve one-hour demand forecasts from an existing training artifact."""

import hashlib
import os
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, NaiveDatetime, field_validator
from threadpoolctl import threadpool_limits

from bike_demand.features import make_features

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


def create_app(model_path: Path | None = None) -> FastAPI:
    """Load the chosen model once, before the service accepts requests."""
    if model_path is None:
        model_path = Path(
            os.environ.get("BIKE_DEMAND_MODEL_PATH", "artifacts/run_01/model.joblib")
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
        with threadpool_limits(limits=2):
            yield

    app = FastAPI(title="Bike demand forecasting", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report readiness after the model has loaded successfully."""
        return {"status": "ok", "model_version": app.state.model_version}

    @app.post("/predict")
    def predict(request: ForecastRequest) -> ForecastResponse:
        """Predict the target hour from the previous week's observations."""
        features = forecast_features(request)[app.state.feature_columns]
        prediction = float(app.state.estimator.predict(features)[0])
        return ForecastResponse(
            target_hour=request.target_hour,
            prediction=prediction,
            model_version=app.state.model_version,
        )

    return app
