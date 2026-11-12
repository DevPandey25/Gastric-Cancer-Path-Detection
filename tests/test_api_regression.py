import io
import pytest
from fastapi import HTTPException, UploadFile
from starlette.requests import Request

from app.main import (
    dashboard,
    dashboard_payload,
    health,
    predict,
    predictions,
)


def test_dashboard_page():
    req = Request({"type": "http"})
    response = dashboard(request=req)
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Gastric Cancer" in response.body.decode("utf-8")


def test_api_health():
    data = health()
    assert data["status"] == "ok"
    assert "Gastric" in data["app"]


def test_api_dashboard_payload(temp_db):
    data = dashboard_payload(db=temp_db)
    assert "model_status" in data
    assert "dataset" in data
    assert "clinical" in data
    assert "biomarkers" in data
    assert "benchmark" in data
    assert "prediction_summary" in data
    assert "recent_predictions" in data


def test_api_predictions_list(temp_db):
    data = predictions(db=temp_db)
    assert "summary" in data
    assert "records" in data
    assert isinstance(data["records"], list)


def test_api_predict_workflow(temp_db, sample_image_bytes):
    # Create valid UploadFile
    upload = UploadFile(file=io.BytesIO(sample_image_bytes), filename="test_patch.png")
    payload = predict(file=upload, db=temp_db)

    assert "predicted_class" in payload
    assert "predicted_label" in payload
    assert "confidence_score" in payload
    assert "cancer_percentage" in payload
    assert "class_probabilities" in payload
    assert len(payload["class_probabilities"]) == 8

    # Verify that the new prediction is recorded in DB
    list_payload = predictions(db=temp_db)
    records = list_payload["records"]
    assert len(records) >= 1
    assert records[0]["predicted_class"] == payload["predicted_class"]


def test_api_predict_missing_filename(temp_db):
    upload = UploadFile(file=io.BytesIO(b""), filename="")
    with pytest.raises(HTTPException) as exc_info:
        predict(file=upload, db=temp_db)
    assert exc_info.value.status_code == 400
