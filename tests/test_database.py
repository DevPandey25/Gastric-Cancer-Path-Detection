import json
from app.db.models import PredictionRecord
from app.services import (
    create_prediction_record,
    get_prediction_stats,
    get_recent_records,
)


def test_database_crud_operations(temp_db):
    result_payload = {
        "predicted_class": "TUM",
        "predicted_label": "Tumor",
        "cancer_percentage": 85.5,
        "risk_band": "High",
        "confidence_score": 92.4,
        "top_probability": 92.4,
        "model_source": "trained-checkpoint",
        "class_probabilities": [
            {"class_code": "TUM", "label": "Tumor", "color": "#a61e4d", "probability": 92.4}
        ],
    }

    record = create_prediction_record(temp_db, "sample_uuid.png", result_payload)
    assert record["id"] is not None
    assert record["predicted_class"] == "TUM"
    assert record["cancer_percentage"] == 85.5

    # Retrieve from DB
    records = get_recent_records(temp_db)
    assert len(records) == 1
    assert records[0]["filename"] == "sample_uuid.png"

    # Check stats
    stats = get_prediction_stats(temp_db)
    assert stats["total_predictions"] == 1
    assert stats["average_percentage"] == 85.5
    assert stats["high_risk_cases"] == 1
    assert stats["most_common_class"] == "TUM"


def test_database_multiple_records_stats(temp_db):
    # Add a low risk case
    rec1 = PredictionRecord(
        filename="rec1.png",
        predicted_class="NOR",
        predicted_label="Normal Mucosa",
        cancer_percentage=10.0,
        risk_band="Low",
        confidence_score=80.0,
        top_probability=80.0,
        model_source="trained-checkpoint",
        class_probabilities_json=json.dumps([]),
    )
    # Add a high risk case
    rec2 = PredictionRecord(
        filename="rec2.png",
        predicted_class="TUM",
        predicted_label="Tumor",
        cancer_percentage=90.0,
        risk_band="High",
        confidence_score=95.0,
        top_probability=95.0,
        model_source="trained-checkpoint",
        class_probabilities_json=json.dumps([]),
    )
    temp_db.add_all([rec1, rec2])
    temp_db.commit()

    stats = get_prediction_stats(temp_db)
    assert stats["total_predictions"] == 2
    assert stats["average_percentage"] == 50.0  # (10 + 90) / 2
    assert stats["high_risk_cases"] == 1

