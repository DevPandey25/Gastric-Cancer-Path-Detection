from __future__ import annotations

import io
from pathlib import Path
import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.core.config as cfg_mod
import app.db.database as db_mod
from app.db.database import Base


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch):
    """Provides an isolated SQLite database file per test and patches the engine."""
    db_file = tmp_path / "test_app.db"
    test_uploads = tmp_path / "test_uploads"
    test_uploads.mkdir(parents=True, exist_ok=True)

    test_engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    # Patch production engine and upload dir so test runs never touch production app.db or uploads
    monkeypatch.setattr(db_mod, "engine", test_engine)
    monkeypatch.setattr(db_mod, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(cfg_mod.settings, "database_url", f"sqlite:///{db_file}")
    monkeypatch.setattr(cfg_mod.settings, "upload_dir", test_uploads)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        test_engine.dispose()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generates synthetic RGB tissue-like image bytes for testing."""
    img = Image.new("RGB", (224, 224), color=(180, 120, 140))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_image_path(tmp_path: Path, sample_image_bytes: bytes) -> Path:
    """Saves a temporary valid test image file."""
    path = tmp_path / "test_patch.png"
    path.write_bytes(sample_image_bytes)
    return path
