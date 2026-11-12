from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.analytics import demo_summary
from app.core.config import settings
from app.db.database import Base, engine, get_db
from app.research_workbench import get_visual_artifact
from app.research_workbench import router as research_router
from app.clinical_reporting_api import router as reporting_router
from app.services import create_prediction_record, get_prediction_stats, get_recent_records, run_prediction, save_upload

ALLOWED_UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB maximum upload size


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
base_dir = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

# Mount dataset images directory if available
if settings.histology_dataset_dir.exists():
    app.mount("/dataset-images", StaticFiles(directory=str(settings.histology_dataset_dir)), name="dataset_images")

# Register Research Workbench API Router
app.include_router(research_router, prefix="/api/research", tags=["research"])
app.include_router(reporting_router)


# Fallback route for legacy /research-artifacts/ paths with strict validation
@app.get("/research-artifacts/{category}/{filename}")
def legacy_research_artifacts(category: str, filename: str):
    """Backward-compatible endpoint routing to allowlisted visual artifacts."""
    return get_visual_artifact(category, filename)


def apply_security_headers(response: Response) -> Response:
    """Applies strict security, privacy, and cache-control headers to an HTTP response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data: https://fastapi.tiangolo.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "font-src 'self' data: https://fonts.gstatic.com;"
    )
    return response


# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    return apply_security_headers(response)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> Response:
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request, "app_name": settings.app_name}
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "app": settings.app_name}


@app.get("/api/dashboard")
def dashboard_payload(db: Session = Depends(get_db)) -> dict[str, Any]:
    payload = demo_summary()
    payload["prediction_summary"] = get_prediction_stats(db)
    payload["recent_predictions"] = get_recent_records(db)
    return payload


@app.get("/api/predictions")
def predictions(db: Session = Depends(get_db)) -> dict[str, Any]:
    return {
        "summary": get_prediction_stats(db),
        "records": get_recent_records(db),
    }


@app.post("/api/predict")
def predict(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A histopathology tissue image file is required.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
        )

    # Validate file size without buffering entire stream unconditionally
    try:
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read upload stream: {exc}")

    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum permitted size of 10 MB ({file_size / (1024*1024):.2f} MB provided).",
        )
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty (0 bytes).")

    # Deep content and decompression-bomb validation using PIL
    try:
        from PIL import Image, UnidentifiedImageError
        orig_max_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = 16_000_000  # Cap at 16 megapixels for safety

        try:
            with Image.open(file.file) as img:
                detected_format = img.format
                if detected_format not in {"PNG", "JPEG", "TIFF", "BMP"}:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image format '{detected_format}' is not supported. Permitted formats: PNG, JPEG, TIFF, BMP.",
                    )
                width, height = img.size
                if width > 4096 or height > 4096:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image dimensions ({width}x{height}) exceed maximum permitted dimensions of 4096x4096 pixels.",
                    )
                img.verify()
        finally:
            Image.MAX_IMAGE_PIXELS = orig_max_pixels
            file.file.seek(0)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be decoded as a valid image or contains corrupted data.",
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Failed to validate uploaded image content.",
        )

    stored_path = save_upload(file)

    try:
        result = run_prediction(stored_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    payload = {"filename": stored_path.name, **result}
    record = create_prediction_record(db, stored_path.name, payload)
    return record
