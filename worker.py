import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from celery import Celery
import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
import logging
from services.job_tracker import job_tracker
from services.request_context import set_request_id

try:
    from sentry_sdk.integrations.redis import RedisIntegration
except Exception:
    RedisIntegration = None


# =========================
# SENTRY SETUP
# =========================
def _has_redis_client() -> bool:
    try:
        import redis  # noqa
        return True
    except Exception:
        return False


_sentry_integrations = [CeleryIntegration()]
if RedisIntegration is not None and _has_redis_client():
    _sentry_integrations.append(RedisIntegration())

_sentry_dsn = os.getenv("SENTRY_DSN")
_sentry_client_ready = False
try:
    _sentry_client_ready = bool(getattr(sentry_sdk.Hub.current, "client", None))
except Exception:
    _sentry_client_ready = False
if _sentry_dsn and not _sentry_client_ready:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=1.0,
        integrations=_sentry_integrations,
    )


# =========================
# CELERY INIT
# =========================
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
logger = logging.getLogger("ahvi.worker")

celery_app = Celery(
    "ahvi_tasks",
    broker=redis_url,
    backend=redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


def _retry_or_fail(task, exc: Exception, *, max_retries: int = 2, request_id: str | None = None):
    retries = int(getattr(task.request, "retries", 0))
    if request_id:
        set_request_id(request_id)
    if retries >= max_retries:
        try:
            job_tracker.mark_failed(str(getattr(task.request, "id", "")), error=str(exc))
        except Exception:
            pass
        raise exc
    countdown = 2 ** (retries + 1)
    try:
        job_tracker.mark_retrying(
            str(getattr(task.request, "id", "")),
            error=str(exc),
            attempt=retries + 1,
            max_retries=max_retries,
        )
    except Exception:
        pass
    raise task.retry(exc=exc, countdown=countdown, max_retries=max_retries)


def _mark_started(task, *, request_id: str | None = None, user_id: str | None = None) -> None:
    try:
        if request_id:
            set_request_id(request_id)
        fields = {}
        if request_id:
            fields["request_id"] = str(request_id)
        if user_id:
            fields["user_id"] = str(user_id)
        if fields:
            job_tracker.update(str(getattr(task.request, "id", "")), **fields)
        attempt = int(getattr(task.request, "retries", 0)) + 1
        job_tracker.mark_started(str(getattr(task.request, "id", "")), attempt=attempt)
    except Exception:
        pass


def _mark_succeeded(task, result_meta: dict | None = None, *, request_id: str | None = None) -> None:
    try:
        if request_id:
            set_request_id(request_id)
        job_tracker.mark_succeeded(str(getattr(task.request, "id", "")), result_meta=result_meta or {})
    except Exception:
        pass


# =========================
# AUDIO TASK
# =========================
@celery_app.task(name="generate_audio", bind=True)
def run_heavy_audio_task(self, text_to_clone, lang, request_id: str = ""):
    from services import audio_service

    _mark_started(self, request_id=request_id)
    try:
        audio_base64 = audio_service.generate_cloned_audio(text_to_clone, lang)
        _mark_succeeded(self, {"task": "generate_audio", "request_id": request_id}, request_id=request_id)
        return {"status": "success", "audio_base64": audio_base64}
    except Exception as e:
        logger.exception("AUDIO TASK ERROR")
        _retry_or_fail(self, e, request_id=request_id)


# =========================
# IMAGE TASKS
# =========================
@celery_app.task(name="bg_remove_task", bind=True)
def bg_remove_task(self, image_base64: str, request_id: str = ""):
    from routers.bg_remover import remove_background_sync

    _mark_started(self, request_id=request_id)
    try:
        result = remove_background_sync(image_base64)
        _mark_succeeded(self, {"task": "bg_remove_task", "request_id": request_id}, request_id=request_id)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("BG TASK ERROR")
        _retry_or_fail(self, e, request_id=request_id)


@celery_app.task(name="vision_analyze_task", bind=True)
def vision_analyze_task(self, image_base64: str, user_id: str = "demo_user", request_id: str = ""):
    from routers.vision import vision_analyze_core

    _mark_started(self, request_id=request_id, user_id=user_id)
    try:
        result = vision_analyze_core(image_base64=image_base64, user_id=user_id)
        _mark_succeeded(self, {"task": "vision_analyze_task", "user_id": user_id, "request_id": request_id}, request_id=request_id)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("VISION TASK ERROR")
        _retry_or_fail(self, e, request_id=request_id)


@celery_app.task(name="capture_analyze_task", bind=True)
def capture_analyze_task(self, user_id: str, image_base64: str, request_id: str = ""):
    from routers.wardrobe_capture import analyze_capture_core

    _mark_started(self, request_id=request_id, user_id=user_id)
    try:
        result = analyze_capture_core(user_id=user_id, image_base64=image_base64)
        _mark_succeeded(self, {"task": "capture_analyze_task", "user_id": user_id, "request_id": request_id}, request_id=request_id)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("CAPTURE ANALYZE TASK ERROR")
        _retry_or_fail(self, e, request_id=request_id)


@celery_app.task(name="capture_save_selected_task", bind=True)
def capture_save_selected_task(self, payload: dict, request_id: str = ""):
    from routers.wardrobe_capture import save_selected_core

    request_id = request_id or str(payload.get("request_id") or "")
    _mark_started(self, request_id=request_id)
    try:
        user_id = str(payload.get("user_id", ""))
        selected_item_ids = payload.get("selected_item_ids", []) or []
        detected_items_raw = payload.get("detected_items", []) or []
        result = save_selected_core(
            user_id=user_id,
            selected_item_ids=selected_item_ids,
            detected_items=detected_items_raw,
        )
        _mark_succeeded(self, {"task": "capture_save_selected_task", "user_id": user_id, "request_id": request_id}, request_id=request_id)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("CAPTURE SAVE TASK ERROR")
        _retry_or_fail(self, e, request_id=request_id)


# =========================
# COMBINED ASYNC UPLOAD PIPELINE
# =========================
@celery_app.task(name="process_upload", bind=True)
def process_upload_task(self, user_id: str, image_base64: str, request_id: str = ""):
    """
    1) Analyze capture
    2) Auto-select all detected items
    3) Save selected items
    """
    _mark_started(self, request_id=request_id, user_id=user_id)
    try:
        from routers.wardrobe_capture import analyze_capture_core, save_selected_core

        analysis_result = analyze_capture_core(user_id=user_id, image_base64=image_base64)
        items = analysis_result.get("items", []) or []
        selected_ids = [i.get("item_id") for i in items if i.get("item_id")]

        saved_result = save_selected_core(
            user_id=user_id,
            selected_item_ids=selected_ids,
            detected_items=items,
        )

        result = {
            "status": "success",
            "analysis": analysis_result,
            "save": {"status": "success", "result": saved_result},
        }
        _mark_succeeded(self, {"task": "process_upload", "user_id": user_id, "request_id": request_id}, request_id=request_id)
        return result
    except Exception as e:
        logger.exception("PROCESS UPLOAD TASK ERROR")
        _retry_or_fail(self, e, request_id=request_id)

