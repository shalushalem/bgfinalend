import json
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._redis = None
        self._redis_prefix = "ahvi:job:"
        self._redis_recent_global = "ahvi:job:recent:global"
        self._redis_recent_user_prefix = "ahvi:job:recent:user:"
        self._max_memory_items = 2000

    def _redis_client(self):
        if self._redis is not None:
            return self._redis
        if redis is None:
            return None
        try:
            url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self._redis = redis.Redis.from_url(url, decode_responses=True)
            self._redis.ping()
            return self._redis
        except Exception:
            self._redis = None
            return None

    def _redis_key(self, job_id: str) -> str:
        return f"{self._redis_prefix}{job_id}"

    def _write(self, payload: Dict[str, Any]) -> None:
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            return
        payload["updated_at"] = _now_iso()

        client = self._redis_client()
        if client is not None:
            try:
                ts = datetime.now(timezone.utc).timestamp()
                client.setex(self._redis_key(job_id), 7 * 24 * 3600, json.dumps(payload, ensure_ascii=True))
                client.zadd(self._redis_recent_global, {job_id: ts})
                client.expire(self._redis_recent_global, 7 * 24 * 3600)
                uid = str(payload.get("user_id") or "").strip()
                if uid:
                    user_key = f"{self._redis_recent_user_prefix}{uid}"
                    client.zadd(user_key, {job_id: ts})
                    client.expire(user_key, 7 * 24 * 3600)
                return
            except Exception:
                pass

        with self._lock:
            self._memory[job_id] = dict(payload)
            if len(self._memory) > self._max_memory_items:
                oldest = sorted(
                    self._memory.items(),
                    key=lambda kv: str(kv[1].get("updated_at") or ""),
                )[: max(0, len(self._memory) - self._max_memory_items)]
                for key, _ in oldest:
                    self._memory.pop(key, None)

    def get(self, job_id: str) -> Dict[str, Any] | None:
        jid = str(job_id or "").strip()
        if not jid:
            return None

        client = self._redis_client()
        if client is not None:
            try:
                raw = client.get(self._redis_key(jid))
                if raw:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass

        with self._lock:
            data = self._memory.get(jid)
            return dict(data) if isinstance(data, dict) else None

    def create(
        self,
        *,
        job_id: str,
        kind: str,
        user_id: str | None = None,
        request_id: str | None = None,
        source: str | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload = {
            "job_id": str(job_id),
            "kind": str(kind or "job"),
            "user_id": str(user_id or ""),
            "request_id": str(request_id or ""),
            "source": str(source or ""),
            "status": "queued",
            "state": "PENDING",
            "attempt": 0,
            "max_retries": 0,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "meta": meta or {},
        }
        self._write(payload)
        return payload

    def update(self, job_id: str, **fields: Any) -> Dict[str, Any] | None:
        current = self.get(job_id)
        if current is None:
            return None
        current.update(fields)
        self._write(current)
        return current

    def mark_started(self, job_id: str, *, attempt: int = 1) -> None:
        current = self.get(job_id) or {"job_id": job_id, "created_at": _now_iso(), "meta": {}}
        current.update({"status": "processing", "state": "STARTED", "attempt": int(attempt)})
        self._write(current)

    def mark_retrying(self, job_id: str, *, error: str, attempt: int, max_retries: int) -> None:
        current = self.get(job_id) or {"job_id": job_id, "created_at": _now_iso(), "meta": {}}
        current.update(
            {
                "status": "retrying",
                "state": "RETRY",
                "error": str(error),
                "attempt": int(attempt),
                "max_retries": int(max_retries),
            }
        )
        self._write(current)

    def mark_succeeded(self, job_id: str, *, result_meta: Dict[str, Any] | None = None) -> None:
        current = self.get(job_id) or {"job_id": job_id, "created_at": _now_iso(), "meta": {}}
        current.update(
            {
                "status": "completed",
                "state": "SUCCESS",
                "error": None,
                "result_meta": result_meta or {},
                "completed_at": _now_iso(),
            }
        )
        self._write(current)

    def mark_failed(self, job_id: str, *, error: str) -> None:
        current = self.get(job_id) or {"job_id": job_id, "created_at": _now_iso(), "meta": {}}
        current.update(
            {
                "status": "failed",
                "state": "FAILURE",
                "error": str(error),
                "completed_at": _now_iso(),
            }
        )
        self._write(current)

    def list_recent(
        self,
        *,
        limit: int = 50,
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))
        uid = str(user_id or "").strip()
        rid = str(request_id or "").strip()

        client = self._redis_client()
        if client is not None:
            try:
                key = f"{self._redis_recent_user_prefix}{uid}" if uid else self._redis_recent_global
                job_ids = client.zrevrange(key, 0, safe_limit - 1)
                rows: List[Dict[str, Any]] = []
                for jid in job_ids:
                    row = self.get(str(jid))
                    if isinstance(row, dict):
                        rows.append(row)
                if rid:
                    rows = [row for row in rows if str(row.get("request_id") or "") == rid]
                return rows
            except Exception:
                pass

        with self._lock:
            items = list(self._memory.values())
        items.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
        if uid:
            items = [row for row in items if str(row.get("user_id") or "") == uid]
        if rid:
            items = [row for row in items if str(row.get("request_id") or "") == rid]
        return [dict(row) for row in items[:safe_limit]]


job_tracker = JobTracker()
