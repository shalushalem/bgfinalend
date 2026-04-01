import base64
import uuid
from typing import Any, Dict, List

from fastapi import HTTPException

from services.appwrite_proxy import AppwriteProxy
from services.embedding_service import encode_metadata
from services.image_embedding_service import encode_image_bytes
from services.image_fingerprint import compute_pixel_hash_from_bytes
from services.qdrant_service import qdrant_service
from services.r2_storage import R2Storage, R2StorageError


def _decode_simple_base64(value: str) -> bytes:
    text = (value or "").strip()
    if "," in text:
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid item base64: {exc}")


def persist_selected_items(
    *,
    user_id: str,
    selected_item_ids: List[str],
    detected_items: List[Any],
    duplicate_threshold: float,
    pixel_max_distance: int,
    image_duplicate_threshold: float,
) -> Dict[str, Any]:
    selected = set(selected_item_ids or [])
    if not selected:
        raise HTTPException(status_code=400, detail="selected_item_ids cannot be empty")

    storage = R2Storage()
    proxy = AppwriteProxy()
    saved: List[Dict[str, Any]] = []
    skipped_duplicates: List[Dict[str, Any]] = []

    for item in detected_items:
        item_id = str(item.get("item_id") or "")
        if item_id not in selected:
            continue

        raw_bytes = _decode_simple_base64(str(item.get("raw_crop_base64") or ""))
        seg_bytes = _decode_simple_base64(str(item.get("segmented_png_base64") or ""))
        image_vector = encode_image_bytes(seg_bytes)
        pixel_hash = compute_pixel_hash_from_bytes(seg_bytes)

        if image_vector:
            image_duplicate = qdrant_service.find_image_duplicate(
                image_vector,
                user_id,
                threshold=image_duplicate_threshold,
            )
            if image_duplicate.get("is_duplicate"):
                skipped_duplicates.append(
                    {
                        "item_id": item_id,
                        "name": item.get("name"),
                        "reason": "image_vector",
                        "duplicate_point_id": image_duplicate.get("id"),
                        "duplicate_score": float(image_duplicate.get("score") or 0.0),
                    }
                )
                continue

        if pixel_hash:
            pixel_duplicate = qdrant_service.find_pixel_duplicate(
                user_id,
                pixel_hash,
                max_distance=pixel_max_distance,
            )
            if pixel_duplicate.get("is_duplicate"):
                skipped_duplicates.append(
                    {
                        "item_id": item_id,
                        "name": item.get("name"),
                        "reason": "pixel_hash",
                        "pixel_hash": pixel_hash,
                        "pixel_distance": pixel_duplicate.get("distance"),
                        "duplicate_point_id": pixel_duplicate.get("id"),
                    }
                )
                continue

        vector_input = {
            "category": item.get("category"),
            "sub_category": item.get("sub_category"),
            "color_code": item.get("color_code"),
            "pattern": item.get("pattern"),
            "occasions": item.get("occasions"),
        }
        vector = encode_metadata(vector_input)
        duplicate = qdrant_service.find_duplicate(vector, user_id, threshold=duplicate_threshold)
        if duplicate.get("is_duplicate"):
            skipped_duplicates.append(
                {
                    "item_id": item_id,
                    "name": item.get("name"),
                    "reason": "semantic",
                    "duplicate_point_id": duplicate.get("id"),
                    "duplicate_score": float(duplicate.get("score") or 0.0),
                }
            )
            continue

        file_id = str(uuid.uuid4())
        try:
            upload = storage.upload_wardrobe_images(
                file_id=file_id,
                raw_image_bytes=raw_bytes,
                masked_image_bytes=seg_bytes,
            )
        except R2StorageError as exc:
            raise HTTPException(status_code=500, detail=f"R2 upload failed: {exc}")

        metadata = {
            "userId": user_id,
            "name": item.get("name"),
            "category": item.get("category"),
            "sub_category": item.get("sub_category"),
            "color_code": item.get("color_code"),
            "pattern": item.get("pattern"),
            "occasions": item.get("occasions"),
            "status": "active",
            "image_url": upload["raw_image_url"],
            "masked_url": upload["masked_image_url"],
            "image_id": upload["raw_file_name"],
            "masked_id": upload["masked_file_name"],
            "qdrant_point_id": file_id,
            "worn": 0,
            "liked": False,
        }

        try:
            doc = proxy.create_document("outfits", metadata)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Appwrite save failed: {exc}")

        qdrant_service.upsert_item(
            item_id=file_id,
            vector=vector,
            payload={
                "userId": user_id,
                "category": item.get("category"),
                "sub_category": item.get("sub_category"),
                "color_code": item.get("color_code"),
                "image_url": upload["masked_image_url"],
                "pixel_hash": pixel_hash,
            },
        )
        if image_vector:
            qdrant_service.upsert_image_vector(
                point_id=file_id,
                vector=image_vector,
                payload={
                    "userId": user_id,
                    "category": item.get("category"),
                    "sub_category": item.get("sub_category"),
                    "color_code": item.get("color_code"),
                    "image_url": upload["masked_image_url"],
                    "pixel_hash": pixel_hash,
                },
            )

        saved.append(
            {
                "item_id": item_id,
                "outfit_doc_id": doc.get("$id"),
                "image_url": upload["masked_image_url"],
                "raw_image_url": upload["raw_image_url"],
                "name": item.get("name"),
                "category": item.get("category"),
                "sub_category": item.get("sub_category"),
            }
        )

    return {
        "success": True,
        "message": f"Saved {len(saved)} selected items to wardrobe.",
        "saved_items": saved,
        "skipped_duplicates": skipped_duplicates,
        "meta": {
            "duplicate_threshold": duplicate_threshold,
            "image_duplicate_threshold": image_duplicate_threshold,
            "pixel_max_distance": pixel_max_distance,
            "duplicates_skipped": len(skipped_duplicates),
        },
    }
