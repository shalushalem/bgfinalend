import base64
import os
import cv2
import numpy as np

from collections import Counter
from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

# Import the new prompt from the prompts folder
from prompts.core_prompts import VISION_ANALYZE_PROMPT

from services.embedding_service import encode_metadata
from services.image_embedding_service import encode_image_base64
from services.image_fingerprint import compute_pixel_hash_from_base64
from services import ai_gateway
from services.qdrant_service import qdrant_service
try:
    from worker import vision_analyze_task
except Exception:
    vision_analyze_task = None
try:
    from services.job_tracker import job_tracker
except Exception:
    job_tracker = None
from services.task_queue import enqueue_task
try:
    from routers.bg_remover import BGRemoveRequest, remove_background_sync
except Exception:
    BGRemoveRequest = None
    remove_background_sync = None

router = APIRouter()


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _vision_enable_similarity() -> bool:
    return _env_bool("VISION_ANALYZE_ENABLE_SIMILARITY", False)


def _duplicate_threshold() -> float:
    raw = str(os.getenv("WARDROBE_DUPLICATE_THRESHOLD", "0.97") or "").strip()
    try:
        value = float(raw)
        if value <= 0.0:
            return 0.97
        if value > 1.0:
            return 1.0
        return value
    except Exception:
        return 0.97


def _pixel_duplicate_distance() -> int:
    raw = str(os.getenv("WARDROBE_PIXEL_DUPLICATE_DISTANCE", "6") or "").strip()
    try:
        value = int(raw)
        if value < 0:
            return 0
        if value > 64:
            return 64
        return value
    except Exception:
        return 6


def _image_duplicate_threshold() -> float:
    raw = str(os.getenv("WARDROBE_IMAGE_DUPLICATE_THRESHOLD", "0.985") or "").strip()
    try:
        value = float(raw)
        if value <= 0.0:
            return 0.985
        if value > 1.0:
            return 1.0
        return value
    except Exception:
        return 0.985


class ImageAnalyzeRequest(BaseModel):
    image_base64: str = Field(..., min_length=20)
    userId: str = "demo_user"


def _normalize_base64_for_model(value: str) -> str:
    text = (value or "").strip()
    if "," in text:
        return text.split(",", 1)[1]
    return text


def _to_png_data_uri(base64_text: str) -> str:
    text = _normalize_base64_for_model(base64_text)
    return f"data:image/png;base64,{text}"


def _input_has_alpha(image_base64: str) -> bool:
    try:
        b64 = _normalize_base64_for_model(image_base64)
        img_data = base64.b64decode(b64, validate=True)
        np_arr = np.frombuffer(img_data, np.uint8)
        decoded = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        return bool(decoded is not None and decoded.ndim == 3 and decoded.shape[2] == 4)
    except Exception:
        return False


def _remove_bg_first(image_base64: str):
    if _input_has_alpha(image_base64):
        return image_base64, True, "input_already_has_alpha"
    if BGRemoveRequest is None or remove_background_sync is None:
        return image_base64, False, "bg_remover_unavailable"
    try:
        req = BGRemoveRequest(image_base64=image_base64)
        result = remove_background_sync(req.image_base64)
        if isinstance(result, dict) and result.get("success") and result.get("image_base64"):
            processed = _to_png_data_uri(result.get("image_base64"))
            bg_removed = bool(result.get("bg_removed", True))
            return processed, bg_removed, result.get("fallback_reason")
        return image_base64, False, "bg_remove_no_image"
    except Exception as exc:
        print(f"BG remove before vision failed: {exc}")
        return image_base64, False, f"bg_remove_failed: {exc}"


def _hex_to_color_name(hex_color: str) -> str:
    try:
        color = hex_color.lstrip("#")
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except Exception:
        return "Multicolor"

    if max(r, g, b) < 40: return "Black"
    if min(r, g, b) > 220: return "White"
    if abs(r - g) < 14 and abs(g - b) < 14: return "Gray"
    if r > 180 and g < 110 and b < 110: return "Red"
    if r > 170 and g > 120 and b < 90: return "Orange"
    if r > 170 and g > 170 and b < 90: return "Yellow"
    if g > 150 and r < 130 and b < 130: return "Green"
    if b > 150 and r < 130 and g < 150: return "Blue"
    if r > 150 and b > 150 and g < 130: return "Purple"
    if r > 140 and g > 100 and b > 70: return "Brown"
    return "Multicolor"


def _infer_pattern(cv_image) -> str:
    try:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        return "checked" if edge_density > 0.09 else "plain"
    except Exception:
        return "plain"


def _extract_foreground_mask(decoded_img) -> np.ndarray | None:
    try:
        if decoded_img is None:
            return None

        if decoded_img.ndim == 3 and decoded_img.shape[2] == 4:
            alpha = decoded_img[:, :, 3]
            mask = alpha > 18
        else:
            bgr = decoded_img if decoded_img.ndim == 3 else None
            if bgr is None: return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            maxc = rgb.max(axis=2)
            minc = rgb.min(axis=2)
            near_white = (
                (rgb[:, :, 0] >= 236) & (rgb[:, :, 1] >= 236) & (rgb[:, :, 2] >= 236) & ((maxc - minc) <= 20)
            )
            saturated = hsv[:, :, 1] > 18
            mask = (~near_white) | saturated

        mask_u8 = mask.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        final_mask = mask_u8 > 0

        if final_mask.sum() < 200:
            return None
        return final_mask
    except Exception:
        return None


def _infer_garment_hint(decoded_img) -> tuple[str, str]:
    mask = _extract_foreground_mask(decoded_img)
    if mask is None:
        return ("Tops", "Shirt")

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return ("Tops", "Shirt")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    box_w = max(1, x_max - x_min + 1)
    box_h = max(1, y_max - y_min + 1)
    aspect = float(box_h) / float(box_w)

    crop = mask[y_min : y_max + 1, x_min : x_max + 1]
    row_fill = crop.mean(axis=1)
    seg = max(2, int(0.2 * len(row_fill)))
    top_fill = float(row_fill[:seg].mean())
    mid_fill = float(row_fill[len(row_fill) // 2 - seg // 2: len(row_fill) // 2 + seg // 2].mean())
    bottom_fill = float(row_fill[-seg:].mean())

    h = crop.shape[0]
    upper_y = min(h - 1, max(0, int(h * 0.15)))
    mid_y = min(h - 1, max(0, int(h * 0.50)))
    low_y = min(h - 1, max(0, int(h * 0.85)))
    upper_width = int(np.count_nonzero(crop[upper_y]))
    mid_width = int(np.count_nonzero(crop[mid_y]))
    low_width = int(np.count_nonzero(crop[low_y]))

    lower = crop[int(h * 0.55):, :]
    cc_count = 0
    if lower.size > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((lower.astype(np.uint8)) * 255)
        large_components = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= max(120, int(lower.size * 0.01)):
                large_components.append(area)
        cc_count = len(large_components)

    likely_bottom = (
        aspect > 1.15
        and (
            cc_count >= 2
            or bottom_fill < (mid_fill * 0.86)
            or (low_width > 0 and mid_width > 0 and (low_width / float(mid_width)) < 0.78)
            or (upper_width > 0 and low_width > 0 and (low_width / float(upper_width)) < 0.74)
        )
    )
    if likely_bottom:
        return ("Bottoms", "Trousers")

    return ("Tops", "Shirt")


def _build_smart_fallback(color_hex: str, hint_category: str = "Tops", hint_sub_category: str = "Shirt"):
    color_name = _hex_to_color_name(color_hex)
    return {
        "name": f"{color_name} {hint_sub_category}",
        "category": hint_category,
        "sub_category": hint_sub_category,
        "occasions": ["casual", "office"],
        "pattern": "plain",
    }


_CATEGORY_ALIASES = {
    "top": "Tops",
    "tops": "Tops",
    "shirt": "Tops",
    "t-shirt": "Tops",
    "tee": "Tops",
    "blouse": "Tops",
    "bottom": "Bottoms",
    "bottoms": "Bottoms",
    "trouser": "Bottoms",
    "trousers": "Bottoms",
    "pants": "Bottoms",
    "jeans": "Bottoms",
    "skirt": "Bottoms",
    "shorts": "Bottoms",
    "shoe": "Footwear",
    "shoes": "Footwear",
    "footwear": "Footwear",
    "sneaker": "Footwear",
    "sneakers": "Footwear",
    "heel": "Footwear",
    "heels": "Footwear",
    "boot": "Footwear",
    "boots": "Footwear",
    "sandal": "Footwear",
    "sandals": "Footwear",
    "outerwear": "Outerwear",
    "jacket": "Outerwear",
    "coat": "Outerwear",
    "blazer": "Outerwear",
    "hoodie": "Outerwear",
    "dress": "Dresses",
    "dresses": "Dresses",
    "gown": "Dresses",
    "jumpsuit": "Dresses",
    "accessory": "Accessories",
    "accessories": "Accessories",
    "belt": "Accessories",
    "scarf": "Accessories",
    "hat": "Accessories",
    "cap": "Accessories",
    "sunglasses": "Accessories",
    "bag": "Bags",
    "bags": "Bags",
    "handbag": "Bags",
    "backpack": "Bags",
    "tote": "Bags",
    "jewelry": "Jewelry",
    "jewellery": "Jewelry",
    "watch": "Jewelry",
    "necklace": "Jewelry",
    "earring": "Jewelry",
    "bracelet": "Jewelry",
    "ring": "Jewelry",
    "indian wear": "Indian Wear",
    "ethnic": "Indian Wear",
    "saree": "Indian Wear",
    "kurta": "Indian Wear",
    "lehenga": "Indian Wear",
    "salwar": "Indian Wear",
}


_DEFAULT_SUB_BY_CATEGORY = {
    "Tops": "Shirt",
    "Bottoms": "Trousers",
    "Footwear": "Shoes",
    "Outerwear": "Jacket",
    "Accessories": "Accessory",
    "Dresses": "Dress",
    "Bags": "Bag",
    "Jewelry": "Jewelry",
    "Indian Wear": "Kurta",
}


def _clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _canonicalize_category(category: str, sub_category: str, name: str, hint_category: str) -> str:
    candidates = [category, sub_category, name, hint_category]

    # Pass 1: Try exact matches on ALL fields first (Safest)
    for candidate in candidates:
        key = _clean_text(candidate).lower()
        # Strip common broad taxonomies that confuse the categorizer
        key = key.replace(" & accessories", "").replace(" and accessories", "")
        
        if key in _CATEGORY_ALIASES:
            return _CATEGORY_ALIASES[key]

    # Pass 2: Token-by-token matching ONLY if no exact match is found
    for candidate in candidates:
        key = _clean_text(candidate).lower()
        tokens = key.replace("/", " ").replace("-", " ").split()

        for token in tokens:
            # Prevent "cap sleeve" from being marked as a hat (accessory)
            if token == "cap" and "sleeve" in tokens:
                continue
                
            if token in _CATEGORY_ALIASES:
                return _CATEGORY_ALIASES[token]

    # Fallback to the hint category (Tops/Bottoms)
    return _CATEGORY_ALIASES.get(_clean_text(hint_category).lower(), "Tops")

def _normalize_occasions(value) -> list[str]:
    if isinstance(value, list):
        out = []
        for item in value:
            text = _clean_text(item)
            if text:
                out.append(text.lower())
        return out
    if isinstance(value, str):
        items = [part.strip().lower() for part in value.split(",")]
        return [item for item in items if item]
    return []


def _shape_vision_output(raw_data, color_hex: str, hint_category: str, hint_sub_category: str, cv_image) -> dict:
    data = dict(raw_data) if isinstance(raw_data, dict) else {}

    name = _clean_text(data.get("name") or data.get("title") or data.get("item_name"))
    category = _clean_text(data.get("category") or data.get("main_category") or data.get("type"))
    sub_category = _clean_text(data.get("sub_category") or data.get("subcategory") or data.get("subType"))
    pattern = _clean_text(data.get("pattern") or data.get("texture"))
    occasions = _normalize_occasions(data.get("occasions") or data.get("occasion"))

    canonical_category = _canonicalize_category(category, sub_category, name, hint_category)
    if not sub_category:
        sub_category = _clean_text(hint_sub_category) or _DEFAULT_SUB_BY_CATEGORY.get(canonical_category, "Item")
    if not name:
        color_name = _hex_to_color_name(color_hex)
        name = f"{color_name} {sub_category}".strip()
    if not pattern:
        pattern = _infer_pattern(cv_image)
    if not occasions:
        occasions = ["casual", "office"]

    data["name"] = name
    data["category"] = canonical_category
    data["sub_category"] = sub_category
    data["pattern"] = pattern
    data["occasions"] = occasions
    return data


def get_dominant_color(cv_image, k=3):
    try:
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        crop_h, crop_w = int(h * 0.25), int(w * 0.25)
        center_image = image[crop_h:h - crop_h, crop_w:w - crop_w]

        center_image = cv2.resize(center_image, (100, 100), interpolation=cv2.INTER_AREA)

        hsv_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2HSV)
        pixels_rgb = center_image.reshape((-1, 3))
        pixels_hsv = hsv_image.reshape((-1, 3))

        mask = (pixels_hsv[:, 1] > 20) & (pixels_hsv[:, 2] > 70) & (pixels_hsv[:, 2] < 245)
        filtered_rgb = pixels_rgb[mask]

        if len(filtered_rgb) < 100:
            mask = (pixels_hsv[:, 2] > 30) & (pixels_hsv[:, 2] < 250)
            filtered_rgb = pixels_rgb[mask]
            if len(filtered_rgb) == 0:
                filtered_rgb = pixels_rgb

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(filtered_rgb)

        counts = Counter(kmeans.labels_)
        dominant_cluster_index = counts.most_common(1)[0][0]
        dominant_rgb = [int(x) for x in kmeans.cluster_centers_[dominant_cluster_index]]

        return "#{:02x}{:02x}{:02x}".format(*dominant_rgb).upper()

    except Exception as e:
        print(f"Color Math Error: {e}")
        return "#000000"


@router.post("/analyze-image")
def analyze_image(request: ImageAnalyzeRequest):
    return vision_analyze_core(request.image_base64, request.userId)


def vision_analyze_core(image_base64: str, user_id: str = "demo_user"):
    # STEP 0: BG REMOVAL FIRST
    vision_input_base64, bg_removed, bg_fallback_reason = _remove_bg_first(image_base64)

    # STEP 1: Decode + Color
    base64_data = _normalize_base64_for_model(vision_input_base64)
    try:
        img_data = base64.b64decode(base64_data, validate=True)
        if len(img_data) > 12 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="image payload too large (max 12MB)")

        np_arr = np.frombuffer(img_data, np.uint8)
        decoded = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise HTTPException(status_code=400, detail="invalid image payload")

        if decoded.ndim == 3 and decoded.shape[2] == 4:
            cv_image = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
        elif decoded.ndim == 3 and decoded.shape[2] == 3:
            cv_image = decoded
        else:
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cv_image is None:
            raise HTTPException(status_code=400, detail="invalid image payload")

        extracted_color_hex = get_dominant_color(cv_image)
        hint_category, hint_sub_category = _infer_garment_hint(decoded)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {str(e)}")

    # STEP 2: RAW LLM ANALYSIS (Using imported prompt)
    llm_fallback = False
    model_used = None
    try:
        final_data, model_used = ai_gateway.ollama_vision_json(
            prompt=VISION_ANALYZE_PROMPT,
            image_base64=_normalize_base64_for_model(vision_input_base64),
        )
    except Exception as e:
        print(f"Image Analyze Error: {str(e)}")
        llm_fallback = True
        # Keep smart fallback ONLY if LLM completely crashes
        final_data = _build_smart_fallback(
            extracted_color_hex,
            hint_category=hint_category,
            hint_sub_category=hint_sub_category,
        )

    final_data = _shape_vision_output(
        final_data,
        color_hex=extracted_color_hex,
        hint_category=hint_category,
        hint_sub_category=hint_sub_category,
        cv_image=cv_image,
    )

    # STEP 3: PREPARE DATA
    # Use extracted hex ONLY if the LLM failed to provide a color code
    final_data["color_code"] = _clean_text(final_data.get("color_code")) or extracted_color_hex
    final_data["userId"] = user_id

    # STEP 4: EMBEDDING + SIMILARITY CHECK
    image_duplicate = {"checked": False, "is_duplicate": False, "id": None, "score": 0.0}
    pixel_duplicate = {"checked": False, "is_duplicate": False, "id": None, "distance": None}
    vector = None
    similar_items = []
    image_vector = []
    pixel_hash = ""
    image_duplicate_threshold = _image_duplicate_threshold()
    pixel_max_distance = _pixel_duplicate_distance()

    if _vision_enable_similarity():
        try:
            vector = encode_metadata(final_data)
            similar_items = qdrant_service.search_similar(vector, user_id, limit=5)
        except Exception as e:
            print("Qdrant similarity search error:", str(e))

        image_vector = encode_image_base64(vision_input_base64)
        if image_vector:
            try:
                image_duplicate = qdrant_service.find_image_duplicate(
                    image_vector, user_id, threshold=image_duplicate_threshold,
                )
            except Exception as e:
                print("Qdrant image duplicate check error:", str(e))

        pixel_hash = compute_pixel_hash_from_base64(vision_input_base64)
        if pixel_hash:
            try:
                pixel_duplicate = qdrant_service.find_pixel_duplicate(
                    user_id, pixel_hash, max_distance=pixel_max_distance,
                )
            except Exception as e:
                print("Qdrant pixel duplicate check error:", str(e))

    duplicate_threshold = _duplicate_threshold()
    top_similarity_score = float(similar_items[0].get("score") or 0.0) if similar_items else 0.0

    probable_duplicate = bool(
        image_duplicate.get("is_duplicate")
        or pixel_duplicate.get("is_duplicate")
        or top_similarity_score >= duplicate_threshold
    )

    return {
        "success": True,
        "data": final_data,
        "processed_image_base64": vision_input_base64,
        "similar_items": similar_items,
        "meta": {
            "bg_removed_first": True,
            "bg_removed": bg_removed,
            "bg_fallback_reason": bg_fallback_reason,
            "llm_fallback": llm_fallback,
            "vision_model_used": model_used,
            "similarity_enabled": _vision_enable_similarity(),
            "embedding_created": vector is not None,
            "similar_items_found": len(similar_items),
            "duplicate_threshold": duplicate_threshold,
            "top_similarity_score": top_similarity_score,
            "image_vector_ready": bool(image_vector),
            "image_duplicate_checked": bool(image_duplicate.get("checked")),
            "image_duplicate_threshold": image_duplicate_threshold,
            "image_duplicate_score": float(image_duplicate.get("score") or 0.0),
            "image_duplicate_point_id": image_duplicate.get("id"),
            "image_probable_duplicate": bool(image_duplicate.get("is_duplicate")),
            "pixel_hash": pixel_hash or None,
            "pixel_duplicate_checked": bool(pixel_duplicate.get("checked")),
            "pixel_duplicate_distance": pixel_duplicate.get("distance"),
            "pixel_duplicate_max_distance": pixel_max_distance,
            "pixel_duplicate_point_id": pixel_duplicate.get("id"),
            "pixel_probable_duplicate": bool(pixel_duplicate.get("is_duplicate")),
            "probable_duplicate": probable_duplicate,
        },
    }


@router.post("/analyze-image/async", status_code=status.HTTP_202_ACCEPTED)
def analyze_image_async(http_request: Request, request: ImageAnalyzeRequest):
    if vision_analyze_task is None:
        raise HTTPException(status_code=503, detail="Celery worker not configured")
    try:
        request_id = str(getattr(http_request.state, "request_id", "") or "")
        task_id = enqueue_task(
            task_func=vision_analyze_task,
            args=[request.image_base64, request.userId],
            kwargs={"request_id": request_id},
            kind="vision_analyze",
            user_id=request.userId,
            request_id=request_id,
            source="api:/api/vision/analyze-image/async",
            meta={"task_type": "vision_analyze_task"},
        )
        return {
            "success": True,
            "status": "queued",
            "task_id": task_id,
            "task_type": "vision_analyze_task",
            "request_id": request_id,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue vision analysis: {exc}")
