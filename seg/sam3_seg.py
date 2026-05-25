"""
SAM3 text-prompt segmentation backend (subprocess).

调用方式与在 ``sam3`` 仓库根目录下手动执行一致，例如::

    /home/ubuntu/miniconda3/envs/sam3/bin/python \\
        /home/ubuntu/stephen/01-code/sam3/scripts/infer.py \\
        --image /path/to/rgb.png \\
        --prompt "Plastic Reel" \\
        --output-dir /path/to/output \\
        --threshold 0.41 \\
        --mask-threshold 0.50

``infer.py`` 会写入 ``{output_dir}/sam6d_results/detection_ism.json``（FoundationPose HTTP 将
``output_dir`` 设为每次请求的 ``service_outputs/<id>/``）。

环境变量（``GENPOSE2_SAM3_*`` 优先，其次 ``SAM6D_SAM3_*``）::

    GENPOSE2_SAM3_ROOT / SAM6D_SAM3_ROOT
    GENPOSE2_SAM3_PYTHON / SAM6D_SAM3_PYTHON
    GENPOSE2_SAM3_INFER_SCRIPT / SAM6D_SAM3_INFER_SCRIPT
    GENPOSE2_SAM3_PROMPT / SAM6D_SAM3_PROMPT
    GENPOSE2_SAM3_THRESHOLD / SAM6D_SAM3_THRESHOLD
    GENPOSE2_SAM3_MASK_THRESHOLD / SAM6D_SAM3_MASK_THRESHOLD
    GENPOSE2_SAM3_CHECKPOINT / SAM6D_SAM3_CHECKPOINT  （可选，传给 infer.py --checkpoint）
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_COCOMASK = None


def _save_instance_id_mask(composite: np.ndarray, mask_path: Path) -> Path:
    """保存实例 id 图（像素值 1..N 表示各实例）。"""
    mask_path = Path(mask_path).expanduser().resolve()
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_path.suffix.lower() == ".exr":
        mask_path = mask_path.with_suffix(".png")
    cv2.imwrite(str(mask_path), composite)
    return mask_path


def _cocomask():
    """延迟加载 pycocotools（genpose2 环境需 ``pip install pycocotools``）。"""
    global _COCOMASK
    if _COCOMASK is None:
        try:
            from pycocotools import mask as cocomask_mod
        except ImportError as exc:
            raise ImportError(
                "SAM3 分割依赖 pycocotools，请在当前 Python 环境安装：pip install pycocotools"
            ) from exc
        _COCOMASK = cocomask_mod
    return _COCOMASK

DEFAULT_SAM3_ROOT = "/home/ubuntu/stephen/01-code/sam3"
DEFAULT_SAM3_PYTHON = "/home/ubuntu/miniconda3/envs/sam3/bin/python"
DEFAULT_SAM3_INFER_SCRIPT = "/home/ubuntu/stephen/01-code/sam3/scripts/infer.py"
DEFAULT_SAM3_PROMPT = "Plastic Reel"
DEFAULT_SAM3_THRESHOLD = 0.41
DEFAULT_SAM3_MASK_THRESHOLD = 0.50
DEFAULT_SAM3_CHECKPOINT = "/home/ubuntu/stephen/02-weight/sam3/sam3.pt"
# VLM→分割：``vlm_seg`` 生成 new_image 前将 VLM bbox 四向扩边；实例分割默认 yolo（``GENPOSE2_SEG_BACKEND``）
DEFAULT_VLM_ROI_MARGIN_PX = 10


def _env_first(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


@dataclass
class Sam3SegmentationResult:
    """SAM3 分割输出：detection_ism.json + GenPose2 用 mask.exr。"""

    detection_ism_path: Path
    mask_exr: Path
    score: float
    num_instances: int = 1
    instance_scores: Optional[List[float]] = None
    instance_dets: Optional[List[Dict[str, Any]]] = None
    vis_ism_path: Optional[Path] = None


# 与 sam3/scripts/infer.py VIS_COLORS 一致（OpenCV BGR）
_VIS_COLORS_BGR: Tuple[Tuple[int, int, int], ...] = (
    (0, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 128),
    (64, 64, 255),
    (255, 64, 64),
)


def _mask_to_rle(binary_mask: np.ndarray) -> Dict[str, object]:
    mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = _cocomask().encode(mask)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def _bbox_xywh_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("SAM3 mask is empty")
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, int(x2 - x1 + 1), int(y2 - y1 + 1)]


def _load_mask_image(path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    gray = np.array(Image.open(path).convert("L"))
    if gray.shape[0] != height or gray.shape[1] != width:
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
    return gray > 127


def _decode_rle_dict(rle: Dict[str, Any]) -> np.ndarray:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    size = rle["size"]
    h, w = int(size[0]), int(size[1])
    decoded = _cocomask().decode({"counts": counts, "size": [h, w]})
    return decoded.astype(bool)


def _pick_from_pred_json(
    data: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, float, List[int]]:
    scores = data.get("pred_scores") or data.get("scores")
    masks = data.get("pred_masks") or data.get("masks")
    boxes = data.get("pred_boxes") or data.get("bbox") or data.get("bboxes")

    if masks is None and "segmentation" in data:
        seg = data["segmentation"]
        if isinstance(seg, dict) and "counts" in seg:
            mask = _decode_rle_dict(seg)
            score = float(data.get("score", data.get("confidence", 1.0)))
            if mask.shape[0] != image_size[1] or mask.shape[1] != image_size[0]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    image_size,
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            return mask, score, _bbox_xywh_from_mask(mask)

    if isinstance(masks, list) and masks:
        idx = 0
        if scores is not None and len(scores):
            idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
            score = float(scores[idx])
        else:
            score = float(data.get("confidence", 1.0))
        item = masks[idx]
        if isinstance(item, dict) and "counts" in item:
            mask = _decode_rle_dict(item)
        elif isinstance(item, (list, np.ndarray)):
            mask = np.asarray(item, dtype=bool)
        else:
            raise RuntimeError(f"unsupported mask entry type in SAM3 json: {type(item)}")
        if mask.shape[0] != image_size[1] or mask.shape[1] != image_size[0]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                image_size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if boxes is not None and len(boxes) > idx:
            box = boxes[idx]
            if isinstance(box, (list, tuple)) and len(box) == 4:
                if all(0.0 <= float(v) <= 1.0 for v in box):
                    w, h = image_size
                    x, y, bw, bh = [float(v) for v in box]
                    bbox = [int(x * w), int(y * h), int(bw * w), int(bh * h)]
                else:
                    bbox = [int(round(v)) for v in box]
            else:
                bbox = _bbox_xywh_from_mask(mask)
        else:
            bbox = _bbox_xywh_from_mask(mask)
        return mask, score, bbox

    raise RuntimeError("SAM3 json has no usable mask fields")


def _parse_sam3_output(work_dir: Path, rgb_path: Path) -> Tuple[np.ndarray, float, List[int]]:
    with Image.open(rgb_path) as im:
        image_size = im.size  # (W, H)

    for jf in sorted(work_dir.rglob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            try:
                return _pick_from_pred_json(data, image_size)
            except RuntimeError:
                continue
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if "segmentation" in data[0] or "bbox" in data[0]:
                det = max(data, key=lambda d: float(d.get("score", 0.0)))
                seg = det["segmentation"]
                mask = _decode_rle_dict(seg) if isinstance(seg, dict) else np.asarray(seg, dtype=bool)
                if mask.shape[0] != image_size[1] or mask.shape[1] != image_size[0]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        image_size,
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                return mask, float(det.get("score", 1.0)), list(det.get("bbox", _bbox_xywh_from_mask(mask)))

    mask_candidates = [
        p
        for p in work_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and "mask" in p.name.lower()
    ]
    if not mask_candidates:
        mask_candidates = list(work_dir.rglob("*.png"))
    if mask_candidates:
        path = sorted(mask_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        mask = _load_mask_image(path, image_size)
        return mask, 1.0, _bbox_xywh_from_mask(mask)

    raise RuntimeError(f"no mask or json found under SAM3 output dir: {work_dir}")


def _sam3_root() -> Path:
    return Path(
        _env_first("GENPOSE2_SAM3_ROOT", "SAM6D_SAM3_ROOT", default=DEFAULT_SAM3_ROOT)
    ).expanduser().resolve()


def _sam3_python() -> str:
    py = _env_first("GENPOSE2_SAM3_PYTHON", "SAM6D_SAM3_PYTHON", default=DEFAULT_SAM3_PYTHON)
    return str(Path(py).expanduser().resolve())


def _sam3_infer_script() -> Path:
    script = _env_first(
        "GENPOSE2_SAM3_INFER_SCRIPT",
        "SAM6D_SAM3_INFER_SCRIPT",
        default=DEFAULT_SAM3_INFER_SCRIPT,
    )
    path = Path(script).expanduser()
    if path.is_file():
        return path.resolve()
    return (_sam3_root() / "scripts" / "infer.py").resolve()


def _sam3_checkpoint() -> Optional[str]:
    ckpt = _env_first("GENPOSE2_SAM3_CHECKPOINT", "SAM6D_SAM3_CHECKPOINT", default="")
    if not ckpt:
        if Path(DEFAULT_SAM3_CHECKPOINT).is_file():
            return DEFAULT_SAM3_CHECKPOINT
        return None
    return str(Path(ckpt).expanduser().resolve())


def _validate_sam3_toolchain(python_exe: str, infer_script: Path) -> None:
    py_path = Path(python_exe)
    if not py_path.is_file():
        raise FileNotFoundError(
            f"SAM3 python 不存在: {py_path}\n"
            f"请设置 GENPOSE2_SAM3_PYTHON 或 SAM6D_SAM3_PYTHON（默认 {DEFAULT_SAM3_PYTHON}）"
        )
    if not infer_script.is_file():
        raise FileNotFoundError(
            f"SAM3 infer.py 不存在: {infer_script}\n"
            f"请设置 GENPOSE2_SAM3_INFER_SCRIPT 或 SAM6D_SAM3_INFER_SCRIPT"
        )


def _decode_detection_mask(det: Dict[str, Any], image_size: Tuple[int, int]) -> np.ndarray:
    """从 detection_ism 单条记录解码 bool mask，image_size 为 (W, H)。"""
    width, height = image_size
    if "segmentation" in det:
        seg = det["segmentation"]
        if isinstance(seg, dict) and "counts" in seg:
            mask = _decode_rle_dict(seg)
        else:
            mask = np.asarray(seg, dtype=bool)
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return mask
    exr = det.get("segmentation_mask_exr")
    if exr and Path(exr).is_file():
        gray = np.array(Image.open(exr).convert("L"))
        if gray.shape[0] != height or gray.shape[1] != width:
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
        return gray > 127
    raise RuntimeError("detection entry has no segmentation or segmentation_mask_exr")


def write_genpose_mask_exr_from_ism(
    ism_json_path: Path,
    rgb_path: Path,
    mask_exr_path: Path,
    *,
    max_instances: int = 0,
) -> tuple[float, int, List[float], List[Dict[str, Any]]]:
    """
    读取 ``detection_ism.json``（COCO RLE），写出 GenPose2 ``mask.exr``。

    ``max_instances <= 0`` 表示保留 ISM 中全部实例（最多 254 个 id）；``max_instances=N`` 则按
    score 取 Top-N。mask 像素 id 与 score 排序一致：id=1 为最高分实例。

    :return: (best_score, num_instances, scores_for_mask_id_1..N, dets_for_each_mask_id)
    """
    ism_json_path = Path(ism_json_path).expanduser().resolve()
    rgb_path = Path(rgb_path).expanduser().resolve()
    mask_exr_path = Path(mask_exr_path).expanduser().resolve()

    dets = json.loads(ism_json_path.read_text(encoding="utf-8"))
    if not isinstance(dets, list) or not dets:
        raise RuntimeError(f"empty or invalid detection_ism.json: {ism_json_path}")

    with Image.open(rgb_path) as im:
        image_size = im.size  # (W, H)
    width, height = image_size

    ordered = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    if max_instances <= 0:
        ordered = ordered[:254]
    else:
        ordered = ordered[: max(1, min(max_instances, 254))]

    composite = np.zeros((height, width), dtype=np.uint8)
    best_score = float(ordered[0].get("score", 1.0))
    instance_scores: List[float] = []
    instance_dets: List[Dict[str, Any]] = []
    next_id = 1
    for det in ordered:
        mask = _decode_detection_mask(det, image_size)
        fill = mask & (composite == 0)
        if not np.any(fill):
            continue
        composite[fill] = np.uint8(next_id)
        instance_scores.append(float(det.get("score", 0.0)))
        instance_dets.append(det)
        next_id += 1

    if not np.any(composite > 0):
        raise RuntimeError(f"no foreground pixels in mask from {ism_json_path}")

    num_instances = len(instance_scores)
    saved_mask_path = _save_instance_id_mask(composite, mask_exr_path)
    print(
        f"[sam3_seg] instance mask -> {saved_mask_path} "
        f"(instances={num_instances}/{len(dets)} ism dets, best_score={best_score:.4f})"
    )
    return best_score, num_instances, instance_scores, instance_dets


def visualize_sam3_ism(
    rgb_path: Path,
    instance_dets: List[Dict[str, Any]],
    output_path: Path,
    *,
    prompt: Optional[str] = None,
    mask_alpha: float = 0.5,
    instance_ids: Optional[List[int]] = None,
) -> Path:
    """
    在 RGB 上绘制 SAM3 多实例 mask、bbox 与 score，写入 ``vis_ism.png`` 等路径。

    ``instance_ids`` 与 ``mask.exr`` 中 id 一致时（默认 1..N），图例显示 ``id=#``。
    """
    rgb_path = Path(rgb_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read rgb for SAM3 vis: {rgb_path}")
    height, width = bgr.shape[:2]
    image_size = (width, height)

    if not instance_dets:
        cv2.imwrite(str(output_path), bgr)
        print(f"[sam3_seg_backend] vis (empty dets) -> {output_path}")
        return output_path

    label = (prompt or DEFAULT_SAM3_PROMPT).split()[0] or "obj"
    overlay = bgr.astype(np.float32)

    for idx, det in enumerate(instance_dets):
        mask = _decode_detection_mask(det, image_size)
        color = _VIS_COLORS_BGR[idx % len(_VIS_COLORS_BGR)]
        color_arr = np.array(color, dtype=np.float32)
        overlay[mask] = mask_alpha * color_arr + (1.0 - mask_alpha) * overlay[mask]

        bbox = det.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, bw, bh = [int(round(v)) for v in bbox]
        else:
            x, y, bw, bh = _bbox_xywh_from_mask(mask)
        x2, y2 = x + bw, y + bh

        inst_id = (
            int(instance_ids[idx])
            if instance_ids is not None and idx < len(instance_ids)
            else idx + 1
        )
        score = float(det.get("score", 0.0))
        cv2.rectangle(overlay, (x, y), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"{label} id={inst_id} {score:.3f}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), overlay.astype(np.uint8))
    print(f"[sam3_seg_backend] vis_ism -> {output_path} ({len(instance_dets)} instances)")
    return output_path


def visualize_sam3_mask_exr(
    rgb_path: Path,
    mask_exr_path: Path,
    output_path: Path,
    *,
    alpha: float = 0.55,
) -> Path:
    """基于实例 id mask 图着色叠加。"""
    rgb_path = Path(rgb_path).expanduser().resolve()
    mask_path = Path(mask_exr_path).expanduser().resolve()
    if mask_path.suffix.lower() == ".exr" and not mask_path.is_file():
        mask_path = mask_path.with_suffix(".png")
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read rgb for SAM3 vis: {rgb_path}")
    id_map = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if id_map is None:
        raise FileNotFoundError(f"cannot read instance mask: {mask_path}")
    if id_map.ndim == 3:
        id_map = id_map[:, :, 0]
    overlay = bgr.astype(np.float32)
    for inst_id in sorted(int(v) for v in np.unique(id_map) if int(v) > 0):
        color = np.array(_VIS_COLORS_BGR[(inst_id - 1) % len(_VIS_COLORS_BGR)], dtype=np.float32)
        region = id_map == inst_id
        overlay[region] = alpha * color + (1.0 - alpha) * overlay[region]
        ys, xs = np.where(region)
        if len(xs):
            cv2.putText(
                overlay,
                f"id={inst_id}",
                (int(xs.min()), max(0, int(ys.min()) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color.tolist(),
                2,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(out), overlay.astype(np.uint8))
    print(f"[sam3_seg] vis_sam3_seg -> {out}")
    return out


def get_instance_bool_masks(
    instance_dets: List[Dict[str, Any]],
    image_size: Tuple[int, int],
) -> List[np.ndarray]:
    """从 SAM3 detection 列表解码各实例 bool mask，``image_size`` 为 (W, H)。"""
    return [_decode_detection_mask(det, image_size) for det in instance_dets]


def run_sam3_segmentation(
    rgb_path: Path,
    output_dir: Path,
    *,
    prompt: Optional[str] = None,
    threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    infer_script: Optional[Path] = None,
    python_exe: Optional[str] = None,
    mask_exr_out: Optional[Path] = None,
    max_instances: int = 0,
) -> Sam3SegmentationResult:
    rgb_path = rgb_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    sam3_root = _sam3_root()
    script = (infer_script or _sam3_infer_script()).expanduser().resolve()
    py = python_exe or _sam3_python()
    _validate_sam3_toolchain(py, script)

    prompt_text = (
        prompt
        if prompt is not None
        else _env_first("GENPOSE2_SAM3_PROMPT", "SAM6D_SAM3_PROMPT", default=DEFAULT_SAM3_PROMPT)
    )
    thresh = (
        float(threshold)
        if threshold is not None
        else float(
            _env_first(
                "GENPOSE2_SAM3_THRESHOLD",
                "SAM6D_SAM3_THRESHOLD",
                default=str(DEFAULT_SAM3_THRESHOLD),
            )
        )
    )
    mask_thresh = (
        float(mask_threshold)
        if mask_threshold is not None
        else float(
            _env_first(
                "GENPOSE2_SAM3_MASK_THRESHOLD",
                "SAM6D_SAM3_MASK_THRESHOLD",
                default=str(DEFAULT_SAM3_MASK_THRESHOLD),
            )
        )
    )

    sam6d_results = output_dir / "sam6d_results"
    sam6d_results.mkdir(parents=True, exist_ok=True)
    json_path = sam6d_results / "detection_ism.json"

    # 与手动命令一致：python / abs/infer.py --image ... --output-dir <abs> ...
    cmd = [
        py,
        str(script),
        "--image",
        str(rgb_path.resolve()),
        "--prompt",
        prompt_text,
        "--output-dir",
        str(output_dir.resolve()),
        "--threshold",
        str(thresh),
        "--mask-threshold",
        str(mask_thresh),
    ]
    checkpoint = _sam3_checkpoint()
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])

    print(f"[sam3_seg_backend] sam3_root(cwd)={sam3_root}")
    print(f"[sam3_seg_backend] cmd: {' '.join(cmd)}")

    env = os.environ.copy()
    if os.environ.get("SAM6D_CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = os.environ["SAM6D_CUDA_VISIBLE_DEVICES"]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(sam3_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[sam3_seg_backend] elapsed_ms={elapsed_ms:.3f} returncode={proc.returncode}")
    if proc.stdout:
        print("[sam3_seg_backend] stdout tail:\n" + "\n".join(proc.stdout.splitlines()[-40:]))
    if proc.returncode != 0:
        raise RuntimeError(
            f"SAM3 infer failed with exit code {proc.returncode}\n"
            f"{proc.stdout[-4000:] if proc.stdout else ''}"
        )

    if not json_path.is_file():
        print(f"[sam3_seg_backend] {json_path} missing, fallback parse under {output_dir}")
        mask, score, bbox_xywh = _parse_sam3_output(output_dir, rgb_path)
        detection = {
            "scene_id": 0,
            "image_id": 0,
            "category_id": 1,
            "bbox": bbox_xywh,
            "score": float(score),
            "time": 0.0,
            "segmentation": _mask_to_rle(mask),
        }
        json_path.write_text(json.dumps([detection]), encoding="utf-8")
        print(f"[sam3_seg_backend] wrote {json_path} score={score:.4f}")
    else:
        dets = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(dets, list):
            raise RuntimeError(f"SAM3 detection_ism.json must be a list, got {type(dets).__name__}")
        print(f"[sam3_seg_backend] using {json_path} n_detections={len(dets)}")

    if mask_exr_out is None:
        mask_exr_out = sam6d_results / "mask.exr"
    else:
        mask_exr_out = Path(mask_exr_out)

    best_score, num_instances, instance_scores, instance_dets = write_genpose_mask_exr_from_ism(
        json_path,
        rgb_path,
        mask_exr_out,
        max_instances=max_instances,
    )

    vis_ism_path = sam6d_results / "vis_ism.png"
    vis_sam3_seg_path = sam6d_results / "vis_sam3_seg.png"
    instance_ids = list(range(1, num_instances + 1))
    if instance_dets:
        visualize_sam3_ism(
            rgb_path,
            instance_dets,
            vis_ism_path,
            prompt=prompt_text,
            instance_ids=instance_ids,
        )
        visualize_sam3_mask_exr(rgb_path, mask_exr_out, vis_sam3_seg_path)
    elif vis_ism_path.is_file():
        print(f"[sam3_seg_backend] keep subprocess vis_ism: {vis_ism_path}")
    else:
        print(f"[sam3_seg_backend] no instance_dets for vis, skip")

    return Sam3SegmentationResult(
        detection_ism_path=json_path,
        mask_exr=mask_exr_out,
        score=best_score,
        num_instances=num_instances,
        instance_scores=instance_scores,
        instance_dets=instance_dets,
        vis_ism_path=vis_ism_path if vis_ism_path.is_file() else None,
    )
