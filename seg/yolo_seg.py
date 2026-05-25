import os

# OpenCV EXR I/O（与 cutoop Dataset.load_mask 一致）需在 import cv2 之前设置
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from seg.sam3_seg import (
    Sam3SegmentationResult,
    _bbox_xywh_from_mask,
    _mask_to_rle,
    _save_instance_id_mask,
    visualize_sam3_ism,
    visualize_sam3_mask_exr,
)

_MODEL_CACHE: Dict[str, YOLO] = {}

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_WEIGHTS = _REPO_ROOT / "seg/weights/yolo.pt"
DEFAULT_YOLO_CONF = 0.25
DEFAULT_YOLO_IMGSZ = 640
DEFAULT_YOLO_CLASS_ID = 0


@dataclass
class YoloSegmentationResult:
    """YOLO 分割输出：GenPose2 / cutoop 可读的单通道 mask.exr。"""

    mask_exr: Path
    score: float = 1.0


def _load_model(weights_path: Path) -> YOLO:
    key = str(weights_path.resolve())
    print(f"[yolo_seg_backend] load request: {weights_path} (resolved: {key})")
    if key not in _MODEL_CACHE:
        if not weights_path.is_file():
            print(f"[yolo_seg_backend] weights not found: {weights_path}")
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        print(f"[yolo_seg_backend] loading YOLO model: {key}")
        # Force segmentation task to avoid TRT engine auto-guess as detect.
        _MODEL_CACHE[key] = YOLO(key, task="segment")
        print(f"[yolo_seg_backend] model loaded and cached: {key}")
    else:
        print(f"[yolo_seg_backend] model cache hit: {key}")
    return _MODEL_CACHE[key]


def _resize_mask(mask: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """将 YOLO mask 缩放到与 RGB 一致。image_size 为 (width, height)。"""
    width, height = image_size
    resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return resized > 0.5


def _read_rgb_size(rgb_path: Path) -> Tuple[int, int]:
    """返回 (width, height)，与 PIL Image.size 一致。"""
    with Image.open(rgb_path) as image:
        w, h = image.size
    return w, h


def instance_ids_to_genpose_exr_float(instance_u8: np.ndarray) -> np.ndarray:
    """
    cutoop Dataset.load_mask：cv2.imread 后对单通道执行 (img * 255).astype(uint8)。
    因此 EXR 中应保存 float32：像素值为 instance_id / 255（背景为 0），
    这样读回后 *255 得到 0,1,2,... 的 uint8 实例编号（与 Omni6D / GenPose2 InferDataset 一致）。
    """
    out = np.zeros(instance_u8.shape, dtype=np.float32)
    fg = instance_u8 > 0
    out[fg] = instance_u8[fg].astype(np.float32) / 255.0
    return out


def save_genpose2_mask_exr(instance_ids_hw: np.ndarray, exr_path: Path) -> None:
    """
    将 uint8 / int 实例图（0=背景，1..N=实例 id）写成 GenPose2 / cutoop 可读的 mask.exr。
    """
    if instance_ids_hw.ndim != 2:
        raise ValueError("instance_ids_hw 必须是二维 (H, W)")
    if int(instance_ids_hw.max()) > 254:
        raise ValueError("实例编号不能超过 254（uint8 与 /255 编码约定）")
    exr_path = Path(exr_path)
    exr_path.parent.mkdir(parents=True, exist_ok=True)
    to_write = instance_ids_to_genpose_exr_float(instance_ids_hw.astype(np.uint8))
    if not cv2.imwrite(str(exr_path), to_write):
        raise RuntimeError(f"cv2.imwrite 失败: {exr_path}（确认 OPENCV_IO_ENABLE_OPENEXR 且 OpenCV 带 EXR）")


def build_instance_mask_from_yolo(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    image_size: Tuple[int, int],
    *,
    class_id: Optional[int] = 0,
    max_instances: int = 1,
) -> np.ndarray:
    """
    根据 YOLO 输出构造 (H, W) uint8 实例图：0 背景，1..N 为实例（高分优先占像素）。
    image_size: (width, height)
    """
    width, height = image_size
    h, w = height, width
    candidate_indexes = list(range(len(scores)))
    if class_id is not None:
        candidate_indexes = [i for i in candidate_indexes if int(classes[i]) == class_id]
    if not candidate_indexes:
        raise RuntimeError(f"YOLO 在 class_id={class_id} 下无可用实例")

    order = sorted(candidate_indexes, key=lambda i: float(scores[i]), reverse=True)
    order = order[: max(1, max_instances)]

    composite = np.zeros((h, w), dtype=np.uint8)
    for rank, det_idx in enumerate(order):
        m = _resize_mask(masks[det_idx], image_size)
        fill = m & (composite == 0)
        composite[fill] = np.uint8(rank + 1)
    return composite


def _yolo_predict_instance_u8(
    weights_path: Path,
    rgb_path: Path,
    *,
    conf: float = 0.25,
    imgsz: int = 640,
    class_id: Optional[int] = 0,
    max_instances: int = 1,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """返回 (instance_u8 HW, best_score, image_size WH)。"""
    t0 = time.perf_counter()
    model = _load_model(weights_path)
    rgb_path = Path(rgb_path)
    image_size = _read_rgb_size(rgb_path)

    results = model.predict(str(rgb_path), conf=conf, imgsz=imgsz, verbose=False, task="segment")
    print(f"[yolo_seg_backend] predict elapsed_ms={(time.perf_counter() - t0) * 1000:.3f}")
    if not results:
        raise RuntimeError("YOLO returned no results")

    result = results[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("YOLO returned no segmentation masks")

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    masks = result.masks.data.detach().cpu().numpy()

    candidate_indexes = list(range(len(scores)))
    if class_id is not None:
        candidate_indexes = [idx for idx in candidate_indexes if classes[idx] == class_id]
    if not candidate_indexes:
        raise RuntimeError(f"YOLO returned no masks for class_id={class_id}")
    best_idx = max(candidate_indexes, key=lambda idx: float(scores[idx]))
    best_score = float(scores[best_idx])

    instance_u8 = build_instance_mask_from_yolo(
        masks, boxes, scores, classes, image_size, class_id=class_id, max_instances=max_instances
    )
    return instance_u8, best_score, image_size


def run_yolo_segmentation(
    weights_path: Path,
    rgb_path: Path,
    output_dir: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    class_id: Optional[int] = 0,
    max_instances: int = 1,
    mask_exr_out: Optional[Path] = None,
) -> YoloSegmentationResult:
    """
    YOLO 分割：仅写 GenPose2 / cutoop 所需的 mask.exr（独立管线，不产出其它检测 JSON）。

    :param mask_exr_out: mask.exr 输出路径；默认 ``output_dir/{rgb词干}_mask.exr``
    :param max_instances: >1 时按置信度从高到低填充实例 id，重叠处保留高分实例。
    """
    instance_u8, best_score, _ = _yolo_predict_instance_u8(
        weights_path,
        rgb_path,
        conf=conf,
        imgsz=imgsz,
        class_id=class_id,
        max_instances=max_instances,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = Path(rgb_path)
    if mask_exr_out is None:
        mask_exr_path = output_dir / f"{rgb_path.stem}_mask.exr"
    else:
        mask_exr_path = Path(mask_exr_out)

    if mask_exr_path.suffix.lower() == ".png":
        _save_instance_id_mask(instance_u8, mask_exr_path)
    else:
        save_genpose2_mask_exr(instance_u8, mask_exr_path)
    print(f"[yolo_seg_backend] mask -> {mask_exr_path} (instances max={int(instance_u8.max())})")

    return YoloSegmentationResult(mask_exr=mask_exr_path, score=best_score)


def _env_first(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def yolo_weights_path() -> Path:
    custom = _env_first("GENPOSE2_YOLO_WEIGHTS", "SAM6D_YOLO_WEIGHTS", default="")
    if custom:
        path = Path(custom).expanduser()
        if not path.is_absolute():
            path = (_REPO_ROOT / path).resolve()
        else:
            path = path.resolve()
        return path
    for cand in (
        DEFAULT_YOLO_WEIGHTS,
        _REPO_ROOT / "weights/yolo.pt",
        Path(__file__).resolve().parent / "weights/yolo.pt",
    ):
        if cand.is_file():
            return cand.resolve()
    return DEFAULT_YOLO_WEIGHTS.resolve()


def _yolo_conf() -> float:
    return float(_env_first("GENPOSE2_YOLO_CONF", default=str(DEFAULT_YOLO_CONF)))


def _yolo_imgsz() -> int:
    return int(_env_first("GENPOSE2_YOLO_IMGSZ", default=str(DEFAULT_YOLO_IMGSZ)))


def _yolo_class_id() -> Optional[int]:
    raw = _env_first("GENPOSE2_YOLO_CLASS_ID", default=str(DEFAULT_YOLO_CLASS_ID))
    if raw.lower() in ("", "none", "all", "-1"):
        return None
    return int(raw)


def instance_u8_to_ism_detections(
    instance_u8: np.ndarray,
    scores_by_id: Dict[int, float],
) -> List[Dict[str, Any]]:
    """将实例 id 图转为与 SAM3 ``detection_ism.json`` 相同结构的列表。"""
    dets: List[Dict[str, Any]] = []
    for inst_id in sorted(int(x) for x in np.unique(instance_u8) if int(x) > 0):
        mask = instance_u8 == inst_id
        if not np.any(mask):
            continue
        dets.append(
            {
                "scene_id": 0,
                "image_id": 0,
                "category_id": 1,
                "bbox": _bbox_xywh_from_mask(mask),
                "score": float(scores_by_id.get(inst_id, 1.0)),
                "time": 0.0,
                "segmentation": _mask_to_rle(mask),
            }
        )
    dets.sort(key=lambda d: float(d["score"]), reverse=True)
    return dets


def publish_detection_results(
    seg_rgb_path: Path,
    result: Sam3SegmentationResult,
    output_dir: Path,
    *,
    original_rgb_path: Optional[Path] = None,
    vis_prompt: str = "YOLO",
) -> Dict[str, Path]:
    """
    将 detection_ism、mask、可视化写入 ``output_dir/results/`` 与 ``sam6d_results/``。
    """
    output_dir = output_dir.expanduser().resolve()
    seg_rgb_path = seg_rgb_path.expanduser().resolve()
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    sam6d_results = output_dir / "sam6d_results"
    sam6d_results.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Path] = {}
    ism_text = result.detection_ism_path.read_text(encoding="utf-8")
    for dest in (
        results_dir / "detection_ism.json",
        sam6d_results / "detection_ism.json",
    ):
        dest.write_text(ism_text, encoding="utf-8")
        out["detection_ism"] = dest

    if result.instance_dets:
        vis_ism = results_dir / "vis_ism.png"
        visualize_sam3_ism(
            seg_rgb_path,
            result.instance_dets,
            vis_ism,
            prompt=vis_prompt,
            instance_ids=list(range(1, result.num_instances + 1)),
        )
        out["vis_ism"] = vis_ism
        sam6d_vis = sam6d_results / "vis_ism.png"
        sam6d_vis.write_bytes(vis_ism.read_bytes())
        out["vis_ism_sam6d"] = sam6d_vis

    if result.mask_exr.is_file():
        vis_seg = results_dir / "vis_sam3_seg.png"
        visualize_sam3_mask_exr(seg_rgb_path, result.mask_exr, vis_seg)
        out["vis_seg"] = vis_seg
        sam6d_seg = sam6d_results / "vis_sam3_seg.png"
        sam6d_seg.write_bytes(vis_seg.read_bytes())
        out["vis_seg_sam6d"] = sam6d_seg

    orig = (
        Path(original_rgb_path).expanduser().resolve()
        if original_rgb_path is not None
        else None
    )
    if (
        orig is not None
        and orig.is_file()
        and orig.resolve() != seg_rgb_path.resolve()
        and result.instance_dets
    ):
        vis_orig = results_dir / "vis_ism_orig.png"
        visualize_sam3_ism(
            orig,
            result.instance_dets,
            vis_orig,
            prompt=f"{vis_prompt} (orig rgb)",
            instance_ids=list(range(1, result.num_instances + 1)),
        )
        out["vis_ism_orig"] = vis_orig
        if result.mask_exr.is_file():
            vis_seg_orig = results_dir / "vis_seg_orig.png"
            visualize_sam3_mask_exr(orig, result.mask_exr, vis_seg_orig)
            out["vis_seg_orig"] = vis_seg_orig

    result.vis_ism_path = out.get("vis_ism")
    print(f"[yolo_seg] results artifacts -> {results_dir}")
    for k, p in out.items():
        print(f"  {k}: {p}")
    return out


def run_yolo_segmentation_ism(
    rgb_path: Path,
    output_dir: Path,
    *,
    weights_path: Optional[Path] = None,
    conf: Optional[float] = None,
    imgsz: Optional[int] = None,
    class_id: Optional[int] = None,
    max_instances: int = 1,
    mask_exr_out: Optional[Path] = None,
    original_rgb_path: Optional[Path] = None,
    write_vis: bool = True,
) -> Sam3SegmentationResult:
    """
    YOLO 实例分割，输出与 ``run_sam3_segmentation`` 相同接口（``detection_ism.json`` + mask 图）。

    典型用法：VLM 生成 ``rgb_vlm_masked.png`` 后，对该图调用本函数。
    """
    weights = (weights_path or yolo_weights_path()).expanduser().resolve()
    rgb_path = rgb_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    sam6d_results = output_dir / "sam6d_results"
    sam6d_results.mkdir(parents=True, exist_ok=True)

    conf_v = DEFAULT_YOLO_CONF if conf is None else float(conf)
    imgsz_v = DEFAULT_YOLO_IMGSZ if imgsz is None else int(imgsz)
    class_v = _yolo_class_id() if class_id is None else class_id

    instance_u8, best_score, _image_size = _yolo_predict_instance_u8(
        weights,
        rgb_path,
        conf=conf_v,
        imgsz=imgsz_v,
        class_id=class_v,
        max_instances=max_instances,
    )

    scores_by_id: Dict[int, float] = {i: best_score for i in range(1, int(instance_u8.max()) + 1)}

    instance_dets = instance_u8_to_ism_detections(instance_u8, scores_by_id)
    if not instance_dets:
        raise RuntimeError("YOLO produced empty instance mask")

    ism_path = sam6d_results / "detection_ism.json"
    ism_path.write_text(json.dumps(instance_dets, indent=2), encoding="utf-8")

    mask_path = Path(mask_exr_out) if mask_exr_out else (output_dir / "results" / "mask_instances.png")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    saved_mask = _save_instance_id_mask(instance_u8, mask_path)

    print(
        f"[yolo_seg] detection_ism -> {ism_path} "
        f"instances={len(instance_dets)} mask={saved_mask}"
    )

    seg_result = Sam3SegmentationResult(
        detection_ism_path=ism_path,
        mask_exr=saved_mask,
        score=float(instance_dets[0]["score"]),
        num_instances=len(instance_dets),
        instance_scores=[float(d["score"]) for d in instance_dets],
        instance_dets=instance_dets,
    )

    if write_vis:
        publish_detection_results(
            rgb_path,
            seg_result,
            output_dir,
            original_rgb_path=original_rgb_path,
            vis_prompt="YOLO",
        )

    return seg_result


def preload_yolo_model(
    weights_path: Path,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    class_id: Optional[int] = 0,
) -> None:
    """Preload YOLO model into cache and run one dummy warmup predict."""
    t0 = time.perf_counter()
    model = _load_model(weights_path)
    t1 = time.perf_counter()
    print(f"[yolo_seg_backend] preload _load_model elapsed_ms={(t1 - t0) * 1000:.3f}")

    # Dummy image (H, W, C) 与常见 RealSense 帧一致 480x640
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    t2 = time.perf_counter()
    results = model.predict(dummy, conf=conf, imgsz=imgsz, verbose=False, task="segment")
    t3 = time.perf_counter()
    print(f"[yolo_seg_backend] preload warmup predict elapsed_ms={(t3 - t2) * 1000:.3f}")

    if results:
        result = results[0]
        n = 0 if result.boxes is None else len(result.boxes)
        print(f"[yolo_seg_backend] preload warmup detections={n} class_filter={class_id}")
    print(f"[yolo_seg_backend] preload total elapsed_ms={(t3 - t0) * 1000:.3f}")


def _resolve_existing_file(path: Path, search_roots: Tuple[Path, ...]) -> Path:
    """依次尝试 path（绝对/相对 cwd）与各 search_roots 下的相对路径。"""
    path = Path(path)
    if path.is_file():
        return path.resolve()
    for root in search_roots:
        cand = (root / path).resolve()
        if cand.is_file():
            return cand
    return path.resolve()


def _cli_main() -> int:
    import argparse
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    seg_dir = Path(__file__).resolve().parent
    roots = (Path.cwd(), repo_root, seg_dir)

    parser = argparse.ArgumentParser(
        description="YOLO 分割：生成 GenPose2 用 mask.exr（cutoop Dataset.load_mask 可读）",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/yolo.pt"),
        help="YOLO 分割权重 .pt（默认相对仓库/segment 或当前目录）",
    )
    parser.add_argument(
        "--rgb",
        type=Path,
        default=Path("data/rgb.png"),
        help="输入 RGB 图像（png/jpg 等）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/yolo_seg_test"),
        help="输出目录；默认在此目录写入 {rgb 文件名}_mask.exr",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="只保留该类；设为 -1 表示不过滤类别",
    )
    parser.add_argument("--max-instances", type=int, default=1)
    parser.add_argument(
        "--mask-exr",
        type=Path,
        default=None,
        help="可选：直接指定 mask.exr 输出路径（默认 output-dir/{rgb_stem}_mask.exr）",
    )
    parser.add_argument(
        "--verify-cutoop",
        action="store_true",
        help="跑完后用 cutoop.Dataset.load_mask 读回 mask 并打印 unique 像素值",
    )
    args = parser.parse_args()

    weights_path = _resolve_existing_file(args.weights, roots)
    rgb_path = _resolve_existing_file(args.rgb, roots)
    if not weights_path.is_file():
        print(f"[cli] 找不到权重: {args.weights}（已尝试 cwd / 仓库根 / segment 目录）", file=sys.stderr)
        return 1
    if not rgb_path.is_file():
        print(f"[cli] 找不到 RGB: {args.rgb}", file=sys.stderr)
        print("[cli] 示例: python segment/yolo_seg_backend.py --rgb path/to/frame_color.png", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    class_id = None if args.class_id < 0 else args.class_id
    try:
        result = run_yolo_segmentation(
            weights_path,
            rgb_path,
            output_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            class_id=class_id,
            max_instances=args.max_instances,
            mask_exr_out=args.mask_exr,
        )
    except Exception as e:
        print(f"[cli] 推理失败: {e}", file=sys.stderr)
        return 1

    print("[cli] 完成")
    print(f"  mask_exr: {result.mask_exr}")

    if args.verify_cutoop:
        try:
            from cutoop.data_loader import Dataset

            loaded = Dataset.load_mask(str(result.mask_exr))
            uniq = np.unique(loaded)
            print(f"[cli] cutoop load_mask unique ids (uint8): {uniq.tolist()}")
        except Exception as e:
            print(f"[cli] verify-cutoop 跳过或失败: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())