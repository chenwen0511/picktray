"""VLM ROI 调试 CLI（实现见 ``seg/vlm_seg.py``）。"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from seg.vlm_seg import DEFAULT_VLM_PROMPT as PROMPT  # noqa: F401 — 兼容旧 import
from seg.vlm_seg import bbox_to_pixel, detect_vlm_roi, draw_vlm_roi_vis

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def detect_image(img_path: Path, *, save_dir: str = "results") -> None:
    img_path = Path(img_path)
    roi = detect_vlm_roi(img_path)
    if roi is None:
        print("[VLM] no ROI")
        return
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    det = {
        "label": roi.label,
        "bbox_2d": list(roi.bbox_pixel),
    }
    draw_vlm_roi_vis(img_path, roi.bbox_pixel, Path(save_dir) / img_path.name, label=roi.label)
    print(f"[VLM] bbox_pixel={roi.bbox_pixel} norm={roi.bbox_norm}")


if __name__ == "__main__":
    detect_image(
        Path(__file__).resolve().parent.parent / "test/multi-tray/dc330a18ba4206ce4fe80f16e1288240.jpg"
    )
