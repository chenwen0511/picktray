from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from coarse import CoarsePoseEstimator
from fine import FinePoseRefiner, PoseSymmetryLocker, SymmetryAxisResolver
from point_cloud import MaskResolver, MeshModelLoader, PointCloudProcessor, RGBDInputLoader
from seg.vlm_seg import run_vlm_sam3_filter_pipeline


@dataclass
class PosePipelineResult:
    rgb_path: str
    depth_path: str
    camera_path: str
    mesh_path: str
    mask_path: str
    coarse_method: str
    segmentation_output_dir: Optional[str]
    depth_scale_m: float
    mesh_scale: float
    symmetry_axis: str
    yaw_locked: bool
    camera_matrix: List[List[float]]
    mask_bbox_xywh: List[int]
    observation_stats: Dict[str, int]
    model_stats: Dict[str, int]
    coarse_alignment: Dict[str, Any]
    fine_alignment: Dict[str, Any]
    coarse_pose_4x4: List[List[float]]
    pose_4x4_before_yaw_lock: List[List[float]]
    pose_4x4: List[List[float]]
    t_m: List[float]
    t_mm: List[float]
    rotation_euler_zyx_rad: List[float]
    outputs: Dict[str, str]
    timing_s: Dict[str, float]


def _rotation_matrix_to_euler_zyx(rotation: np.ndarray) -> List[float]:
    r00, r10 = float(rotation[0, 0]), float(rotation[1, 0])
    r20, r21, r22 = float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2])
    r01, r11 = float(rotation[0, 1]), float(rotation[1, 1])

    sy = math.sqrt(r00 * r00 + r10 * r10)
    if sy > 1e-6:
        rx = math.atan2(r21, r22)
        ry = math.atan2(-r20, sy)
        rz = math.atan2(r10, r00)
    else:
        rx = math.atan2(-r01, r11)
        ry = math.atan2(-r20, sy)
        rz = 0.0
    return [float(rx), float(ry), float(rz)]


def _project_points(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = points_cam[:, 2:3]
    uvw = (K @ points_cam.T).T
    return uvw[:, :2] / np.clip(z, 1e-9, None)


def _transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (pose[:3, :3] @ points.T).T + pose[:3, 3]


def _draw_projected_model_points(
    image: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray,
    model_points: np.ndarray,
    *,
    color: Tuple[int, int, int] = (0, 255, 255),
    radius: int = 1,
    alpha: float = 0.8,
    max_points: int = 5000,
) -> np.ndarray:
    vis = image.copy()
    if model_points.size == 0:
        return vis

    points = np.asarray(model_points, dtype=np.float64)
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
        points = points[idx]

    points_cam = _transform_points(points, pose)
    valid_z = points_cam[:, 2] > 1e-6
    if not np.any(valid_z):
        return vis

    points_cam = points_cam[valid_z]
    uv = np.rint(_project_points(points_cam, K)).astype(np.int32)
    h, w = vis.shape[:2]
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    uv = uv[inside]
    if len(uv) == 0:
        return vis

    overlay = vis.copy()
    for u, v in uv:
        cv2.circle(overlay, (int(u), int(v)), radius, color, -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0.0)


def _draw_pose_axes(
    image: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray,
    *,
    origin_model: np.ndarray,
    axis_length: float,
) -> np.ndarray:
    vis = image.copy()
    axes_model = np.asarray(
        [
            origin_model,
            origin_model + np.asarray([axis_length, 0.0, 0.0]),
            origin_model + np.asarray([0.0, axis_length, 0.0]),
            origin_model + np.asarray([0.0, 0.0, axis_length]),
        ],
        dtype=np.float64,
    )
    axes_cam = _transform_points(axes_model, pose)
    if np.any(axes_cam[:, 2] <= 1e-6):
        return vis
    uv = _project_points(axes_cam, K).astype(np.int32)
    origin_uv = tuple(uv[0])
    cv2.line(vis, origin_uv, tuple(uv[1]), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(vis, origin_uv, tuple(uv[2]), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(vis, origin_uv, tuple(uv[3]), (0, 0, 255), 2, cv2.LINE_AA)
    return vis


def _overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = image.copy()
    color = np.zeros_like(vis)
    color[:, :, 1] = 255
    alpha = 0.28
    vis[mask] = ((1.0 - alpha) * vis[mask] + alpha * color[mask]).astype(np.uint8)
    return vis


def run_pose_pipeline(
    *,
    rgb_path: Path,
    depth_path: Path,
    camera_path: Path,
    mesh_path: Path,
    output_dir: Path,
    mesh_scale: float = 0.001,
    mask_path: Optional[Path] = None,
    detection_ism_path: Optional[Path] = None,
    seg_output_dir: Optional[Path] = None,
    instance_id: int = 1,
    run_segmentation: bool = False,
    skip_vlm: bool = False,
    voxel_size: float = 0.005,
    sor_nb_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
    sample_points: int = 8000,
    max_depth_m: float = 2.0,
    lock_symmetry_yaw: bool = True,
    symmetry_axis: str = "auto",
    yaw_reference_axis: str = "x",
) -> PosePipelineResult:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timing: Dict[str, float] = {}
    seg_dir: Optional[Path] = seg_output_dir.expanduser().resolve() if seg_output_dir else None
    auto_run_segmentation = run_segmentation or (mask_path is None and detection_ism_path is None and seg_dir is None)

    if auto_run_segmentation:
        t0 = time.perf_counter()
        seg_dir = output_dir / "segmentation"
        seg_result = run_vlm_sam3_filter_pipeline(rgb_path, seg_dir, skip_vlm=skip_vlm)
        timing["segmentation_s"] = time.perf_counter() - t0
        mask_path = seg_dir / "results" / "mask_instances.png"
        detection_ism_path = seg_result.sam3.detection_ism_path
    elif seg_dir is not None:
        seg_mask_path, seg_ism_path = MaskResolver.resolve_segmentation_inputs(seg_dir)
        mask_path = mask_path or seg_mask_path
        detection_ism_path = detection_ism_path or seg_ism_path

    frame = RGBDInputLoader.load(rgb_path, depth_path, camera_path)
    mask, mask_source = MaskResolver.load(
        frame.image_size,
        mask_path=mask_path,
        detection_ism_path=detection_ism_path,
        instance_id=instance_id,
    )
    if int(mask.sum()) < 4:
        raise RuntimeError("mask 太小，无法进行 3D 配准")

    processor = PointCloudProcessor(
        voxel_size=voxel_size,
        sor_nb_neighbors=sor_nb_neighbors,
        sor_std_ratio=sor_std_ratio,
        max_depth_m=max_depth_m,
    )

    t0 = time.perf_counter()
    obs_pcd, obs_stats = processor.build_observation_cloud(frame, mask)
    timing["pointcloud_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mesh_loader = MeshModelLoader(mesh_scale=mesh_scale, sample_points=sample_points, processor=processor)
    mesh_model, model_stats = mesh_loader.load(mesh_path)
    resolved_symmetry_axis = SymmetryAxisResolver.resolve(mesh_model.extents, symmetry_axis)
    timing["model_s"] = time.perf_counter() - t0

    translation_hint = processor.guess_translation_from_mask(mask, frame.depth, frame.K)

    t0 = time.perf_counter()
    coarse_result = CoarsePoseEstimator(voxel_size=voxel_size).estimate(
        mesh_model.processed_pcd,
        obs_pcd,
        translation_hint=translation_hint,
    )
    timing["coarse_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    fine_result = FinePoseRefiner(voxel_size=voxel_size).refine(
        mesh_model.processed_pcd,
        obs_pcd,
        coarse_result.pose,
    )
    timing["icp_s"] = time.perf_counter() - t0

    final_pose = fine_result.pose.copy()
    if lock_symmetry_yaw:
        final_pose = PoseSymmetryLocker.lock_self_rotation(
            final_pose,
            symmetry_axis=resolved_symmetry_axis,
            reference_axis=yaw_reference_axis,
        )

    axis_length = max(float(np.max(mesh_model.extents)), 1e-3) * 0.35

    t0 = time.perf_counter()
    vis = _overlay_mask(frame.rgb, mask)
    vis = _draw_projected_model_points(vis, frame.K, final_pose, mesh_model.sampled_points)
    vis = _draw_pose_axes(
        vis,
        frame.K,
        final_pose,
        origin_model=mesh_model.origin_model,
        axis_length=axis_length,
    )
    vis_path = results_dir / "vis_pose.png"
    cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    timing["vis_s"] = time.perf_counter() - t0

    scene_ply = results_dir / "scene_filtered.ply"
    model_registered_ply = results_dir / "model_registered.ply"
    coarse_registered_ply = results_dir / "model_coarse.ply"
    processor.write_debug_point_cloud(scene_ply, obs_pcd)
    model_coarse_pcd = copy.deepcopy(mesh_model.processed_pcd)
    model_coarse_pcd.transform(coarse_result.pose)
    processor.write_debug_point_cloud(coarse_registered_ply, model_coarse_pcd)
    model_final_pcd = copy.deepcopy(mesh_model.processed_pcd)
    model_final_pcd.transform(final_pose)
    processor.write_debug_point_cloud(model_registered_ply, model_final_pcd)

    pose_txt = results_dir / "pose_4x4.txt"
    coarse_pose_txt = results_dir / "pose_coarse_4x4.txt"
    np.savetxt(str(pose_txt), final_pose)
    np.savetxt(str(coarse_pose_txt), coarse_result.pose)

    t_m = final_pose[:3, 3].astype(float).tolist()
    t_mm = (final_pose[:3, 3] * 1000.0).astype(float).tolist()
    euler = _rotation_matrix_to_euler_zyx(final_pose[:3, :3])

    result = PosePipelineResult(
        rgb_path=str(frame.rgb_path),
        depth_path=str(frame.depth_path),
        camera_path=str(frame.camera_path),
        mesh_path=str(mesh_model.mesh_path),
        mask_path=str(mask_source),
        coarse_method=coarse_result.selected_method,
        segmentation_output_dir=str(seg_dir) if seg_dir is not None else None,
        depth_scale_m=float(frame.depth_scale),
        mesh_scale=float(mesh_scale),
        symmetry_axis=resolved_symmetry_axis,
        yaw_locked=bool(lock_symmetry_yaw),
        camera_matrix=frame.K.astype(float).tolist(),
        mask_bbox_xywh=MaskResolver.bbox_xywh(mask),
        observation_stats=asdict(obs_stats),
        model_stats=asdict(model_stats),
        coarse_alignment=coarse_result.metrics,
        fine_alignment=fine_result.metrics,
        coarse_pose_4x4=coarse_result.pose.astype(float).tolist(),
        pose_4x4_before_yaw_lock=fine_result.pose.astype(float).tolist(),
        pose_4x4=final_pose.astype(float).tolist(),
        t_m=t_m,
        t_mm=t_mm,
        rotation_euler_zyx_rad=euler,
        outputs={
            "pose_txt": str(pose_txt),
            "pose_coarse_txt": str(coarse_pose_txt),
            "vis_pose": str(vis_path),
            "scene_filtered_ply": str(scene_ply),
            "model_coarse_ply": str(coarse_registered_ply),
            "model_registered_ply": str(model_registered_ply),
        },
        timing_s=timing,
    )

    result_path = results_dir / "pose_result.json"
    result.outputs["pose_result_json"] = str(result_path)
    result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PickTray: RGB-D + K + CAD pose pipeline")
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seg-output-dir",
        type=Path,
        default=None,
        help="已有分割输出目录；若不提供且也不提供 mask/detection，则默认自动执行 VLM+SAM3 分割",
    )
    parser.add_argument("--mask", type=Path, default=None, help="直接指定二值/实例 mask 图")
    parser.add_argument("--detection-ism", type=Path, default=None, help="直接指定 detection_ism.json")
    parser.add_argument("--instance-id", type=int, default=1)
    parser.add_argument("--mesh-scale", type=float, default=0.001, help="CAD 顶点缩放，默认 mm -> m")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="点云体素大小（米）")
    parser.add_argument("--sor-nb-neighbors", type=int, default=20)
    parser.add_argument("--sor-std-ratio", type=float, default=2.0)
    parser.add_argument("--sample-points", type=int, default=8000)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument(
        "--run-segmentation",
        action="store_true",
        help="显式强制重新执行 VLM+SAM3 分割；默认在未提供 mask/detection/seg-output-dir 时自动启用",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="仅在需要执行分割时生效：跳过 VLM ROI，直接使用 SAM3 实例结果",
    )
    parser.add_argument(
        "--lock-symmetry-yaw",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="是否锁定模型对称轴上的自旋角",
    )
    parser.add_argument("--symmetry-axis", choices=["x", "y", "z", "auto"], default="auto")
    parser.add_argument("--yaw-reference-axis", choices=["x", "y", "z"], default="x")
    return parser


def main() -> int:
    parser = _build_argparser()
    args = parser.parse_args()

    result = run_pose_pipeline(
        rgb_path=args.rgb,
        depth_path=args.depth,
        camera_path=args.camera,
        mesh_path=args.mesh,
        output_dir=args.output_dir,
        mesh_scale=args.mesh_scale,
        mask_path=args.mask,
        detection_ism_path=args.detection_ism,
        seg_output_dir=args.seg_output_dir,
        instance_id=args.instance_id,
        run_segmentation=args.run_segmentation,
        skip_vlm=args.skip_vlm,
        voxel_size=args.voxel_size,
        sor_nb_neighbors=args.sor_nb_neighbors,
        sor_std_ratio=args.sor_std_ratio,
        sample_points=args.sample_points,
        max_depth_m=args.max_depth_m,
        lock_symmetry_yaw=args.lock_symmetry_yaw,
        symmetry_axis=args.symmetry_axis,
        yaw_reference_axis=args.yaw_reference_axis,
    )

    print(
        json.dumps(
            {
                "pose_result_json": result.outputs["pose_result_json"],
                "pose_txt": result.outputs["pose_txt"],
                "vis_pose": result.outputs["vis_pose"],
                "coarse_method": result.coarse_method,
                "t_mm": result.t_mm,
                "rotation_euler_zyx_rad": result.rotation_euler_zyx_rad,
                "timing_s": result.timing_s,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
