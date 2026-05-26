from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh

from seg.sam3_seg import get_instance_bool_masks


@dataclass
class CloudStats:
    raw_points: int
    voxel_points: int
    filtered_points: int


@dataclass
class RGBDFrame:
    rgb_path: Path
    depth_path: Path
    camera_path: Path
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    depth_scale: float

    @property
    def image_size(self) -> Tuple[int, int]:
        h, w = self.rgb.shape[:2]
        return w, h


@dataclass
class MeshModel:
    mesh_path: Path
    mesh: trimesh.Trimesh
    sampled_points: np.ndarray
    processed_pcd: o3d.geometry.PointCloud
    origin_model: np.ndarray
    extents: np.ndarray


class RGBDInputLoader:
    @staticmethod
    def camera_json_to_K(camera: dict[str, Any]) -> np.ndarray:
        k = camera.get("cam_K")
        if not isinstance(k, list) or len(k) != 9:
            raise ValueError("camera.json 必须包含 9 个元素的 cam_K")
        return np.asarray(k, dtype=np.float64).reshape(3, 3)

    @staticmethod
    def load_rgb_array(rgb_path: Path) -> np.ndarray:
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cannot read rgb: {rgb_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def load_depth_array(depth_path: Path, depth_scale: float) -> np.ndarray:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"cannot read depth: {depth_path}")
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return np.asarray(depth, dtype=np.float32) * float(depth_scale)

    @staticmethod
    def resolve_depth_scale(camera_json: dict[str, Any], depth_path: Path) -> float:
        depth_scale = float(camera_json.get("depth_scale", 0.001))
        if depth_path.suffix.lower() == ".png" and depth_scale == 1.0:
            depth_scale = 0.001
        return depth_scale

    @staticmethod
    def prepare_depth_meters(depth: np.ndarray) -> np.ndarray:
        out = depth.astype(np.float32, copy=True)
        out[(out < 0.001) | ~np.isfinite(out)] = 0.0
        return out

    @classmethod
    def load(cls, rgb_path: Path, depth_path: Path, camera_path: Path) -> RGBDFrame:
        rgb_path = rgb_path.expanduser().resolve()
        depth_path = depth_path.expanduser().resolve()
        camera_path = camera_path.expanduser().resolve()
        with camera_path.open("r", encoding="utf-8") as f:
            camera_json = json.load(f)
        rgb = cls.load_rgb_array(rgb_path)
        K = cls.camera_json_to_K(camera_json)
        depth_scale = cls.resolve_depth_scale(camera_json, depth_path)
        depth = cls.prepare_depth_meters(cls.load_depth_array(depth_path, depth_scale))
        h, w = rgb.shape[:2]
        if depth.shape[:2] != (h, w):
            raise ValueError(f"rgb/depth size mismatch: rgb={(h, w)}, depth={depth.shape[:2]}")
        return RGBDFrame(
            rgb_path=rgb_path,
            depth_path=depth_path,
            camera_path=camera_path,
            rgb=rgb,
            depth=depth,
            K=K,
            depth_scale=float(depth_scale),
        )


class MaskResolver:
    @staticmethod
    def read_mask_png(mask_path: Path, *, instance_id: int = 1) -> np.ndarray:
        gray = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if gray is None:
            raise FileNotFoundError(f"cannot read mask: {mask_path}")
        if gray.ndim == 3:
            gray = gray[:, :, 0]
        gray = np.asarray(gray)
        if gray.dtype == np.bool_:
            return gray.astype(bool)
        if instance_id > 0 and np.any(gray == instance_id):
            return gray == instance_id
        return gray > 0

    @staticmethod
    def load_detection_mask(
        detection_ism_path: Path,
        image_size: Tuple[int, int],
        *,
        instance_id: int = 1,
    ) -> np.ndarray:
        dets = json.loads(detection_ism_path.read_text(encoding="utf-8"))
        if not isinstance(dets, list) or not dets:
            raise RuntimeError(f"empty detection list: {detection_ism_path}")
        masks = get_instance_bool_masks(dets, image_size)
        idx = max(0, instance_id - 1)
        if idx >= len(masks):
            raise IndexError(f"instance_id={instance_id} 超出 detection_ism.json 范围（共 {len(masks)} 个实例）")
        return masks[idx].astype(bool)

    @classmethod
    def load(
        cls,
        image_size: Tuple[int, int],
        *,
        mask_path: Optional[Path],
        detection_ism_path: Optional[Path],
        instance_id: int,
    ) -> Tuple[np.ndarray, Path]:
        if mask_path is not None and mask_path.is_file():
            return cls.read_mask_png(mask_path, instance_id=instance_id), mask_path.resolve()
        if detection_ism_path is not None and detection_ism_path.is_file():
            return (
                cls.load_detection_mask(detection_ism_path, image_size, instance_id=instance_id),
                detection_ism_path.resolve(),
            )
        raise FileNotFoundError("未找到可用的 mask 输入；请提供 --mask、--detection-ism 或 --seg-output-dir")

    @staticmethod
    def bbox_xywh(mask: np.ndarray) -> list[int]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return [0, 0, 0, 0]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        return [x1, y1, int(x2 - x1 + 1), int(y2 - y1 + 1)]

    @staticmethod
    def resolve_segmentation_inputs(seg_output_dir: Optional[Path]) -> Tuple[Optional[Path], Optional[Path]]:
        if seg_output_dir is None:
            return None, None
        results_dir = seg_output_dir / "results"
        sam6d_dir = seg_output_dir / "sam6d_results"
        mask_path = results_dir / "mask_instances.png"
        for candidate in (
            results_dir / "detection_ism.json",
            results_dir / "detection_ism_filtered.json",
            sam6d_dir / "detection_ism.json",
        ):
            if candidate.is_file():
                return (mask_path if mask_path.is_file() else None, candidate)
        return (mask_path if mask_path.is_file() else None, None)


class PointCloudProcessor:
    def __init__(
        self,
        *,
        voxel_size: float,
        sor_nb_neighbors: int,
        sor_std_ratio: float,
        max_depth_m: float,
    ) -> None:
        self.voxel_size = float(voxel_size)
        self.sor_nb_neighbors = int(sor_nb_neighbors)
        self.sor_std_ratio = float(sor_std_ratio)
        self.max_depth_m = float(max_depth_m)

    @staticmethod
    def make_point_cloud(points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return pcd

    @staticmethod
    def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int = 30) -> None:
        if len(pcd.points) == 0:
            return
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        pcd.normalize_normals()

    def preprocess(self, pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, CloudStats]:
        raw_points = len(pcd.points)
        voxel = pcd.voxel_down_sample(voxel_size=max(self.voxel_size, 1e-4))
        voxel_points = len(voxel.points)
        if voxel_points == 0:
            raise RuntimeError("体素下采样后点云为空，请检查 mask 或 depth")
        filtered, _ = voxel.remove_statistical_outlier(
            nb_neighbors=self.sor_nb_neighbors,
            std_ratio=self.sor_std_ratio,
        )
        filtered_points = len(filtered.points)
        if filtered_points == 0:
            raise RuntimeError("SOR 过滤后点云为空，请调大阈值或检查输入")
        radius = max(self.voxel_size * 2.5, 0.008)
        self.estimate_normals(filtered, radius=radius)
        return filtered, CloudStats(
            raw_points=raw_points,
            voxel_points=voxel_points,
            filtered_points=filtered_points,
        )

    def backproject_masked_depth(
        self,
        frame: RGBDFrame,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        depth = frame.depth
        K = frame.K
        valid = mask.astype(bool) & np.isfinite(depth) & (depth >= 0.001) & (depth <= self.max_depth_m)
        ys, xs = np.where(valid)
        if len(xs) == 0:
            raise RuntimeError("mask 内没有有效深度点，无法反投影")

        z = depth[ys, xs].astype(np.float64)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x = (xs.astype(np.float64) - cx) * z / fx
        y = (ys.astype(np.float64) - cy) * z / fy
        points = np.stack([x, y, z], axis=1)
        colors = frame.rgb[ys, xs].astype(np.float64) / 255.0
        return points, colors

    def build_observation_cloud(self, frame: RGBDFrame, mask: np.ndarray) -> Tuple[o3d.geometry.PointCloud, CloudStats]:
        points, colors = self.backproject_masked_depth(frame, mask)
        raw_pcd = self.make_point_cloud(points, colors)
        return self.preprocess(raw_pcd)

    @staticmethod
    def guess_translation_from_mask(mask: np.ndarray, depth: np.ndarray, K: np.ndarray) -> Optional[np.ndarray]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        valid = mask.astype(bool) & (depth >= 0.001)
        if not np.any(valid):
            return None
        uc = float((xs.min() + xs.max()) / 2.0)
        vc = float((ys.min() + ys.max()) / 2.0)
        zc = float(np.median(depth[valid]))
        center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1.0], dtype=np.float64).reshape(3, 1)) * zc
        return center.reshape(3).astype(np.float64)

    @staticmethod
    def write_debug_point_cloud(path: Path, pcd: o3d.geometry.PointCloud) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)


class MeshModelLoader:
    def __init__(self, *, mesh_scale: float, sample_points: int, processor: PointCloudProcessor) -> None:
        self.mesh_scale = float(mesh_scale)
        self.sample_points = int(sample_points)
        self.processor = processor

    def _load_mesh(self, mesh_path: Path) -> trimesh.Trimesh:
        loaded = trimesh.load(str(mesh_path), process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        if isinstance(loaded, trimesh.points.PointCloud):
            mesh = trimesh.Trimesh(
                vertices=np.asarray(loaded.vertices),
                faces=np.empty((0, 3), dtype=np.int64),
                process=False,
            )
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded.copy()
        else:
            raise TypeError(f"不支持的 mesh 类型: {type(loaded).__name__}")
        if self.mesh_scale != 1.0:
            mesh.apply_scale(self.mesh_scale)
        return mesh

    def _sample_model_cloud(self, mesh: trimesh.Trimesh) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        if len(mesh.faces) > 0:
            points, face_idx = trimesh.sample.sample_surface(mesh, self.sample_points)
            normals = np.asarray(mesh.face_normals)[np.asarray(face_idx)]
        else:
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            if len(verts) == 0:
                raise RuntimeError("mesh 没有可用顶点")
            if len(verts) >= self.sample_points:
                choice = np.random.default_rng(0).choice(len(verts), size=self.sample_points, replace=False)
            else:
                choice = np.random.default_rng(0).choice(len(verts), size=self.sample_points, replace=True)
            points = verts[choice]
            normals = None
        pcd = self.processor.make_point_cloud(points)
        if normals is not None and len(normals) == len(points):
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
        else:
            self.processor.estimate_normals(pcd, radius=0.01)
        return pcd, np.asarray(points, dtype=np.float64)

    def load(self, mesh_path: Path) -> Tuple[MeshModel, CloudStats]:
        mesh_path = mesh_path.expanduser().resolve()
        mesh = self._load_mesh(mesh_path)
        raw_model_pcd, sampled_points = self._sample_model_cloud(mesh)
        processed_pcd, model_stats = self.processor.preprocess(raw_model_pcd)
        model = MeshModel(
            mesh_path=mesh_path,
            mesh=mesh,
            sampled_points=sampled_points,
            processed_pcd=processed_pcd,
            origin_model=np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0),
            extents=np.asarray(mesh.extents, dtype=np.float64),
        )
        return model, model_stats
