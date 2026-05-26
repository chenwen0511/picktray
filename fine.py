from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d


@dataclass
class FineAlignmentResult:
    pose: np.ndarray
    metrics: Dict[str, Any]


class FinePoseRefiner:
    def __init__(self, *, voxel_size: float) -> None:
        self.voxel_size = float(voxel_size)

    @staticmethod
    def _estimate_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int = 30) -> None:
        if len(pcd.points) == 0:
            return
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        pcd.normalize_normals()

    def refine(
        self,
        model_pcd: o3d.geometry.PointCloud,
        obs_pcd: o3d.geometry.PointCloud,
        init_pose: np.ndarray,
    ) -> FineAlignmentResult:
        source = copy.deepcopy(model_pcd)
        target = copy.deepcopy(obs_pcd)
        thresholds = [
            max(self.voxel_size * 4.0, 0.02),
            max(self.voxel_size * 2.0, 0.01),
            max(self.voxel_size, 0.005),
        ]
        pose = np.asarray(init_pose, dtype=np.float64)
        stage_results: List[Dict[str, Any]] = []

        for idx, threshold in enumerate(thresholds, start=1):
            self._estimate_normals(source, radius=max(threshold * 2.0, 0.01))
            self._estimate_normals(target, radius=max(threshold * 2.0, 0.01))
            result = o3d.pipelines.registration.registration_icp(
                source,
                target,
                threshold,
                pose,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
            )
            pose = np.asarray(result.transformation, dtype=np.float64)
            stage_results.append(
                {
                    "stage": idx,
                    "max_correspondence_distance": float(threshold),
                    "fitness": float(result.fitness),
                    "inlier_rmse": float(result.inlier_rmse),
                }
            )

        final_eval = o3d.pipelines.registration.evaluate_registration(source, target, thresholds[-1], pose)
        return FineAlignmentResult(
            pose=pose,
            metrics={
                "stages": stage_results,
                "fitness": float(final_eval.fitness),
                "inlier_rmse": float(final_eval.inlier_rmse),
            },
        )


class SymmetryAxisResolver:
    @staticmethod
    def resolve(mesh_extents: np.ndarray, symmetry_axis: str) -> str:
        axis = str(symmetry_axis).lower()
        if axis in {"x", "y", "z"}:
            return axis
        if axis != "auto":
            raise ValueError(f"unsupported symmetry axis: {symmetry_axis}")
        idx = int(np.argmin(np.asarray(mesh_extents, dtype=np.float64)))
        return ("x", "y", "z")[idx]


class PoseSymmetryLocker:
    @staticmethod
    def lock_self_rotation(
        pose: np.ndarray,
        *,
        symmetry_axis: str,
        reference_axis: str = "x",
    ) -> np.ndarray:
        axis_names = ("x", "y", "z")
        sym_idx = axis_names.index(symmetry_axis.lower())
        ref_name = reference_axis.lower()
        if ref_name not in axis_names or ref_name == symmetry_axis.lower():
            ref_name = next(name for name in axis_names if name != symmetry_axis.lower())
        ref_idx = axis_names.index(ref_name)

        world_basis = {
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }

        out = pose.copy()
        rotation = out[:3, :3]
        axes: List[Optional[np.ndarray]] = [None, None, None]

        preserved = rotation[:, sym_idx].astype(np.float64)
        preserved /= np.linalg.norm(preserved) + 1e-12
        axes[sym_idx] = preserved

        ref = world_basis[ref_name]
        if abs(np.dot(ref, preserved)) > 0.95:
            for alt_name in axis_names:
                alt = world_basis[alt_name]
                if alt_name != symmetry_axis.lower() and abs(np.dot(alt, preserved)) <= 0.95:
                    ref = alt
                    ref_idx = axis_names.index(alt_name)
                    break

        ref_proj = ref - preserved * float(np.dot(ref, preserved))
        ref_proj /= np.linalg.norm(ref_proj) + 1e-12
        axes[ref_idx] = ref_proj

        if axes[0] is not None and axes[1] is not None:
            axes[2] = np.cross(axes[0], axes[1])
            axes[2] /= np.linalg.norm(axes[2]) + 1e-12
            axes[1] = np.cross(axes[2], axes[0])
            axes[1] /= np.linalg.norm(axes[1]) + 1e-12
        elif axes[1] is not None and axes[2] is not None:
            axes[0] = np.cross(axes[1], axes[2])
            axes[0] /= np.linalg.norm(axes[0]) + 1e-12
            axes[2] = np.cross(axes[0], axes[1])
            axes[2] /= np.linalg.norm(axes[2]) + 1e-12
        elif axes[2] is not None and axes[0] is not None:
            axes[1] = np.cross(axes[2], axes[0])
            axes[1] /= np.linalg.norm(axes[1]) + 1e-12
            axes[0] = np.cross(axes[1], axes[2])
            axes[0] /= np.linalg.norm(axes[0]) + 1e-12
        else:
            raise RuntimeError("failed to build orthonormal axes from symmetry constraint")

        out[:3, :3] = np.column_stack(axes)
        return out
