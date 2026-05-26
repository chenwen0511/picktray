from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import open3d as o3d


@dataclass
class AlignmentMetrics:
    method: str
    fitness: float
    inlier_rmse: float
    translation_error_m: Optional[float]


@dataclass
class CoarseAlignmentResult:
    pose: np.ndarray
    selected_method: str
    metrics: Dict[str, Any]


class CoarsePoseEstimator:
    def __init__(self, *, voxel_size: float) -> None:
        self.voxel_size = float(voxel_size)

    @staticmethod
    def _pca_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh.T
        if np.linalg.det(basis) < 0:
            basis[:, -1] *= -1.0
        return centroid, basis

    @staticmethod
    def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation.reshape(3)
        return pose

    def _estimate_pca(
        self,
        model_pcd: o3d.geometry.PointCloud,
        obs_pcd: o3d.geometry.PointCloud,
        *,
        translation_hint: Optional[np.ndarray],
        eval_threshold: float,
    ) -> Tuple[np.ndarray, AlignmentMetrics]:
        model_pts = np.asarray(model_pcd.points)
        obs_pts = np.asarray(obs_pcd.points)
        model_center, model_basis = self._pca_frame(model_pts)
        obs_center, obs_basis = self._pca_frame(obs_pts)

        sign_options = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    diag = np.diag([sx, sy, sz])
                    if np.linalg.det(diag) > 0:
                        sign_options.append(diag)

        best_pose = np.eye(4, dtype=np.float64)
        best_rank = (-1.0, -float("inf"), -float("inf"))
        best_metric = AlignmentMetrics(method="pca", fitness=0.0, inlier_rmse=float("inf"), translation_error_m=None)

        for diag in sign_options:
            rotation = obs_basis @ diag @ model_basis.T
            if np.linalg.det(rotation) < 0:
                continue
            translation = obs_center - rotation @ model_center
            pose = self._make_transform(rotation, translation)
            eval_result = o3d.pipelines.registration.evaluate_registration(
                model_pcd,
                obs_pcd,
                eval_threshold,
                pose,
            )
            translation_error = None
            if translation_hint is not None:
                translation_error = float(np.linalg.norm(translation - translation_hint))
            rank = (
                float(eval_result.fitness),
                -float(eval_result.inlier_rmse),
                -(translation_error if translation_error is not None else 0.0),
            )
            if rank > best_rank:
                best_rank = rank
                best_pose = pose
                best_metric = AlignmentMetrics(
                    method="pca",
                    fitness=float(eval_result.fitness),
                    inlier_rmse=float(eval_result.inlier_rmse),
                    translation_error_m=translation_error,
                )
        return best_pose, best_metric

    def _estimate_fpfh(
        self,
        model_pcd: o3d.geometry.PointCloud,
        obs_pcd: o3d.geometry.PointCloud,
    ) -> Optional[Tuple[np.ndarray, AlignmentMetrics]]:
        if len(model_pcd.points) < 30 or len(obs_pcd.points) < 30:
            return None

        source = model_pcd.voxel_down_sample(max(self.voxel_size * 1.5, 0.003))
        target = obs_pcd.voxel_down_sample(max(self.voxel_size * 1.5, 0.003))
        if len(source.points) < 20 or len(target.points) < 20:
            return None

        normal_radius = max(self.voxel_size * 3.0, 0.01)
        feature_radius = max(self.voxel_size * 6.0, 0.02)
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source,
            o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target,
            o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
        )

        threshold = max(self.voxel_size * 6.0, 0.02)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999),
        )
        if result.transformation is None:
            return None

        return np.asarray(result.transformation, dtype=np.float64), AlignmentMetrics(
            method="fpfh_ransac",
            fitness=float(result.fitness),
            inlier_rmse=float(result.inlier_rmse),
            translation_error_m=None,
        )

    def estimate(
        self,
        model_pcd: o3d.geometry.PointCloud,
        obs_pcd: o3d.geometry.PointCloud,
        *,
        translation_hint: Optional[np.ndarray],
    ) -> CoarseAlignmentResult:
        threshold = max(self.voxel_size * 5.0, 0.02)
        pca_pose, pca_metric = self._estimate_pca(
            model_pcd,
            obs_pcd,
            translation_hint=translation_hint,
            eval_threshold=threshold,
        )
        metrics: Dict[str, Any] = {"pca": asdict(pca_metric)}
        best_pose = pca_pose
        best_metric = pca_metric

        fpfh = self._estimate_fpfh(model_pcd, obs_pcd)
        if fpfh is not None:
            fpfh_pose, fpfh_metric = fpfh
            metrics["fpfh_ransac"] = asdict(fpfh_metric)
            pca_rank = (pca_metric.fitness, -pca_metric.inlier_rmse)
            fpfh_rank = (fpfh_metric.fitness, -fpfh_metric.inlier_rmse)
            if fpfh_rank > pca_rank:
                best_pose = fpfh_pose
                best_metric = fpfh_metric

        metrics["selected"] = best_metric.method
        return CoarseAlignmentResult(
            pose=best_pose,
            selected_method=best_metric.method,
            metrics=metrics,
        )
