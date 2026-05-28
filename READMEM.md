# PickTray 工程说明

## 1. 工程功能概览

`PickTray` 是一个基于 `RGB-D + 相机内参 K + CAD` 的目标位姿估计工程，面向细长/薄片类托盘对象，核心能力包括：

- 实例分割：`SAM3` 产出候选实例，`VLM` 提供目标 ROI，按 ROI 交集规则筛选唯一目标实例。
- 点云构建：由 `depth + K + mask` 反投影得到观测点云，并进行降采样和去噪。
- 粗配准：基于 `PCA` 与 `FPFH+RANSAC` 获取初始位姿候选并自动择优。
- 精配准：多阶段 `ICP(Point-to-Plane)` 迭代优化位姿。
- 对称约束：支持自动识别模型对称轴并锁定自旋，减少轴对称物体姿态抖动。
- 可视化输出：叠加实例 mask、CAD 采样点投影、RGB 坐标轴，并导出调试点云与位姿文件。

---

## 2. 方案原理（默认流程）

默认入口为 `pose_pipeline.py`，完整链路如下：

1. 对输入 `rgb` 调用 `SAM3`（HTTP）得到实例分割候选。
2. 对同一张 `rgb` 调用 `VLM` 获取目标 ROI。
3. 解码每个实例 mask，与 ROI 计算交集像素数：

```python
intersection_px = int(np.count_nonzero(instance_mask & roi_mask))
```

4. 仅保留与 ROI 交集最大的实例（并列时按分数打破并列）。
5. 用保留实例 mask 从深度图反投影生成观测点云。
6. 将 CAD 采样为模型点云，执行粗配准（PCA / FPFH-RANSAC）与精配准（ICP）。
7. 根据对称轴约束规范化姿态，输出最终 `4x4 pose` 与可视化结果。

```text
输入: RGB / Depth / K / CAD
  ├─ RGB -> SAM3(HTTP) 实例分割 -> 候选实例
  ├─ RGB -> VLM 提取 ROI
  └─ CAD -> 采样模型点云

候选实例 + ROI
  -> 计算 ROI 与每个实例 mask 的交集面积
  -> 保留交集面积最大的实例

保留实例 + Depth + K
  -> 反投影得到观测点云

观测点云 + 模型点云
  -> 粗配准 (PCA / FPFH-RANSAC)
  -> 精配准 (多阶段 ICP)
  -> 对称轴自旋约束
  -> 输出 pose_4x4 / vis_pose / 调试点云
```

---

## 3. 工程目录与关键文件

- `pose_pipeline.py`：端到端位姿估计主入口。
- `seg/vlm_seg.py`：`VLM + SAM3 + ROI` 分割筛选入口。
- `seg/sam3_seg.py`：SAM3 HTTP 调用与分割结果解析。
- `point_cloud.py`：输入解析、mask 读取、点云构建与预处理。
- `coarse.py`：粗配准模块（PCA / FPFH-RANSAC）。
- `fine.py`：精配准模块（ICP）与对称约束模块。
- `test/inputs/`：示例输入（`rgb/depth/camera.json`）。
- `test/cad/`：示例 CAD 模型。

---

## 4. 输入与输出说明

### 输入

- `rgb`：彩色图（分割与可视化使用）。
- `depth`：深度图（米或毫米编码，程序按配置自动处理）。
- `camera.json`：至少包含 `cam_K`（9 个元素）。
- `mesh`：CAD 网格（如 `ply`）。

### 主要输出（`output-dir/results/`）

- `pose_4x4.txt`：最终位姿矩阵。
- `pose_coarse_4x4.txt`：粗配准位姿。
- `pose_result.json`：完整结构化结果（包含 timing、各阶段指标）。
- `vis_pose.png`：mask + CAD 投影 + 坐标轴可视化。
- `scene_filtered.ply`：观测点云（预处理后）。
- `model_coarse.ply`：粗配准后的模型点云。
- `model_registered.ply`：最终配准后的模型点云。

分割相关输出（`output-dir/segmentation/results/` 或独立分割输出目录）：

- `detection_ism.json` / `detection_ism_filtered.json`
- `mask_instances.png`
- `vis_ism.png`
- `vis_sam3_seg.png`
- `vlm_roi.json`

---

## 5. 环境准备

### 5.1 Python 环境

建议使用你当前项目环境（例如 `foundationpose`）：

```bash
conda activate foundationpose
```

### 5.2 依赖安装

如果本地尚未安装，可在环境中安装：

```bash
pip install numpy opencv-python pillow requests open3d trimesh pycocotools
```

> 说明：`open3d`/`pycocotools` 通常是最容易缺失的依赖。

### 5.3 外部服务准备

本工程默认依赖两个 HTTP 服务：

- `SAM3` 推理服务（默认 `http://127.0.0.1:18002/infer`）
- `VLM` 服务（默认 `http://192.168.100.92:8000/v1/chat/completions`）

---

## 6. 运行前环境变量

在仓库根目录设置（可按需调整）：

```bash
export GENPOSE2_SAM3_PROMPT="Plastic Reel"
export GENPOSE2_SAM3_THRESHOLD=0.41
export GENPOSE2_SAM3_MASK_THRESHOLD=0.50
export GENPOSE2_SAM3_API_URL="http://127.0.0.1:18002/infer"
export GENPOSE2_SAM3_TIMEOUT_S=300

export GENPOSE2_USE_VLM_ROI_FILTER=1
export GENPOSE2_VLM_API_URL="http://192.168.100.92:8000/v1/chat/completions"
export GENPOSE2_VLM_MODEL="qwen3-vl-4b"
export GENPOSE2_VLM_ROI_MARGIN_PX=10
export GENPOSE2_VLM_MIN_INTERSECTION_PX=1
```

---

## 7. 如何运行

### 7.1 仅运行分割（SAM3 + VLM ROI 筛选）

```bash
python seg/vlm_seg.py \
  --image test/inputs/rgb.png \
  --output-dir test/output_sam3_vlm
```

关闭 VLM、仅保留 SAM3 原始结果：

```bash
python seg/vlm_seg.py \
  --image test/inputs/rgb.png \
  --output-dir test/output_sam3_only \
  --skip-vlm
```

### 7.2 运行完整位姿估计（默认自动先分割）

```bash
python pose_pipeline.py \
  --rgb test/inputs/rgb.png \
  --depth test/inputs/depth.png \
  --camera test/inputs/camera.json \
  --mesh test/cad/tray_180mm_centered_mesh_v2.ply \
  --output-dir test/output_pose_pipeline
```

### 7.3 复用已有分割结果运行位姿估计

```bash
python pose_pipeline.py \
  --rgb test/inputs/rgb.png \
  --depth test/inputs/depth.png \
  --camera test/inputs/camera.json \
  --mesh test/cad/tray_180mm_centered_mesh_v2.ply \
  --seg-output-dir test/output_sam3_vlm \
  --output-dir test/output_pose_pipeline_reuse_seg
```

---

## 8. SAM3 HTTP 接口示例

`seg/sam3_seg.py` 当前通过 REST 调用：

```bash
curl -X POST http://127.0.0.1:18002/infer \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/rgb.png",
    "prompt": "Plastic Reel",
    "threshold": 0.41,
    "mask_threshold": 0.50,
    "save_vis": true,
    "output_dir": "/path/to/output"
  }'
```

程序会优先读取服务落盘的：

- `output_dir/sam6d_results/detection_ism.json`

---

## 9. 常见问题排查

- `ModuleNotFoundError: open3d`：当前环境缺少 `open3d`，请先安装依赖。
- `SAM3 HTTP infer failed`：检查 `GENPOSE2_SAM3_API_URL` 是否可达、服务是否启动。
- `VLM request failed`：检查 `GENPOSE2_VLM_API_URL` 与模型服务状态。
- `mask 太小，无法进行 3D 配准`：分割结果不正确或目标过小，先检查 `vis_ism.png` 与 `vlm_roi_vis.png`。
- `rgb/depth size mismatch`：输入图像分辨率不一致，需要先对齐数据。

---

## 10. 当前默认策略说明

- 默认开启 `VLM ROI` 筛选，不是直接用 SAM3 最大实例。
- 实例选择规则是 **ROI 交集面积最大**，不是实例总面积最大。
- 位姿估计默认会自动触发分割；仅在显式传入 `--mask` / `--detection-ism` / `--seg-output-dir` 时复用已有分割结果。
# PickTray 实例分割说明

当前仓库默认使用 `SAM3 + VLM ROI` 的两阶段流程做实例级分割，目标是先让 `SAM3` 在整张原图上产出多个候选实例，再让 `VLM` 只负责指出“要哪一个目标区域”，最后按 `ROI` 与实例掩码的交集面积做筛选。

## 输入数据

本仓库当前这组测试数据位于 `test/`：

- `test/inputs/rgb.png`：实例分割实际使用的输入图像
- `test/inputs/depth.png`：后续位姿估计使用
- `test/inputs/camera.json`：后续位姿估计使用的相机内参
- `test/cad/tray_180mm_centered_mesh_v2.ply`：后续 CAD 配准或位姿估计使用

注意：**实例分割阶段只直接读取 `rgb.png`**。`depth`、`camera.json`、`cad` 不参与当前这一步的分割推理。

## 默认分割流程

默认入口是 `seg/vlm_seg.py`，实际流程如下：

1. 对原始 `rgb.png` 调用 `SAM3`，输出整张图上的实例分割结果。
2. `SAM3` 输出会写到 `output_dir/sam6d_results/detection_ism.json`，其中每个实例都是一个单独的 mask。
3. 对同一张原始 `rgb.png` 调用 `VLM`，让它只返回目标物体的一个 `ROI bbox`。
4. 将 `ROI bbox` 从 `0~1000` 归一化坐标还原到像素坐标，并按 `GENPOSE2_VLM_ROI_MARGIN_PX` 进行适度扩边。
5. 将 `SAM3` 的每个实例 mask 解码成布尔图，分别计算它和 `ROI` 的交集像素数：

```python
intersection_px = int(np.count_nonzero(instance_mask & roi_mask))
```

6. 过滤掉与 `ROI` 没有交集的实例，只保留**与 ROI 交集面积最大的那个实例**。
7. 如果交集像素数相同，则继续用检测分数 `score` 做 tie-break；如果还相同，再用实例面积打破并列。
8. 将筛选后的单实例结果重新写回 `results/detection_ism.json`、`results/mask_instances.png`、`results/vis_ism.png`、`results/vis_sam3_seg.png`，供后续位姿估计或配准模块使用。

## 关键输出文件

执行 `seg/vlm_seg.py` 后，常见输出包括：

- `output_dir/sam6d_results/detection_ism.json`：`SAM3` 原始实例结果
- `output_dir/results/detection_ism_raw.json`：原始实例结果备份
- `output_dir/results/detection_ism_filtered.json`：按 ROI 筛选后的实例结果
- `output_dir/results/detection_ism.json`：筛选后结果的最终别名
- `output_dir/results/mask_instances.png`：筛选后单实例 mask
- `output_dir/results/vis_ism.png`：筛选后实例叠加可视化
- `output_dir/results/vis_sam3_seg.png`：筛选后 mask 可视化
- `output_dir/results/vlm_roi.json`：本次 ROI、交集像素、保留实例索引等调试信息
- `output_dir/results/vlm_roi_vis.png`：VLM 返回 ROI 的可视化

## 运行前环境变量

建议先在仓库根目录设置以下环境变量：

```bash
export GENPOSE2_SAM3_PROMPT="Plastic Reel"
export GENPOSE2_SAM3_THRESHOLD=0.41
export GENPOSE2_SAM3_MASK_THRESHOLD=0.50
export GENPOSE2_SAM3_API_URL="http://127.0.0.1:18002/infer"
export GENPOSE2_SAM3_TIMEOUT_S=300

export GENPOSE2_USE_VLM_ROI_FILTER=1
export GENPOSE2_VLM_API_URL="http://192.168.100.92:8000/v1/chat/completions"
export GENPOSE2_VLM_MODEL="qwen3-vl-4b"
export GENPOSE2_VLM_ROI_MARGIN_PX=10
export GENPOSE2_VLM_MIN_INTERSECTION_PX=1
```

说明：

- `seg/vlm_seg.py` 本身建议从当前项目所用的 Python / conda 环境启动。
- `SAM3` 分割现在改为 HTTP 服务调用，不再通过本地命令行脚本启动推理。
- `GENPOSE2_SAM3_API_URL` 需要指向可用的 `SAM3` 推理接口（默认 `/infer`）。
- `SAM3` 健康检查默认地址为 `/health`（由 `/infer` 自动推导）。
- 如果你的 `VLM` 服务地址不是默认值，需要修改 `GENPOSE2_VLM_API_URL`。

## SAM3 HTTP 调用说明

`seg/sam3_seg.py` 当前通过 REST API 调用 `SAM3`，请求示例如下：

```bash
curl -X POST http://127.0.0.1:18002/infer \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/rgb.png",
    "prompt": "Plastic Reel",
    "threshold": 0.41,
    "mask_threshold": 0.50,
    "save_vis": true,
    "output_dir": "/path/to/output"
  }'
```

服务侧若正确落盘 `output_dir/sam6d_results/detection_ism.json`，本工程会继续复用该文件，生成后续筛选与可视化结果。

## 命令行调用方法

### 1. 默认流程：SAM3 + VLM ROI 筛选

在仓库根目录执行：

```bash
python seg/vlm_seg.py \
  --image test/inputs/rgb.png \
  --output-dir test/output_sam3_vlm
```

这条命令会：

- 先在整张 `rgb.png` 上跑 `SAM3`
- 再调用 `VLM` 产出 ROI
- 最后只保留与 ROI 交集面积最大的实例

### 2. 只跑 SAM3，不启用 VLM ROI 筛选

```bash
python seg/vlm_seg.py \
  --image test/inputs/rgb.png \
  --output-dir test/output_sam3_only \
  --skip-vlm
```

这条命令会保留 `SAM3` 的原始实例结果，不做 ROI 过滤。

### 3. 针对当前测试数据的一组完整示例

先激活你当前项目的运行环境，再执行命令。例如：

```bash
cd /home/ubuntu/stephen/01-code/PickTray

# 例如：
# conda activate foundationpose

export GENPOSE2_SAM3_PROMPT="Plastic Reel"
export GENPOSE2_SAM3_API_URL="http://127.0.0.1:18002/infer"
export GENPOSE2_SAM3_TIMEOUT_S=300
export GENPOSE2_USE_VLM_ROI_FILTER=1

python seg/vlm_seg.py \
  --image test/inputs/rgb.png \
  --output-dir test/output_picktray_demo
```

运行完成后，优先查看：

```bash
test/output_picktray_demo/results/vlm_roi_vis.png
test/output_picktray_demo/results/vis_ism.png
test/output_picktray_demo/results/vis_sam3_seg.png
test/output_picktray_demo/results/vlm_roi.json
```

## 补充说明

- 当前默认选择规则是：**保留与 ROI 交集面积最大的实例**，不是保留实例总面积最大的那个。
- `VLM` 只负责“指目标区域”，不直接输出最终 mask。
- `SAM3` 负责“产出候选实例”，最终由 `ROI` 交集规则决定留下哪一个实例。
- 后续如果要接 `depth + K + CAD` 做位姿估计，应直接使用这里筛选后的 `detection_ism.json` 或 `mask_instances.png`。

## 位姿估计默认流程

`pose_pipeline.py` 的默认主流程是：

1. `VLM` 在原图上提取目标 `bbox / ROI`
2. `SAM3` 在原图上做实例分割
3. 只保留与 `ROI` 交集面积最大的那个实例
4. 用该实例的 `mask + depth + K + CAD` 继续做位姿估计

也就是说，**默认情况下位姿估计会先调用分割**，再继续做 3D 配准；只有在你显式提供已有的 `--mask`、`--detection-ism` 或 `--seg-output-dir` 时，才会复用现成分割结果。

### 位姿估计命令行示例

默认一键流程：

```bash
python pose_pipeline.py \
  --rgb test/inputs/rgb.png \
  --depth test/inputs/depth.png \
  --camera test/inputs/camera.json \
  --mesh test/cad/tray_180mm_centered_mesh_v2.ply \
  --output-dir test/output_pose_pipeline
```

这条命令默认会自动执行：

- `VLM` 提取目标 `bbox / ROI`
- `SAM3` 实例分割
- ROI 筛选实例
- 基于筛选实例做位姿估计

如果你已经有现成分割输出，也可以显式复用：

```bash
python pose_pipeline.py \
  --rgb test/inputs/rgb.png \
  --depth test/inputs/depth.png \
  --camera test/inputs/camera.json \
  --mesh test/cad/tray_180mm_centered_mesh_v2.ply \
  --seg-output-dir test/output_sam3_vlm \
  --output-dir test/output_pose_pipeline_reuse_seg
```
