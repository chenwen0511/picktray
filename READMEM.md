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
export GENPOSE2_SAM3_ROOT=/home/ubuntu/stephen/01-code/sam3
export GENPOSE2_SAM3_PYTHON=/home/ubuntu/miniconda3/envs/sam3/bin/python
export GENPOSE2_SAM3_INFER_SCRIPT=/home/ubuntu/stephen/01-code/sam3/scripts/infer.py
export GENPOSE2_SAM3_CHECKPOINT=/home/ubuntu/stephen/02-weight/sam3/sam3.pt

export GENPOSE2_SAM3_PROMPT="Plastic Reel"
export GENPOSE2_SAM3_THRESHOLD=0.41
export GENPOSE2_SAM3_MASK_THRESHOLD=0.50

export GENPOSE2_USE_VLM_ROI_FILTER=1
export GENPOSE2_VLM_API_URL="http://192.168.100.92:8000/v1/chat/completions"
export GENPOSE2_VLM_MODEL="qwen3-vl-4b"
export GENPOSE2_VLM_ROI_MARGIN_PX=10
export GENPOSE2_VLM_MIN_INTERSECTION_PX=1
```

说明：

- `seg/vlm_seg.py` 本身建议从当前项目所用的 Python / conda 环境启动。
- 真正执行 `SAM3` 推理时，会再调用 `GENPOSE2_SAM3_PYTHON` 指向的 Python。
- 如果你的 `VLM` 服务地址不是默认值，需要修改 `GENPOSE2_VLM_API_URL`。

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

export GENPOSE2_SAM3_ROOT=/home/ubuntu/stephen/01-code/sam3
export GENPOSE2_SAM3_PYTHON=/home/ubuntu/miniconda3/envs/sam3/bin/python
export GENPOSE2_SAM3_INFER_SCRIPT=/home/ubuntu/stephen/01-code/sam3/scripts/infer.py
export GENPOSE2_SAM3_CHECKPOINT=/home/ubuntu/stephen/02-weight/sam3/sam3.pt
export GENPOSE2_SAM3_PROMPT="Plastic Reel"
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
