# CNN/UNet 可视化工具说明（静态参考）

本文档解释 `explain_cnn_visualization.py` 生成的图片和报告字段含义，供实验/研究时统一参考。

## 1. 工具输出文件

针对每个样本（例如 `sample_0007`）会输出：

- `sample_0007_explain_panel.png`
- `sample_0007_gradcam.png`
- `sample_0007_guided_gradient.png`
- `sample_0007_guided_gradcam.png`
- `sample_0007_scorecam.png`
- `sample_0007_explain_report.json`

批量模式会额外输出：

- `batch_explain_summary.json`
- `batch_explain_grid.png`（启用 `--batch-grid-png` 时）

## 2. Panel 图（总览图）怎么读

`sample_xxxx_explain_panel.png` 是 2x4 面板：

第一行（模型结果）：

1. `Input`：输入 mask
2. `Prediction`：模型输出
3. `Ground Truth`：真实标签（若数据集有 `aerials`）
4. `|Pred-GT|`：绝对误差图（颜色越亮误差越大）

第二行（解释图）：

1. `Grad-CAM`：基于梯度的粗粒度关注区域
2. `Guided Gradient`：输入空间细节敏感图（边缘纹理更细）
3. `Guided Grad-CAM`：将细节图与 Grad-CAM 融合后的结果
4. `Score-CAM`：无梯度 CAM，对梯度噪声更稳健

## 3. 各解释图的含义与注意事项

### 3.1 Grad-CAM

- 含义：显示“目标输出”主要依赖哪些高层特征位置。
- 特点：稳定、快、常用；但空间分辨率较粗。
- 常见用法：先用它判断模型到底关注了哪一块区域。

### 3.2 Guided Gradient

- 含义：输入像素对目标输出的敏感性细节（高频信息多）。
- 特点：纹理细、易受噪声影响，不宜单独当因果证据。
- 常见用法：用于观察边缘/细节敏感区域。

### 3.3 Guided Grad-CAM

- 含义：用 Grad-CAM 提供“区域”，用 Guided Gradient 提供“细节”。
- 特点：通常是最直观、最适合报告展示的一张图。
- 常见用法：论文/汇报主图优先候选。

### 3.4 Score-CAM

- 含义：通过遮罩激活图并前向打分获得权重，不依赖梯度。
- 特点：计算更慢，但常作为 Grad-CAM 的稳健性对照。
- 常见用法：与 Grad-CAM 并排看一致性，检查是否梯度伪影。

## 4. JSON 报告字段说明

关键字段：

- `target_layer`：用于 CAM 的层名
- `target_mode`：目标函数定义（如 `mean`/`pixel`/`gt_weighted`）
- `target_score`：该样本目标得分
- `mse_to_ground_truth`：预测与真值 MSE（若真值存在）
- `saliency_stats`：解释图统计（均值、相关性）
- `perturbation_top20`：简单扰动检验（top 20% 解释区域）

## 5. 扰动指标如何理解（`perturbation_top20`）

针对每种解释图，会做两种操作：

- `score_after_remove_topk`：把最重要 20% 区域抹掉后得分
- `score_after_keep_topk`：只保留最重要 20% 区域后的得分

推荐关注：

- `drop_remove_topk = base_score - score_after_remove_topk`
  - 越大通常说明解释图抓到的区域越关键。
- `ratio_keep_topk = score_after_keep_topk / base_score`
  - 越高表示少量区域已保留较多目标信息。

注意：这只是轻量 sanity check，不是严格因果证明。

## 6. 使用建议（研究实践）

- 单样本：先看 `panel`，再看对应 JSON。
- 多样本：用批量模式，优先看 `batch_explain_summary.json` 的均值趋势。
  - 若启用 `--batch-grid-png`，可快速浏览多个样本解释图整体分布。
- 层选择：默认自动层可用；做研究时建议固定 1-2 个层名重复实验，保证可比性。
- 目标函数：回归任务常用 `mean`；局部行为分析用 `pixel`；与真值关联可用 `gt_weighted`。

## 7. 解释图的边界

- 解释图反映的是模型内部相关性，不等同于物理因果。
- 不同方法之间有差异很正常，建议做多方法交叉验证。
- 当模型输出很平或接近常数时，解释图会退化（低对比/低相关）。

## 8. 批量模式命令示例

```bash
python explain_cnn_visualization.py \
  --model litho_model_780_e200.keras \
  --dataset litho_dataset_780.npz \
  --sample-idx 0 \
  --batch-count 12 \
  --batch-grid-png \
  --batch-grid-cols 4 \
  --output-dir explain_outputs_batch
```
