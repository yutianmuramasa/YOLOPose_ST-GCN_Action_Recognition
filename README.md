# 基于 YOLOPose + ST-GCN 的骨骼动作识别

四人小组实习项目：用摄像头实时做**骨骼关键点驱动的动作识别** —— YOLOv8n-pose 提取人体 17 关键点 → 收集 60 帧关键点序列 → 训练轻量时空分类网络 → 实时识别（背景 / 喝水 / 举手 / 张臂）。

> **本人分工**：数据采集与数据集构建 —— 开发 `record_pose.py`（骨骼关键点实时提取与序列录制）、编写 `dataset.py`（PyTorch 数据管道）。模型结构与训练脚本由组员负责。

## 流程

```
摄像头 → YOLOv8n-pose 关键点检测 → 60 帧关键点序列缓冲 → 轻量时空分类网络 → 动作类别
                ↑                                            ↑
          record_pose.py 录制数据集                     train_stgcn.py 训练
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `record_pose.py`（本人） | 打开摄像头，按 `S` 录制 60 帧骨骼序列（保存 `.npy`），按类别存到 `dataset2/<类别>/`；录制不同动作时修改 `ACTION_NAME` |
| `dataset.py`（本人） | PyTorch Dataset：按文件夹名做类别索引，加载 `.npy` 骨骼序列 [N, 3, T, 17]（通道 = x/y 坐标 + 置信度，T = 60 帧，17 个 COCO 关键点） |
| `models/stgcn_model.py` | 轻量化时空分类网络：BatchNorm 归一化 + 1×1 卷积逐通道特征提取 + 全局池化做时序聚合（简化版 ST-GCN，避免复杂图邻接矩阵计算，适合轻量实时部署） |
| `train_stgcn.py` | 训练脚本（8:2 划分训练/验证，交叉熵 + 类别权重缓解样本不平衡） |
| `main.py` | 实时推理：YOLOv8n-pose + 分类网络，画面实时输出动作类别 |

类别：`background` / `drink` / `raisehand` / `zhangbi`（索引顺序与文件夹字母序一致，改动需同步 `main.py` 的 `ACTION_LABELS`）。

## 结果

在小组合力自建的动作数据集上，通过**多帧输入 + 时序聚合**的改进（CNN 结构消融：单帧基线 → 多帧 → 多帧+时序聚合），验证集准确率由 **79.2% 提升至 91.6%**（详见实习报告）。系统在机房 GPU/CPU 上均能流畅实时识别。

## 运行

```bash
# 依赖
pip install ultralytics torch opencv-python numpy

# 1. 录制自己的动作数据（每类多录几段）
python record_pose.py        # 记得改 ACTION_NAME

# 2. 训练
python train_stgcn.py

# 3. 实时识别
python main.py
```

说明：
- `yolov8n-pose.pt` 首次运行由 ultralytics 自动下载
- 数据集（`dataset2/`）为自己录制的 `.npy` 序列，已在 `.gitignore` 中排除，不随仓库分发
- 实习报告 docx / 汇报 pptx 含小组个人信息，同样不入库
