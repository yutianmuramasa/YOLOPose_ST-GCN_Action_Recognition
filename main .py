import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from models.stgcn_model import STGCN

# ======================
# 配置
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 60

# ⚠️⚠️⚠️ 关键：这里必须按文件夹名的字母顺序排列 ⚠️⚠️⚠️
# 如果你的文件夹是 'raisehand' 和 'zhangbi'
# 字母顺序 r 在 z 前面，所以 raisehand 是 0
ACTION_LABELS = [
    "background",
    "drink",
    "raisehand",
    "zhangbi"
]
NUM_CLASS = len(ACTION_LABELS)

print(f"动作识别马上开始......")

# ======================
# 模型加载
# ======================
pose_model = YOLO("yolov8n-pose.pt")
stgcn = STGCN(num_class=NUM_CLASS).to(DEVICE)

try:
    stgcn.load_state_dict(torch.load("weights/stgcn_best.pth", map_location=DEVICE, weights_only=True))
    stgcn.eval()
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# ======================
# 主循环
# ======================
cap = cv2.VideoCapture(0)
pose_buffer = deque(maxlen=SEQ_LEN)

while True:
    ret, frame = cap.read()
    if not ret: break

    results = pose_model(frame, verbose=False)

    current_kps = None

    if results[0].keypoints is not None:
        # 使用 xyn (归一化)
        kps = results[0].keypoints.xyn
        confs = results[0].keypoints.conf

        if len(kps) > 0:
            kp = kps[0].cpu().numpy()  # [17, 2]
            cf = confs[0].cpu().numpy()  # [17]

            # 关键修改：同时保存 (x, y, conf)
            # [17, 2] + [17, 1] -> [17, 3]
            frame_data = np.concatenate([kp, cf[:, None]], axis=1)
            pose_buffer.append(frame_data)

            current_kps = results[0].keypoints.xy[0].cpu().numpy()

    # 画图
    if current_kps is not None:
        for x, y in current_kps:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    # 识别
    action_text = "Buffering..."
    if len(pose_buffer) == SEQ_LEN:
        # [SEQ_LEN, 17, 3]
        data = np.stack(pose_buffer)

        # [T, V, C] -> [C, T, V]
        # 你的 dataset.py 是 transpose(1, 0, 2) -> [3, 60, 17]
        # 这里对应: transpose(2, 0, 1) -> [3, 60, 17]
        data = data.transpose(2, 0, 1)

        input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = stgcn(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

            label = ACTION_LABELS[pred_idx]
            action_text = f"{label} ({confidence:.2f})"

            # 打印概率条
            for i, p in enumerate(probs):
                name = ACTION_LABELS[i]
                cv2.putText(frame, f"{name}: {p:.2f}", (10, 80 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.putText(frame, action_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Action Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()