import os
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ======================
# 配置
# ======================
SEQ_LEN = 60
SAVE_ROOT = "dataset2"
ACTION_NAME = "background"  # <--- 记得录制不同动作时修改这里！
DEVICE_CAM = 0

# ======================
# 初始化
# ======================
os.makedirs(os.path.join(SAVE_ROOT, ACTION_NAME), exist_ok=True)

pose_model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(DEVICE_CAM)

pose_buffer = deque(maxlen=SEQ_LEN)
recording = False
sample_id = len(os.listdir(os.path.join(SAVE_ROOT, ACTION_NAME)))

print("🎥 摄像头已启动")
print(f"👉 当前录制类别: {ACTION_NAME}")
print("👉 按 S 开始录制 60 帧")
print("👉 按 Q 退出")

# ======================
# 主循环
# ======================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pose_model(frame, verbose=False)

    if results[0].keypoints is not None:
        # 1. 获取像素坐标用于画图 (Pixels)
        kps_pixel = results[0].keypoints.xy

        # 2. 获取归一化坐标用于保存 (Normalized)
        kps_norm = results[0].keypoints.xyn
        confs = results[0].keypoints.conf

        if len(kps_pixel) > 0:
            # 用于画图
            kp_draw = kps_pixel[0].cpu().numpy()

            # 用于保存
            kp_save = kps_norm[0].cpu().numpy()  # [17,2]
            cf = confs[0].cpu().numpy()  # [17]

            # 拼成 [3,17] 用于保存
            kp_full = np.vstack([kp_save.T, cf])

            if recording:
                pose_buffer.append(kp_full)

            # 画关键点 (使用像素坐标)
            for x, y in kp_draw:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            # 画骨架连线 (可选)
            # 这里可以加连线逻辑，但画点已经足够确认检测到了

    # 显示状态
    status = f"Recording {len(pose_buffer)}/{SEQ_LEN}" if recording else f"Ready: {ACTION_NAME} (Press S)"
    color = (0, 0, 255) if recording else (0, 255, 0)
    cv2.putText(frame, status, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("YOLO-Pose Auto Recorder", frame)

    key = cv2.waitKey(1) & 0xFF

    # 开始录制
    if key == ord('s') and not recording:
        pose_buffer.clear()
        recording = True
        print(f"⏺ 开始录制 {ACTION_NAME}...")

    # 录制完成
    if recording and len(pose_buffer) == SEQ_LEN:
        data = np.stack(pose_buffer)  # [60,3,17]
        data = data.transpose(1, 0, 2)  # [3,60,17]

        save_path = os.path.join(
            SAVE_ROOT,
            ACTION_NAME,
            f"{sample_id}.npy"
        )
        np.save(save_path, data)
        print(f"✅ 保存样本: {save_path}")

        recording = False
        sample_id += 1
        pose_buffer.clear()

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()