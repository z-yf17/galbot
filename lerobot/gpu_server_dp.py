#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZMQ GPU server for Diffusion policy (LeRobot-compatible).

- 完全复用 lerobot 的：
  - LeRobotDataset (meta.stats)
  - make_policy / make_pre_post_processors
  - predict_action
- 只复用你原来的通信格式（兼容原来的 client）：
  - REQ: {"i": int, "joint": [7], "grip": float}
  - REP: {
        "ok": bool,
        "did_infer": bool,
        "bucket_size": int,
        "latency_ms": float,
        "i": int,
        "joints_cmd": [7] or None,
        "grip_cmd": float or None
    }
"""

from __future__ import annotations

import time
import threading
from collections import deque
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import cv2
import zmq

# ===== LeRobot imports =====
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device


# ===================== 配置：按需修改 =====================

# 训练时使用的数据集（和 lerobot-train 一致）
DATASET_REPO_ID = "galbot/pick_place_simple_lerobot"
DATASET_ROOT = (
    "/home/galbot/test_galbot/galbot/Tele/scripts/dataset/"
    "pick_place_simple/my_lerobot_pick_place_simple"
)

# 训练好的 ACT 模型目录（包含 config.json / model.safetensors / train_config.json）
POLICY_DIR = "/home/galbot/zyf_test/lerobot/outputs/train/diffusion_pick_place_test/checkpoints/last/pretrained_model"

# 机械臂关节数（Franka = 7）
N_JOINTS = 7
HAS_GRIPPER = True

# ZMQ 绑定地址（和 client 中的 SERVER_ADDR 对应）
BIND_ADDR = "tcp://0.0.0.0:5555"

# 控制主循环频率（不要太高，100 Hz 就够，client 那边控制 200 Hz）
CONTROL_HZ = 100
DT = 1.0 / CONTROL_HZ

# 动作平滑窗口长度（越大越平滑）
SMOOTH_WINDOW = 5

# 摄像头参数（按你实际训练时用的来改）
CAMERA_DEV = "/dev/video0"
CAMERA_SIZE = (640, 480)   # (width, height)
CAMERA_FPS = 30

# 是否显示调试画面（你当前环境是 headless，先关掉）
SHOW = False
WIN_NAME = "LeRobot ACT Inference"
_LAST_SHOW_FAILED = False

# =========================================================

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


class CameraReader:
    """简单的异步摄像头读取器，保持最新一帧。"""

    def __init__(
        self,
        dev: str = CAMERA_DEV,
        api: int = cv2.CAP_V4L2,
        fourcc: str = "MJPG",
        size: Tuple[int, int] = CAMERA_SIZE,
        fps: int = CAMERA_FPS,
    ) -> None:
        self.dev, self.api = dev, api
        self.fourcc, self.size, self.fps = fourcc, size, fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._stop = False
        self._t: Optional[threading.Thread] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_ts: float = 0.0

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.dev, self.api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera device: {self.dev}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self.latest_frame = frame
                self.latest_ts = time.time()

    def get_latest(self, copy_frame: bool = True) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            f = self.latest_frame
            ts = self.latest_ts
        if f is None:
            return None, 0.0
        return (f.copy() if copy_frame else f), ts

    def stop(self) -> None:
        self._stop = True
        if self._t is not None:
            self._t.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


def auto_detect_state_image_keys(dataset: LeRobotDataset) -> Tuple[str, str]:
    """
    从 dataset.features 里自动找一个 state key 和 image key：

    - state_key:
        - 必须以 "observation." 开头
        - 名字里包含 "state" / "qpos" / "joint" 之一
    - image_key:
        - 必须以 "observation." 开头
        - 名字里包含 "image"
    """
    if not hasattr(dataset, "features"):
        raise RuntimeError(
            f"LeRobotDataset 没有 features 属性，当前属性有: {dir(dataset)}"
        )

    feat_keys = list(dataset.features.keys())

    # 找候选 state keys
    state_candidates = [
        k
        for k in feat_keys
        if k.startswith("observation.")
        and any(s in k.lower() for s in ["state", "qpos", "joint"])
    ]

    # 找候选 image keys
    image_candidates = [
        k
        for k in feat_keys
        if k.startswith("observation.") and "image" in k.lower()
    ]

    state_key = state_candidates[0] if state_candidates else None
    image_key = image_candidates[0] if image_candidates else None

    if state_key is None:
        raise RuntimeError(
            "自动没有找到 state key，请手动指定。\n"
            f"当前 dataset.features 的 keys 为：\n{feat_keys}"
        )
    if image_key is None:
        raise RuntimeError(
            "自动没有找到 image key，请手动指定。\n"
            f"当前 dataset.features 的 keys 为：\n{feat_keys}"
        )

    print(f"[server] auto-detected STATE_KEY = {state_key}")
    print(f"[server] auto-detected IMAGE_KEY = {image_key}")
    return state_key, image_key


def get_dataset_image_shape(dataset: LeRobotDataset, image_key: str) -> Tuple[int, int]:
    """
    从数据集中读一帧，拿到 observation.image 的 (H, W)。

    兼容两种情况：
    - HWC: (H, W, 3)
    - CHW: (3, H, W)
    """
    sample = dataset[0]
    img = sample[image_key]
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):  # CHW
            h, w = arr.shape[1], arr.shape[2]
        else:  # HWC
            h, w = arr.shape[0], arr.shape[1]
    elif arr.ndim == 2:
        h, w = arr.shape[0], arr.shape[1]
    else:
        # 兜底策略：随便给个值，后面 resize 会用
        h, w = 180, 320
    print(f"[server] dataset {image_key} shape ~ (H={h}, W={w})")
    return int(h), int(w)


def preprocess_image_for_observation(
    frame: np.ndarray,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """
    这里实现和你原来类似的图像处理逻辑：

    1. 如果图像是左右拼接（比如 2560x720），取右半部分；
    2. 然后 resize 到数据集图像的尺寸 (H, W)；
    3. 保持 BGR 顺序，交给 lerobot 的 preprocessor 做后续归一化/转换。
    """
    h, w, _ = frame.shape
    target_h, target_w = target_hw

    img = frame

    # 1) 判断是否是左右双目 / 两视角拼接：宽度明显远大于高度 => 取右半边
    # 你之前的代码是 2560x720 -> 取右半边 1280x720
    if w >= 2 * h:
        img = img[:, w // 2 :, :]

    # 2) resize 到数据集图像尺寸
    if img.shape[0] != target_h or img.shape[1] != target_w:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return img


def extract_joint_and_grip_from_action(
    action_values: Any,
    n_joints: int = N_JOINTS,
    has_gripper: bool = HAS_GRIPPER,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    从 predict_action 返回的 action_values 里解析出 joints(7) + grip(1)。

    - action_values 通常是 dict 或 torch.Tensor / np.ndarray。
    - 我们假设最终有一维向量，长度 >= n_joints (+1 if has_gripper)。
    """
    arr = None

    if isinstance(action_values, dict):
        if "action" in action_values:
            arr = action_values["action"]
        else:
            # 找第一个 tensor/ndarray 当主 action
            for v in action_values.values():
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    arr = v
                    break
    elif isinstance(action_values, torch.Tensor):
        arr = action_values
    elif isinstance(action_values, np.ndarray):
        arr = action_values
    else:
        return None, None

    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)

    # 去掉 batch / 时间维
    if arr.ndim == 3:   # (B, T, D)
        arr = arr[0]
    if arr.ndim == 2:   # (T, D)
        arr = arr[0]
    elif arr.ndim > 2:
        arr = arr.reshape(-1)

    if arr.ndim != 1:
        return None, None

    min_len = n_joints + (1 if has_gripper else 0)
    if arr.size < min_len:
        return None, None

    joints = arr[:n_joints].astype(np.float32)
    grip = float(arr[n_joints]) if has_gripper else None
    return joints, grip


def maybe_show(img_bgr: Optional[np.ndarray], overlay_lines: list[str]) -> None:
    """
    非阻塞可视化：在图像上叠加文本，用于调试。
    - 按 'q' 或 ESC 可以退出整个 server（抛 KeyboardInterrupt）
    - 如果环境不支持 GUI，会在第一次失败后自动关闭 SHOW，避免影响推理
    """
    global SHOW, _LAST_SHOW_FAILED

    if not SHOW or img_bgr is None:
        return

    try:
        disp = img_bgr.copy()
        y = 20
        for line in overlay_lines:
            cv2.putText(
                disp,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            y += 22

        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WIN_NAME, disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or k == 27:  # 'q' or ESC
            raise KeyboardInterrupt

    except KeyboardInterrupt:
        # 往上抛，让 main 的 try/except 捕获，正常退出
        raise
    except Exception as e:
        if not _LAST_SHOW_FAILED:
            print(f"[viz] imshow failed: {e}. Disable SHOW to keep service running.")
            _LAST_SHOW_FAILED = True
        SHOW = False


def main() -> None:
    # ===== 1. 加载数据集，用于拿 meta.stats & feature 名 =====
    print(
        f"[server] loading LeRobotDataset: "
        f"repo_id={DATASET_REPO_ID}, root={DATASET_ROOT}"
    )
    dataset = LeRobotDataset(DATASET_REPO_ID, root=DATASET_ROOT)

    # 自动找出 state/image 的 key（必要时你可以修改为写死）
    state_key, image_key = auto_detect_state_image_keys(dataset)

    # 根据数据集里真实存储的 observation.image 形状，确定目标 (H, W)
    target_h, target_w = get_dataset_image_shape(dataset, image_key)

    # ===== 2. 加载 policy config & policy, 构造 pre/post processor =====
    print(f"[server] loading PreTrainedConfig from {POLICY_DIR}")
    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_DIR, cli_overrides={})
    policy_cfg.device = DEVICE_STR
    policy_cfg.pretrained_path = POLICY_DIR

    print("[server] building policy via make_policy(...)")
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)

    print("[server] building pre/post processors via make_pre_post_processors(...)")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=POLICY_DIR,
        dataset_stats=rename_stats(dataset.meta.stats, rename_map={}),
        preprocessor_overrides={
            "device_processor": {"device": DEVICE_STR},
            # 如有 rename_map，可在此填入：
            # "rename_observations_processor": {"rename_map": {...}},
        },
    )

    policy.to(DEVICE)
    policy.eval()
    use_amp = getattr(policy.config, "use_amp", False)
    print(f"[server] policy ready on {DEVICE_STR}, use_amp={use_amp}")

    device_torch = get_safe_torch_device(DEVICE_STR)

    # ===== 3. 启动摄像头 =====
    cam = CameraReader()
    cam.start()
    print(f"[server] camera started on {CAMERA_DEV}")

    # ===== 4. 启动 ZMQ REP server =====
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(BIND_ADDR)
    print(f"[server] ZMQ REP server on {BIND_ADDR}")

    # 平滑 buffer
    joint_buffer: deque[np.ndarray] = deque(maxlen=SMOOTH_WINDOW)
    last_grip_cmd: float = 0.0
    i_step: int = -1

    last_latency_ms: float = 0.0
    last_did_infer: bool = False

    try:
        while True:
            t0 = time.perf_counter()

            # ---- 收 request ----
            try:
                req = sock.recv_json()
            except Exception as e:
                sock.send_json({"ok": False, "error": f"bad json: {e}"})
                continue

            if "joint" not in req or "grip" not in req:
                sock.send_json(
                    {"ok": False, "error": "missing keys: joint/grip"}
                )
                continue

            if not isinstance(req["joint"], (list, tuple)) or len(req["joint"]) != N_JOINTS:
                sock.send_json(
                    {
                        "ok": False,
                        "error": f"joint must be list len {N_JOINTS}",
                    }
                )
                continue

            i_step = int(req.get("i", i_step + 1))
            joint_state = np.asarray(req["joint"], dtype=np.float32)
            grip_width = float(req.get("grip", 0.0))

            frame, _ts = cam.get_latest(copy_frame=True)

            did_infer = False
            joints_cmd = None
            grip_cmd = None

            vis_img = None

            if frame is not None:
                # ===== 5. 图像预处理：裁剪 + resize 成和数据集一致的视野 =====
                proc_img = preprocess_image_for_observation(
                    frame, target_hw=(target_h, target_w)
                )
                vis_img = proc_img  # 用处理后的图像做可视化会更直观

                # ===== 5. 构造 observation_frame，键名和 dataset.features 对齐 =====
                obs_frame: Dict[str, Any] = {}

                # 5.1 state：拼 joint + grip，保持和训练数据同类型（np.ndarray）
                if HAS_GRIPPER:
                    state_vec = np.concatenate(
                        [joint_state, [grip_width]], axis=0
                    ).astype(np.float32)
                else:
                    state_vec = joint_state.astype(np.float32)

                obs_frame[state_key] = state_vec

                # 5.2 image：使用裁剪后的图像
                obs_frame[image_key] = proc_img

                try:
                    # ===== 6. 调用 lerobot 的 predict_action（归一化+反归一化都在里面）=====
                    action_values = predict_action(
                        observation=obs_frame,
                        policy=policy,
                        device=device_torch,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=use_amp,
                        task=None,
                        robot_type=dataset.meta.info.get("robot_type", ""),
                    )
                    

                    raw_joints_cmd, raw_grip_cmd = extract_joint_and_grip_from_action(
                        action_values,
                        n_joints=N_JOINTS,
                        has_gripper=HAS_GRIPPER,
                    )


                    if raw_joints_cmd is not None:
                        joint_buffer.append(raw_joints_cmd)
                        joints_cmd = np.mean(
                            np.stack(joint_buffer, axis=0), axis=0
                        ).astype(np.float32)
                        did_infer = True

                    if raw_grip_cmd is not None:
                        grip_cmd = float(raw_grip_cmd)
                        last_grip_cmd = grip_cmd
                    else:
                        grip_cmd = last_grip_cmd

                except KeyError as e:
                    print(f"[server] inference KeyError: {e}")
                except Exception as e:
                    print(f"[server] inference error: {e}")

            # ===== 7. 回包 =====
            latency_ms = (time.perf_counter() - t0) * 1000.0
            last_latency_ms = float(latency_ms)
            last_did_infer = bool(did_infer)

            resp = {
                "ok": True,
                "did_infer": bool(did_infer),
                "bucket_size": len(joint_buffer),
                "latency_ms": float(latency_ms),
                "i": int(i_step),
                "joints_cmd": joints_cmd.tolist() if joints_cmd is not None else None,
                "grip_cmd": float(grip_cmd) if grip_cmd is not None else None,
            }
            sock.send_json(resp)

            # ===== 8. 可视化当前图像（如果 SHOW=True 且环境支持）=====
            if vis_img is not None:
                overlay = [
                    f"step={i_step}  did_infer={last_did_infer}  "
                    f"buffer={len(joint_buffer)}",
                    f"latency={last_latency_ms:.1f} ms",
                    "Press 'q' or ESC to quit",
                ]
                maybe_show(vis_img, overlay)

            # 控制循环频率
            loop_elapsed = time.perf_counter() - t0
            if loop_elapsed < DT:
                time.sleep(DT - loop_elapsed)

    except KeyboardInterrupt:
        print("\n[server] Ctrl-C received. Stopping...")
    finally:
        try:
            cam.stop()
        except Exception:
            pass
        try:
            sock.close(0)
        except Exception:
            pass
        try:
            if SHOW:
                cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()

