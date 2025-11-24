#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import zmq
from polymetis import RobotInterface, GripperInterface

# ===== 与机器人通信（保持你原始地址）=====
ARM_IP, ARM_PORT = "127.0.0.1", 50051
GRIP_IP, GRIP_PORT = "127.0.0.1", 50052

# ===== 控制频率 =====
CONTROL_HZ = 200
DT = 1.0 / CONTROL_HZ

# ===== 策略请求频率（仅限速策略，不影响控制环）=====
POLICY_HZ = 20            # 按需改成 15/20/30 等
POLICY_DT = 1.0 / POLICY_HZ

# Franka Panda 关节限幅（弧度）
JOINT_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) - 0.2
JOINT_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]) + 0.2

# 夹爪参数
GRIP_OPEN_WIDTH   = 0.078
GRIP_OPEN_SPEED   = 0.2
GRIP_OPEN_FORCE   = 0.1
GRIP_CLOSE_SPEED  = 0.2
GRIP_CLOSE_FORCE  = 0.1
GRIP_THRESH       = 0.001   # > 0.001 视为张开

# ===== 推理服务地址（固定为本机）=====
SERVER_ADDR = "tcp://127.0.0.1:5555"

def clip_joints(q: np.ndarray) -> np.ndarray:
    return np.clip(q, JOINT_MIN, JOINT_MAX).astype(np.float32)

def main():
    # === 连接机器人 ===
    robot = RobotInterface(ip_address=ARM_IP, port=ARM_PORT)
    gripper = GripperInterface(ip_address=GRIP_IP, port=GRIP_PORT)

    # 读取一次当前状态，作为“首帧期望值”
    state0 = robot.get_robot_state()
    q0 = torch.tensor(state0.joint_positions, dtype=torch.float32)

    # 启动关节阻抗控制器，并立刻 latch 一次（server 未回也先保活）
    robot.start_joint_impedance(adaptive=False)
    robot.update_desired_joint_positions(q0)
    last_cmd = q0.clone()

    # 夹爪初始状态（用于防抖）
    try:
        gstate0 = gripper.get_state()
        grip_width0 = float(getattr(gstate0, "width", 0.0))
    except Exception:
        grip_width0 = 0.0
    last_grip_open = None  # None 表示未知

    # === ZeroMQ REQ 客户端（严格配对的状态机）===
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 5)   # ms 级短超时
    sock.setsockopt(zmq.SNDTIMEO, 5)
    sock.connect(SERVER_ADDR)
    print(f"[client] connected to inference server {SERVER_ADDR}")

    awaiting_reply = False
    pending_req = None  # 保存尚未配对的请求（避免重复构造）

    i = -1
    t_next = time.perf_counter()         # 控制环相位
    t_next_policy = t_next               # 策略请求相位（单独限速）

    try:
        while True:
            i += 1
            t_next += DT
            loop_t0 = time.time_ns()

            # ---- 读取当前状态 ----
            state = robot.get_robot_state()
            joint_state = np.array(state.joint_positions, dtype=np.float32)
            try:
                gstate = gripper.get_state()
                grip_width = float(getattr(gstate, "width", 0.0))
            except Exception:
                grip_width = grip_width0

            # ---- 若到达策略请求时刻且不在等回包，则发送新请求（限速策略频率）----
            now = time.perf_counter()
            if (not awaiting_reply) and (now >= t_next_policy):
                pending_req = {"i": i, "joint": joint_state.tolist(), "grip": grip_width}
                try:
                    sock.send_json(pending_req, flags=0)
                    awaiting_reply = True
                    t_next_policy += POLICY_DT
                except Exception:
                    # 发送失败：下轮重试
                    awaiting_reply = False
                    # 不调整 t_next_policy，保持节拍

            # ---- 若在等回包，尝试非阻塞接收（保证 REQ/REP 不乱序）----
            did_infer = False
            bucket_size = 0
            infer_ms = -1.0
            joints_cmd = None
            grip_cmd   = None

            if awaiting_reply:
                try:
                    resp = sock.recv_json(flags=zmq.NOBLOCK)
                    awaiting_reply = False  # 完成一对
                    pending_req = None
                    # 兼容两种 server 的键名
                    infer_ms   = float(resp.get("latency_ms", -1.0))
                    did_infer  = bool(resp.get("did_infer", False))
                    bucket_size = int(resp.get("bucket_size", 0))
                    joints_cmd = resp.get("joints_cmd", None)
                    grip_cmd   = resp.get("grip_cmd", None)
                except zmq.Again:
                    # 暂未有回包：保持 awaiting_reply=True，本轮不再 send
                    pass
                except Exception:
                    # 接收异常：丢弃这次配对，防卡死
                    awaiting_reply = False
                    pending_req = None

            # ---- 生成/选择要下发的关节指令（None 时也要保活）----
            if joints_cmd is not None:
                joints_cmd = clip_joints(np.asarray(joints_cmd, dtype=np.float32))
                last_cmd = torch.from_numpy(joints_cmd).float()

            # 每帧必喂，保证控制器保活（保持 200 Hz）
            try:
                robot.update_desired_joint_positions(last_cmd)
            except Exception:
                # 自愈：重启控制器并立即保活一次
                try:
                    robot.start_joint_impedance(adaptive=False)
                    robot.update_desired_joint_positions(last_cmd)
                except Exception:
                    pass


            # ---- 夹爪（防抖，仅状态变化才下发）----
            if grip_cmd is not None:
                try:
                    want_open = bool(float(grip_cmd) > GRIP_THRESH)
                    if last_grip_open is None or want_open != last_grip_open:
                        if want_open:
                            gripper.goto(GRIP_OPEN_WIDTH, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE, blocking=False)
                        else:
                            gripper.grasp(GRIP_CLOSE_SPEED, GRIP_CLOSE_FORCE, blocking=False)
                        last_grip_open = want_open
                except Exception:
                    pass

            # ---- 精准定频（200 Hz）；不再覆盖 sleep 为常数 ----
            sleep_t = t_next - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                # 落后则重置相位，避免连锁欠采样
                t_next = time.perf_counter()

            loop_s = (time.time_ns() - loop_t0) / 1e9
            '''
            print(
                f"[client] step={i:05d} infer_ms={infer_ms:.2f} loop_s={loop_s:.6f} "
                f"did_infer={did_infer} bucket={bucket_size} awaiting={awaiting_reply}"
            )
            '''

    except KeyboardInterrupt:
        print("\n[client] Ctrl-C received. Stopping...")
    finally:
        try:
            sock.close(0)
        except Exception:
            pass

if __name__ == "__main__":
    main()
