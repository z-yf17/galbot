# Using LeRobot Policies to Control a Franka Emika Research 3

This folder contains three scripts that connect **LeRobot-trained policies** to a  
**Franka Emika Research 3** arm (Franka Panda) via your existing communication stack.

At a high level:

- `gpu_server_act.py` and `gpu_server_dp.py` run LeRobot policies on a GPU machine and expose them as simple inference services.
- `inference.py` runs on the robot side and periodically queries one of these services for actions, then applies those actions to the real robot.

> **Important:** Always **start a GPU server first** (`gpu_server_act.py` or `gpu_server_dp.py`),  
> then run **`inference.py`** on the robot side.

---

## File Overview

### `gpu_server_act.py`

**Purpose**

Runs an inference service for a **LeRobot ACT policy** on a GPU machine.

**Key responsibilities**

- Load a LeRobot dataset and its metadata (for normalization, shapes, etc.).
- Load a pretrained ACT policy checkpoint (config + weights).
- Read images from a camera (or another video device).
- Combine visual observations and robot state into the correct input format for the ACT policy.
- Perform forward passes of the policy on the GPU.
- Return joint targets and a gripper command through a simple request–response interface.

**Configuration**

Edit the top of `gpu_server_act.py` to match your setup, for example:

```python
# Dataset used during training (for meta & stats)
DATASET_REPO_ID = "galbot/pick_place_simple_lerobot"
DATASET_ROOT = (
    "/home/galbot/test_galbot/galbot/Tele/scripts/dataset/"
    "pick_place_simple/my_lerobot_pick_place_simple"
)

# Trained ACT policy directory (contains config.json / model.safetensors / train_config.json)
POLICY_DIR = (
    "/home/galbot/zyf_test/lerobot/"
    "outputs/train/act_pick_place_test/checkpoints/last/pretrained_model"
)

# Number of joints for the robot (Franka = 7)
N_JOINTS = 7
HAS_GRIPPER = True

# Address to bind the inference server on
BIND_ADDR = "tcp://0.0.0.0:5555"

# Camera device and settings
CAMERA_DEV = "/dev/video0"
CAMERA_SIZE = (640, 480)  # (width, height)
CAMERA_FPS = 30
SHOW = False  # Set to True if you want to visualize the camera feed
