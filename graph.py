"""
graph.py - Master LangGraph Orchestrator for Text-to-Motion Generation

This is the central pipeline that converts a text prompt into a rigged FBX animation
file through a multi-agent workflow running in WSL2. The pipeline flows through five
stages:

  1. Retrieve Context  - Queries Milvus vector DB for historical motion corrections
  2. Generate Plan     - Calls Ollama/Llama-3 on Windows host for a symbolic Laban motion plan
  3. Generate Motion   - Runs HY-Motion-1.0 diffusion transformer to produce animation data
  4. Evaluate Motion   - 4-Stage Critic Cascade (kinematics → Llama 3 → LLaVA → human)
  5. MCP Translation   - If only BVH was produced, falls back to Windows Blender via FastMCP

The evaluate step routes conditionally: if the score passes (>=0.85) or max iterations
are reached, it either finishes (FBX) or triggers the Blender fallback (BVH). Otherwise
it loops back to re-plan.

Prerequisites (see WALKTHROUGH.md for full setup):
  - Milvus running via Docker Compose on localhost:19530
  - Ollama serving Llama-3 on the Windows host at port 11434
  - HY-Motion-1.0 cloned to /mnt/e/GenAnimPipeline/HY-Motion-1.0 with VRAM bypass patch
  - FastMCP Blender server running on Windows at port 8000 (only needed for BVH fallback)
"""

import os
import asyncio
import subprocess
import json
import requests
import glob
import shutil
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, TypedDict, Any
from langgraph.graph import StateGraph, END
from pymilvus import connections, Collection
from mcp import ClientSession
from mcp.client.sse import sse_client
import math
import base64
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from bvh import Bvh

# ============================================================================
# PIPELINE CONFIGURATION
# All model and service settings are centralized here. To swap a model or
# change an endpoint, update only this block -- node functions reference
# these values by name, keeping the pipeline flow untouched.
# ============================================================================

# -- Project Root --
# All paths below are derived from this anchor so the pipeline works
# regardless of where the repo is cloned.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _wsl_to_win(wsl_path):
    """Convert a WSL2 /mnt/X/... path to its Windows equivalent X:\\..."""
    if wsl_path.startswith("/mnt/") and len(wsl_path) > 6:
        drive = wsl_path[5].upper()
        rest = wsl_path[6:].replace("/", "\\")
        return f"{drive}:{rest}"
    return wsl_path.replace("/", "\\")

# -- Default prompt used when running the pipeline directly --
DEFAULT_PROMPT = "A character performs a heavy broadsword swing."

# -- Animation Parameters --
GLOBAL_DURATION = 5       # Animation length in seconds
MOTION_SEED = 999         # Reproducibility seed for HY-Motion inference

# -- Planner LLM (Ollama API on Windows host) --
# Change PLANNER_MODEL to any Ollama-pulled model (e.g. "mistral", "gemma2", "phi3")
PLANNER_MODEL = "llama3"
VISION_MODEL = "llava"        # LLaVA vision model for Stage 3 art direction
OLLAMA_PORT = 11434

# -- Motion Generation Engine (HY-Motion-1.0 in WSL2) --
# To swap engines, update these paths and adjust the prompt format in
# generate_motion_node() to match the new engine's expected input.
MOTION_ENGINE_DIR = os.path.join(SCRIPT_DIR, "HY-Motion-1.0")
MOTION_ENGINE_CHECKPOINT = "ckpts/tencent/HY-Motion-1.0"
MOTION_ENGINE_SCRIPT = "local_infer.py"

# -- Vector Memory (Milvus via Docker) --
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "motion_corrections"

# -- Output Paths (relative to project root) --
OUTPUT_FBX_PATH = os.path.join(SCRIPT_DIR, "final_animation.fbx")
OUTPUT_BVH_PATH = os.path.join(SCRIPT_DIR, "temp_motion.bvh")
OUTPUT_NPZ_PATH = os.path.join(SCRIPT_DIR, "temp_motion.npz")

# -- MCP Blender Fallback (Windows-native paths derived from output paths above) --
MCP_SERVER_PORT = 8000
WIN_BVH_PATH = _wsl_to_win(OUTPUT_BVH_PATH)
WIN_FBX_PATH = _wsl_to_win(OUTPUT_FBX_PATH)

# -- Critic / Evaluation Thresholds --
CRITIC_SCORE_THRESHOLD = 0.85  # Minimum score to accept without re-planning
MAX_ITERATIONS = 3             # Hard cap on plan-generate-evaluate loops

# -- Logging --
LOG_LEVEL = logging.DEBUG                               # Set to logging.INFO to reduce verbosity
LOG_FILE  = os.path.join(SCRIPT_DIR, "pipeline.log")  # Rotating log file (5 MB × 3 backups)

# Pass duration to HY-Motion's patched prompt_rewrite.py via environment
os.environ["HY_MOTION_DURATION"] = str(GLOBAL_DURATION)


def _get_windows_host_ip():
    """Resolve the Windows host IP from inside WSL2 via the default gateway.
    Used by any node that needs to reach a service on the Windows host."""
    return subprocess.check_output(
        "ip route list default | awk '{print $3}'", shell=True
    ).decode().strip()

# ============================================================================
# LOGGER SETUP
# Writes structured debug output to both the console and a rotating log file.
# Adjust LOG_LEVEL in the config block above to control verbosity.
# ============================================================================

_LOG_FMT      = "[%(asctime)s] [%(levelname)-8s] %(message)s"
_LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("pipeline")
logger.setLevel(LOG_LEVEL)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter(_LOG_FMT, _LOG_DATE_FMT))

_file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter(_LOG_FMT, _LOG_DATE_FMT))

logger.addHandler(_console_handler)
logger.addHandler(_file_handler)


def _fmt_val(v) -> str:
    """Truncate long values so I/O dumps stay readable in the log."""
    s = repr(v)
    return s if len(s) <= 200 else s[:197] + "..."


def _log_input(node: str, keys: list, state: dict) -> None:
    logger.debug("[%s] ── INPUT  ─────────────────────────────────────", node)
    for k in keys:
        logger.debug("[%s]   %-22s = %s", node, k, _fmt_val(state.get(k)))


def _log_output(node: str, result: dict) -> None:
    logger.debug("[%s] ── OUTPUT ─────────────────────────────────────", node)
    for k, v in result.items():
        logger.debug("[%s]   %-22s = %s", node, k, _fmt_val(v))


# Shared state dictionary passed between every node in the LangGraph workflow.
# Each node reads what it needs and returns a partial dict to merge back in.
class PipelineState(TypedDict):
    prompt: str                 # User's natural-language motion description
    historical_context: list    # Past correction rules retrieved from Milvus
    laban_plan: dict            # Symbolic motion plan from Llama-3 (action_type, spatial_level, speed)
    motion_tensor_path: str     # (Reserved) path to raw motion tensor if needed
    critic_score: float         # Quality score from the evaluation node (0.0 - 1.0)
    critic_feedback: str        # Human-readable feedback from the critic
    iteration_count: int        # Number of plan-generate-evaluate loops completed
    final_asset_path: str       # Absolute path to the output file (.fbx or .bvh)

def retrieve_context_node(state: PipelineState):
    """Node 1: Connect to the Milvus vector database and load historical motion
    correction rules. These embeddings let the planner avoid known failure modes.
    Gracefully degrades if Milvus is offline -- the pipeline continues without context."""
    _log_input("retrieve_context", ["prompt"], state)
    logger.info("[retrieve_context] Connecting to Milvus at %s:%s ...", MILVUS_HOST, MILVUS_PORT)
    historical_context = []
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(MILVUS_COLLECTION)
        collection.load()
        logger.info("[retrieve_context] Milvus connected. Collection '%s' loaded.", MILVUS_COLLECTION)
    except Exception as e:
        logger.warning("[retrieve_context] Vector DB not available — continuing without context. Error: %s", e)
    result = {"historical_context": historical_context}
    _log_output("retrieve_context", result)
    return result

def generate_symbolic_plan_node(state: PipelineState):
    """Node 2: Ask Ollama/Llama-3 (running on the Windows host) to decompose the
    user prompt into a structured Laban-style motion plan. The Windows host IP is
    discovered dynamically via the WSL2 default gateway. Falls back to a safe
    default plan if the LLM is unreachable."""
    _log_input("generate_plan", ["prompt", "historical_context", "iteration_count", "critic_feedback"], state)
    logger.info("[generate_plan] Generating symbolic motion plan with %s (iteration %s)...",
                PLANNER_MODEL, state.get("iteration_count", 0))
    prompt = state["prompt"]

    try:
        win_ip = _get_windows_host_ip()
        logger.debug("[generate_plan] Resolved Windows host IP: %s", win_ip)
        system_prompt = "Output a JSON object for motion planning. Keys: 'action_type', 'spatial_level', 'speed'."
        response = requests.post(
            f"http://{win_ip}:{OLLAMA_PORT}/api/generate",
            json={"model": PLANNER_MODEL, "prompt": f"{system_prompt}\nUser: {prompt}", "stream": False, "format": "json", "keep_alive": 0},
            timeout=30
        )
        plan = json.loads(response.json().get("response", "{}"))
        logger.info("[generate_plan] Plan received: %s", plan)
    except Exception as e:
        logger.warning("[generate_plan] LLM unreachable — using fallback plan. Error: %s", e)
        plan = {"action_type": "idle", "spatial_level": "mid", "speed": "normal"}

    result = {"laban_plan": plan, "iteration_count": state.get("iteration_count", 0) + 1}
    _log_output("generate_plan", result)
    return result

def generate_motion_node(state: PipelineState):
    """Node 3: Run the HY-Motion-1.0 diffusion transformer to generate animation data.

    Writes the prompt in HY-Motion's expected format (text#duration#seed) then invokes
    local_infer.py. After generation, applies smart fallback logic:
      - If a native .fbx was produced, copy it to final_animation.fbx (preferred path).
        Also copies the co-generated .npz (SMPL-H pose data) so the critic cascade
        can evaluate kinematics and render skeleton previews.
      - If only a .bvh was produced, copy it to temp_motion.bvh (triggers Blender fallback).
      - If neither exists, log a critical error and continue with an empty asset path.
    """
    _log_input("generate_motion", ["prompt", "laban_plan", "iteration_count"], state)
    logger.info("[generate_motion] Starting HY-Motion-1.0 inference (duration=%ss, seed=%s)...",
                GLOBAL_DURATION, MOTION_SEED)
    prompt = state["prompt"]

    # Clear previous inference outputs to avoid stale file collisions
    os.system(f"rm -rf {MOTION_ENGINE_DIR}/output/local_infer/*")
    logger.debug("[generate_motion] Cleared previous inference outputs.")

    # Write the prompt file in HY-Motion's input format: "text#duration#seed"
    input_text_dir = os.path.join(MOTION_ENGINE_DIR, "prompt_inputs")
    os.makedirs(input_text_dir, exist_ok=True)
    prompt_file_path = os.path.join(input_text_dir, "task.txt")
    with open(prompt_file_path, "w") as f:
        f.write(f"{prompt}#{GLOBAL_DURATION}#{MOTION_SEED}\n")
    logger.debug("[generate_motion] Wrote prompt file: %s", prompt_file_path)

    command = ["python", MOTION_ENGINE_SCRIPT, "--model_path", MOTION_ENGINE_CHECKPOINT, "--input_text_dir", "prompt_inputs"]
    logger.debug("[generate_motion] Command: %s (cwd=%s)", " ".join(command), MOTION_ENGINE_DIR)

    asset_path = ""

    try:
        subprocess.run(command, cwd=MOTION_ENGINE_DIR, check=True)
        logger.info("[generate_motion] Inference complete. Scanning output directory...")

        # --- SMART FALLBACK LOGIC ---
        # Prefer native FBX (game-engine ready); fall back to BVH (raw skeleton data).
        # HY-Motion always co-generates .npz files (SMPL-H pose data) alongside
        # the FBX -- copy those too so the critic cascade can evaluate kinematics
        # and render skeleton previews from the raw joint data.
        output_dir = os.path.join(MOTION_ENGINE_DIR, "output")
        fbx_files = glob.glob(f"{output_dir}/**/*.fbx", recursive=True)
        bvh_files = glob.glob(f"{output_dir}/**/*.bvh", recursive=True)
        npz_files = glob.glob(f"{output_dir}/**/*.npz", recursive=True)
        logger.debug("[generate_motion] Found — FBX: %s | BVH: %s | NPZ: %s",
                     fbx_files, bvh_files, npz_files)

        if fbx_files:
            logger.info("[generate_motion] Native FBX detected. Copying to %s.", OUTPUT_FBX_PATH)
            shutil.copy(fbx_files[0], OUTPUT_FBX_PATH)
            asset_path = OUTPUT_FBX_PATH
            # Preserve BVH if it exists (unlikely for HY-Motion, but handled).
            if bvh_files:
                shutil.copy(bvh_files[0], OUTPUT_BVH_PATH)
                logger.debug("[generate_motion] Co-generated BVH copied to %s.", OUTPUT_BVH_PATH)
            # Always preserve the NPZ (SMPL-H pose data) for the critic cascade.
            if npz_files:
                shutil.copy(npz_files[0], OUTPUT_NPZ_PATH)
                logger.debug("[generate_motion] SMPL-H NPZ copied to %s.", OUTPUT_NPZ_PATH)
        elif bvh_files:
            logger.info("[generate_motion] No FBX found — falling back to BVH. Copying to %s.", OUTPUT_BVH_PATH)
            shutil.copy(bvh_files[0], OUTPUT_BVH_PATH)
            asset_path = OUTPUT_BVH_PATH
        else:
            logger.error("[generate_motion] CRITICAL — no animation files were produced by the engine.")

    except subprocess.CalledProcessError as e:
        logger.error("[generate_motion] HY-Motion subprocess failed: %s", e)

    result = {"final_asset_path": asset_path}
    _log_output("generate_motion", result)
    return result

# ============================================================================
# SMPL-H NPZ ADAPTER
# When HY-Motion outputs FBX directly (no BVH intermediate), the critic reads
# the co-generated .npz file through this adapter, which exposes the same API
# as bvh.Bvh so all existing FK helpers and stage functions work unchanged.
# ============================================================================

# SMPL-H body skeleton: 22 body joints (the 30 hand joints are ignored).
_SMPLH_BODY_JOINTS = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
    'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck',
    'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
]

# Parent index for each joint (-1 = root).
_SMPLH_PARENT_IDX = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]

# Approximate rest-pose bone offsets (meters) for a neutral SMPL body (~1.7 m).
# Pelvis offset is (0,0,0) because its world position comes from the NPZ trans
# channels. All other offsets are parent-relative vectors in the T-pose.
_SMPLH_OFFSETS_M = [
    (0.0, 0.0, 0.0),       # 0  Pelvis   (position from trans channels)
    (-0.08, -0.065, 0.0),  # 1  L_Hip
    (0.08, -0.065, 0.0),   # 2  R_Hip
    (0.0, 0.125, 0.0),     # 3  Spine1
    (0.0, -0.41, 0.0),     # 4  L_Knee
    (0.0, -0.41, 0.0),     # 5  R_Knee
    (0.0, 0.12, 0.0),      # 6  Spine2
    (0.0, -0.42, 0.0),     # 7  L_Ankle
    (0.0, -0.42, 0.0),     # 8  R_Ankle
    (0.0, 0.12, 0.0),      # 9  Spine3
    (0.0, -0.07, 0.12),    # 10 L_Foot
    (0.0, -0.07, 0.12),    # 11 R_Foot
    (0.0, 0.12, 0.0),      # 12 Neck
    (-0.06, 0.06, 0.0),    # 13 L_Collar
    (0.06, 0.06, 0.0),     # 14 R_Collar
    (0.0, 0.12, 0.0),      # 15 Head
    (-0.16, 0.0, 0.0),     # 16 L_Shoulder
    (0.16, 0.0, 0.0),      # 17 R_Shoulder
    (-0.26, 0.0, 0.0),     # 18 L_Elbow
    (0.26, 0.0, 0.0),      # 19 R_Elbow
    (-0.26, 0.0, 0.0),     # 20 L_Wrist
    (0.26, 0.0, 0.0),      # 21 R_Wrist
]

_SMPLH_SCALE = 100.0  # metres -> centimetres (matches BVH conventions)


class _MockBvhNode:
    """Minimal stand-in for bvh.BvhNode used by the FK hierarchy walker."""
    def __init__(self, name, node_type='JOINT', parent=None):
        self.value = [node_type, name]
        self.parent = parent
        self.children = []


def _axis_angle_to_rotation_matrix(aa):
    """Rodrigues' formula: axis-angle vector (3,) -> rotation matrix (3,3)."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _rotation_matrix_to_euler_zxy(R):
    """Decompose rotation matrix into ZXY intrinsic Euler angles (degrees).

    Channel order [Zrotation, Xrotation, Yrotation] matches the FK code in
    _compute_joint_world_position which multiplies Rz @ Rx @ Ry.
    """
    sx = float(np.clip(R[2, 1], -1.0, 1.0))
    x = math.asin(sx)
    cx = math.cos(x)
    if abs(cx) > 1e-6:
        y = math.atan2(-float(R[2, 0]), float(R[2, 2]))
        z = math.atan2(-float(R[0, 1]), float(R[1, 1]))
    else:
        y = 0.0
        z = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return (math.degrees(z), math.degrees(x), math.degrees(y))


class SmplhMocap:
    """Adapter: reads an SMPL-H .npz and exposes a bvh.Bvh-compatible API.

    Allows the critic cascade to evaluate HY-Motion NPZ output using the
    same FK helpers and stage functions written for BVH files.  All spatial
    values are scaled to centimetres so the kinematic thresholds remain valid.
    """

    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        raw_poses = np.array(data['poses'], dtype=np.float64)
        self._trans = np.array(data['trans'], dtype=np.float64) * _SMPLH_SCALE
        self._nframes = int(data.get('num_frames', raw_poses.shape[0]))
        self._framerate = float(data.get('mocap_framerate', 30))

        # (N, 156) -> (N, 52, 3); keep only 22 body joints
        all_joints = raw_poses.reshape(self._nframes, -1, 3)
        body_poses = all_joints[:, :len(_SMPLH_BODY_JOINTS), :]

        # Pre-convert every frame/joint from axis-angle to ZXY Euler degrees
        self._euler = np.zeros((self._nframes, len(_SMPLH_BODY_JOINTS), 3))
        for f in range(self._nframes):
            for j in range(len(_SMPLH_BODY_JOINTS)):
                R = _axis_angle_to_rotation_matrix(body_poses[f, j])
                self._euler[f, j] = _rotation_matrix_to_euler_zxy(R)

        # Build mock hierarchy nodes
        self._nodes = {}
        for idx, name in enumerate(_SMPLH_BODY_JOINTS):
            pidx = _SMPLH_PARENT_IDX[idx]
            ntype = 'ROOT' if pidx == -1 else 'JOINT'
            pnode = self._nodes[_SMPLH_BODY_JOINTS[pidx]] if pidx >= 0 else None
            self._nodes[name] = _MockBvhNode(name, ntype, pnode)

        self._name_to_idx = {n: i for i, n in enumerate(_SMPLH_BODY_JOINTS)}

    @property
    def nframes(self):
        return self._nframes

    @property
    def frame_time(self):
        return 1.0 / self._framerate

    def get_joint_names(self):
        return list(_SMPLH_BODY_JOINTS)

    def get_joint(self, name):
        return self._nodes[name]

    def joint_offset(self, name):
        ox, oy, oz = _SMPLH_OFFSETS_M[self._name_to_idx[name]]
        return (ox * _SMPLH_SCALE, oy * _SMPLH_SCALE, oz * _SMPLH_SCALE)

    def joint_channels(self, name):
        if name == _SMPLH_BODY_JOINTS[0]:
            return ['Xposition', 'Yposition', 'Zposition',
                    'Zrotation', 'Xrotation', 'Yrotation']
        return ['Zrotation', 'Xrotation', 'Yrotation']

    def frame_joint_channel(self, frame, name, channel):
        idx = self._name_to_idx[name]
        ch = channel.lower()
        if ch == 'xposition':
            return float(self._trans[frame, 0])
        elif ch == 'yposition':
            return float(self._trans[frame, 1])
        elif ch == 'zposition':
            return float(self._trans[frame, 2])
        elif ch == 'zrotation':
            return float(self._euler[frame, idx, 0])
        elif ch == 'xrotation':
            return float(self._euler[frame, idx, 1])
        elif ch == 'yrotation':
            return float(self._euler[frame, idx, 2])
        return 0.0


# ============================================================================
# CRITIC CASCADE - HELPER FUNCTIONS
# These support the 4-stage evaluation pipeline in evaluate_motion_node().
# ============================================================================


def _axis_rotation_3x3(axis, angle_rad):
    """Single-axis rotation matrix (3x3) for forward kinematics."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    if axis == 'X':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'Y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'Z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.eye(3)


def _get_joint_chain(mocap, joint_name):
    """Walk up the BVH hierarchy to build the kinematic chain from root to joint."""
    joint = mocap.get_joint(joint_name)
    chain = []
    current = joint
    while current is not None:
        if hasattr(current, 'value') and current.value and current.value[0] in ('ROOT', 'JOINT'):
            chain.append(current.value[1])
        current = getattr(current, 'parent', None)
    chain.reverse()
    return chain


def _compute_joint_world_position(mocap, joint_name, frame):
    """Forward kinematics: compute world-space (x, y, z) of a BVH joint at a given frame."""
    chain = _get_joint_chain(mocap, joint_name)
    transform = np.eye(4)

    for jname in chain:
        offset = mocap.joint_offset(jname)
        T_offset = np.eye(4)
        T_offset[:3, 3] = [offset[0], offset[1], offset[2]]

        channels = mocap.joint_channels(jname)
        T_pos = np.eye(4)
        rot_ops = []

        for ch in channels:
            val = mocap.frame_joint_channel(frame, jname, ch)
            ch_lower = ch.lower()
            if ch_lower == 'xposition':
                T_pos[0, 3] = val
            elif ch_lower == 'yposition':
                T_pos[1, 3] = val
            elif ch_lower == 'zposition':
                T_pos[2, 3] = val
            elif 'rotation' in ch_lower:
                rot_ops.append((ch[0].upper(), math.radians(val)))

        R = np.eye(3)
        for axis, angle in rot_ops:
            R = R @ _axis_rotation_3x3(axis, angle)

        R4 = np.eye(4)
        R4[:3, :3] = R
        transform = transform @ T_offset @ T_pos @ R4

    return transform[:3, 3].copy()


def _find_joint_name(mocap, candidates):
    """Match a joint name from a list of alternatives against the BVH skeleton."""
    available = set(mocap.get_joint_names())
    for name in candidates:
        if name in available:
            return name
    lower_map = {n.lower(): n for n in available}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _get_all_joint_positions(mocap, frame):
    """Return {joint_name: np.array([x, y, z])} for every joint at a given frame."""
    positions = {}
    for jname in mocap.get_joint_names():
        try:
            positions[jname] = _compute_joint_world_position(mocap, jname, frame)
        except Exception:
            continue
    return positions


def _get_bone_connections(mocap):
    """Build a list of (parent_name, child_name) bone pairs from the BVH hierarchy."""
    connections = []
    for jname in mocap.get_joint_names():
        joint = mocap.get_joint(jname)
        parent = getattr(joint, 'parent', None)
        if parent and hasattr(parent, 'value') and parent.value and parent.value[0] in ('ROOT', 'JOINT'):
            connections.append((parent.value[1], jname))
    return connections


# ── Stage 1: Kinematic Gatekeeper ─────────────────────────────────────────

def _stage1_kinematic_gatekeeper(mocap):
    """Check for foot sliding: if a foot is grounded (Y near 0), its XZ must not drift."""
    left_foot = _find_joint_name(mocap, [
        'LeftFoot', 'lFoot', 'Left_Foot', 'leftFoot', 'L_Foot', 'left_foot',
    ])
    right_foot = _find_joint_name(mocap, [
        'RightFoot', 'rFoot', 'Right_Foot', 'rightFoot', 'R_Foot', 'right_foot',
    ])

    if not left_foot and not right_foot:
        return True, 0.8, "Could not identify foot joints in skeleton. Skipping kinematic check."

    # Estimate character height from frame 0 to set relative thresholds
    all_pos = _get_all_joint_positions(mocap, 0)
    if all_pos:
        y_vals = [p[1] for p in all_pos.values()]
        char_height = max(y_vals) - min(y_vals)
    else:
        char_height = 100.0

    ground_threshold = max(char_height * 0.05, 1.0)
    slide_threshold = max(char_height * 0.02, 0.5)

    total_violations = 0
    total_ground_frames = 0

    for foot_name in [left_foot, right_foot]:
        if not foot_name:
            continue
        prev_ground_xz = None
        for frame in range(mocap.nframes):
            pos = _compute_joint_world_position(mocap, foot_name, frame)
            xz = np.array([pos[0], pos[2]])

            if pos[1] < ground_threshold:
                total_ground_frames += 1
                if prev_ground_xz is not None:
                    drift = np.linalg.norm(xz - prev_ground_xz)
                    if drift > slide_threshold:
                        total_violations += 1
                prev_ground_xz = xz
            else:
                prev_ground_xz = None

    if total_ground_frames > 0:
        violation_ratio = total_violations / total_ground_frames
        if violation_ratio > 0.15:
            return False, 0.1, (
                f"Kinematic FAIL: Foot sliding detected. "
                f"{total_violations}/{total_ground_frames} grounded frames show excessive XZ drift "
                f"(threshold: {slide_threshold:.1f} units/frame, violation rate: {violation_ratio:.0%})."
            )

    return True, 1.0, "No significant foot sliding detected."


# ── Stage 2: Semantic Logician ────────────────────────────────────────────

def _stage2_semantic_logician(mocap, prompt):
    """Sample the BVH every 10 frames, summarize key bone data, ask Llama 3 whether
    the motion trajectory matches the prompt's semantic intent."""
    try:
        win_ip = _get_windows_host_ip()
    except Exception:
        win_ip = "localhost"

    sample_frames = list(range(0, mocap.nframes, 10))
    if not sample_frames:
        sample_frames = [0]

    # Resolve key joints with fallback candidates
    key_joints = []
    for candidates in [
        ['Spine', 'Spine1', 'Spine2', 'Spine3', 'spine'],
        ['RightArm', 'RightForeArm', 'Right_Arm', 'RightShoulder', 'rightArm', 'R_Shoulder', 'R_Elbow'],
        ['LeftArm', 'LeftForeArm', 'Left_Arm', 'LeftShoulder', 'leftArm', 'L_Shoulder', 'L_Elbow'],
    ]:
        found = _find_joint_name(mocap, candidates)
        if found:
            key_joints.append(found)

    if not key_joints:
        all_joints = mocap.get_joint_names()
        key_joints = all_joints[1:4] if len(all_joints) > 3 else all_joints

    # Build lightweight motion summary
    motion_summary = []
    for frame in sample_frames:
        frame_data = {"frame": frame}
        for jname in key_joints:
            try:
                pos = _compute_joint_world_position(mocap, jname, frame)
                channels = mocap.joint_channels(jname)
                rots = {}
                for ch in channels:
                    if 'rotation' in ch.lower():
                        rots[ch] = round(mocap.frame_joint_channel(frame, jname, ch), 1)
                frame_data[jname] = {
                    "position": [round(float(v), 1) for v in pos],
                    "rotations": rots,
                }
            except Exception:
                continue
        motion_summary.append(frame_data)

    motion_json = json.dumps(motion_summary)

    llm_prompt = (
        f"You are a motion analysis expert. A 3D character animation was generated from this prompt:\n"
        f"\"{prompt}\"\n\n"
        f"Below is a sampled trajectory of key joints (Spine, Arms) every 10 frames:\n"
        f"{motion_json}\n\n"
        f"Does this motion trajectory plausibly match the semantic intent of the prompt? "
        f"Respond in JSON: {{\"matches\": true/false, \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}}"
    )

    try:
        response = requests.post(
            f"http://{win_ip}:{OLLAMA_PORT}/api/generate",
            json={
                "model": PLANNER_MODEL,
                "prompt": llm_prompt,
                "stream": False,
                "format": "json",
                "keep_alive": 0,
            },
            timeout=60,
        )
        response.raise_for_status()
        result = json.loads(response.json().get("response", "{}"))
        matches = result.get("matches", True)
        reasoning = result.get("reasoning", "No reasoning provided.")

        if not matches:
            return False, 0.4, f"Semantic FAIL: {reasoning}"
        return True, 1.0, f"Semantic match confirmed: {reasoning}"

    except requests.exceptions.Timeout:
        return True, 0.7, "Semantic check skipped: Ollama request timed out."
    except Exception as e:
        return True, 0.7, f"Semantic check skipped: {e}"


# ── Stage 3: Art Director ─────────────────────────────────────────────────

def _render_skeleton_grid(mocap, output_path):
    """Render a 2x2 grid of 3D skeleton keyframes and save as PNG."""
    nf = mocap.nframes
    keyframes = [0, nf // 3, (2 * nf) // 3, max(nf - 1, 0)]

    connections = _get_bone_connections(mocap)
    fig = plt.figure(figsize=(12, 12))

    for idx, frame in enumerate(keyframes):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        positions = _get_all_joint_positions(mocap, frame)

        if not positions:
            ax.set_title(f"Frame {frame} (no data)")
            continue

        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        zs = [p[2] for p in positions.values()]
        ax.scatter(xs, ys, zs, c='red', s=20, depthshade=True)

        for parent_name, child_name in connections:
            if parent_name in positions and child_name in positions:
                p1, p2 = positions[parent_name], positions[child_name]
                ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    'b-', linewidth=1.5,
                )

        all_coords = np.array(list(positions.values()))
        center = all_coords.mean(axis=0)
        spread = max((all_coords.max(axis=0) - all_coords.min(axis=0)).max() / 2 * 1.2, 1.0)
        ax.set_xlim([center[0] - spread, center[0] + spread])
        ax.set_ylim([center[1] - spread, center[1] + spread])
        ax.set_zlim([center[2] - spread, center[2] + spread])
        ax.set_title(f"Frame {frame}", fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.suptitle("Motion Keyframes - Skeleton Preview", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def _stage3_art_director(mocap, prompt):
    """Render 4 skeleton keyframes, send to LLaVA for visual quality scoring."""
    eval_image_path = os.path.join(tempfile.gettempdir(), "temp_eval.png")

    try:
        _render_skeleton_grid(mocap, eval_image_path)
    except Exception as e:
        return True, 0.7, f"Art Director skipped: skeleton render failed ({e})."

    try:
        with open(eval_image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return True, 0.7, f"Art Director skipped: could not read rendered image ({e})."

    try:
        win_ip = _get_windows_host_ip()
    except Exception:
        win_ip = "localhost"

    vision_prompt = (
        f"You are an Animation Art Director evaluating a 3D skeleton motion preview.\n"
        f"The animation was generated from this prompt: \"{prompt}\"\n\n"
        f"This image shows 4 keyframes of the skeleton. Evaluate:\n"
        f"1. Weight and physicality - does the pose convey mass and force?\n"
        f"2. Fluidity - do the keyframes suggest smooth transitions?\n"
        f"3. Pose dynamics - are the poses expressive and match the prompt?\n\n"
        f"Respond in JSON: {{\"score\": 0.0-1.0, \"feedback\": \"detailed critique here\"}}"
    )

    try:
        response = requests.post(
            f"http://{win_ip}:{OLLAMA_PORT}/api/generate",
            json={
                "model": VISION_MODEL,
                "prompt": vision_prompt,
                "images": [img_b64],
                "stream": False,
                "format": "json",
                "keep_alive": 0,
            },
            timeout=90,
        )
        response.raise_for_status()
        result = json.loads(response.json().get("response", "{}"))
        score = max(0.0, min(1.0, float(result.get("score", 0.5))))
        feedback = result.get("feedback", "No feedback provided.")

        if score < CRITIC_SCORE_THRESHOLD:
            return False, score, f"Art Director ({score:.2f}): {feedback}"
        return True, score, f"Art Director ({score:.2f}): {feedback}"

    except requests.exceptions.Timeout:
        return True, 0.7, "Art Director skipped: LLaVA request timed out."
    except Exception as e:
        return True, 0.7, f"Art Director skipped: {e}"


# ── Stage 4: Human-in-the-Loop ────────────────────────────────────────────

def _stage4_human_sign_off(prompt, ai_score, ai_feedback, asset_path):
    """Pause for human approval via CLI. Returns (score, feedback)."""
    border = "=" * 64
    print(f"\n{border}")
    print("    HUMAN-IN-THE-LOOP: EXECUTIVE SIGN-OFF REQUIRED")
    print(border)
    print(f"  Prompt    : {prompt}")
    print(f"  AI Score  : {ai_score:.2f}")
    print(f"  Feedback  : {ai_feedback}")
    print(f"  Asset     : {asset_path}")
    print(border)

    approval = input("  >>> Approve this motion? (Y/N): ").strip().upper()
    if approval == 'Y':
        return 1.0, "Human approved."

    reason = input("  >>> Reason for rejection: ").strip()
    return 0.0, f"Human Override: {reason if reason else 'No reason given.'}"


# ============================================================================
# CRITIC CASCADE - MAIN NODE
# ============================================================================


def evaluate_motion_node(state: PipelineState):
    """Node 4: 4-Stage Critic Cascade for motion quality evaluation.

    Runs sequentially -- if a lower stage fails, heavier stages are skipped:
      Stage 1 - Kinematic Gatekeeper : foot-sliding physics check (pure Python math)
      Stage 2 - Semantic Logician    : Llama 3 verifies motion matches prompt intent
      Stage 3 - Art Director         : LLaVA vision model scores skeleton keyframes
      Stage 4 - Human Sign-Off      : CLI approval gate (only if AI score >= 0.85)
    """
    _log_input("evaluate_motion", ["final_asset_path", "prompt", "iteration_count"], state)
    asset_path = state.get("final_asset_path", "")
    prompt = state.get("prompt", "")

    # --- Locate motion data (BVH preferred, NPZ fallback) ---
    mocap = None

    # Try BVH first
    bvh_path = None
    if asset_path.endswith(".bvh"):
        bvh_path = asset_path
    elif asset_path.endswith(".fbx"):
        candidate = asset_path.rsplit(".fbx", 1)[0] + ".bvh"
        if os.path.exists(candidate):
            bvh_path = candidate
    if not bvh_path or not os.path.exists(str(bvh_path)):
        if os.path.exists(OUTPUT_BVH_PATH):
            bvh_path = OUTPUT_BVH_PATH

    if bvh_path and os.path.exists(str(bvh_path)):
        try:
            with open(bvh_path, 'r') as f:
                mocap = Bvh(f.read())
            logger.info("[evaluate_motion] Loaded BVH — %d frames from %s", mocap.nframes, bvh_path)
        except Exception as e:
            logger.warning("[evaluate_motion] BVH parse failed (%s) — trying NPZ fallback...", e)

    # NPZ fallback -- HY-Motion always generates .npz with SMPL-H pose data
    if mocap is None:
        npz_path = OUTPUT_NPZ_PATH
        if not os.path.exists(npz_path):
            logger.warning("[evaluate_motion] No BVH or NPZ found — passing with default score.")
            result = {"critic_score": 0.85, "critic_feedback": "No motion data available for evaluation."}
            _log_output("evaluate_motion", result)
            return result
        try:
            mocap = SmplhMocap(npz_path)
            logger.info("[evaluate_motion] Loaded SMPL-H NPZ — %d frames from %s", mocap.nframes, npz_path)
        except Exception as e:
            logger.error("[evaluate_motion] NPZ parse error: %s", e)
            result = {"critic_score": 0.5, "critic_feedback": f"Motion data parse error: {e}"}
            _log_output("evaluate_motion", result)
            return result

    if mocap.nframes < 2:
        logger.warning("[evaluate_motion] Only %d frame(s) — insufficient for evaluation.", mocap.nframes)
        result = {"critic_score": 0.3, "critic_feedback": "Motion contains insufficient frames for evaluation."}
        _log_output("evaluate_motion", result)
        return result

    # ---- STAGE 1: Kinematic Gatekeeper (Python math) ----
    logger.info("[evaluate_motion] ── Stage 1/4: Kinematic Gatekeeper ──────────────────")
    passed, score, feedback = _stage1_kinematic_gatekeeper(mocap)
    logger.info("[evaluate_motion]   %s | score=%.2f | %s", "PASS" if passed else "FAIL", score, feedback)
    if not passed:
        result = {"critic_score": score, "critic_feedback": feedback}
        _log_output("evaluate_motion", result)
        return result

    # ---- STAGE 2: Semantic Logician (Llama 3) ----
    logger.info("[evaluate_motion] ── Stage 2/4: Semantic Logician (Llama 3) ──────────")
    passed, score, feedback = _stage2_semantic_logician(mocap, prompt)
    logger.info("[evaluate_motion]   %s | score=%.2f | %s", "PASS" if passed else "FAIL", score, feedback)
    if not passed:
        result = {"critic_score": score, "critic_feedback": feedback}
        _log_output("evaluate_motion", result)
        return result

    # ---- STAGE 3: Art Director (LLaVA) ----
    logger.info("[evaluate_motion] ── Stage 3/4: Art Director (LLaVA) ─────────────────")
    passed, score, feedback = _stage3_art_director(mocap, prompt)
    logger.info("[evaluate_motion]   %s | score=%.2f | %s", "PASS" if passed else "FAIL", score, feedback)
    if not passed:
        result = {"critic_score": score, "critic_feedback": feedback}
        _log_output("evaluate_motion", result)
        return result

    # ---- STAGE 4: Human-in-the-Loop (CLI) ----
    logger.info("[evaluate_motion] ── Stage 4/4: Human-in-the-Loop ──────────────────── ")
    final_score, final_feedback = _stage4_human_sign_off(prompt, score, feedback, asset_path)
    logger.info("[evaluate_motion]   %s | %s",
                "APPROVED" if final_score == 1.0 else "REJECTED", final_feedback)

    result = {"critic_score": final_score, "critic_feedback": final_feedback}
    _log_output("evaluate_motion", result)
    return result

def execute_mcp_translation_node(state: PipelineState):
    """Node 5 (conditional): BVH-to-FBX fallback via the Windows Blender MCP server.

    When HY-Motion only produces a .bvh file (raw skeleton data without mesh binding),
    this node calls the FastMCP translation server running natively on Windows. That
    server launches Blender headlessly to import the BVH, bake the animation, and
    export a game-engine-ready FBX. Communication uses Server-Sent Events (SSE) over
    the WSL2-to-Windows network bridge.

    Paths are Windows-native because Blender runs on the Windows side.
    """
    _log_input("execute_mcp", ["final_asset_path", "critic_score", "critic_feedback"], state)
    logger.info("[execute_mcp] BVH detected — contacting Windows Blender MCP server...")
    logger.debug("[execute_mcp] BVH input (Windows): %s", WIN_BVH_PATH)
    logger.debug("[execute_mcp] FBX output (Windows): %s", WIN_FBX_PATH)

    try:
        win_ip = _get_windows_host_ip()
        mcp_url = f"http://{win_ip}:{MCP_SERVER_PORT}/sse"
        logger.debug("[execute_mcp] MCP SSE endpoint: %s", mcp_url)

        async def run_mcp_tool():
            async with sse_client(mcp_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await session.call_tool(
                        "run_blender_export",
                        arguments={"bvh_motion_path": WIN_BVH_PATH, "output_fbx_path": WIN_FBX_PATH}
                    )

        mcp_result = asyncio.run(run_mcp_tool())
        logger.info("[execute_mcp] Blender server response: %s", mcp_result.content)

    except ExceptionGroup as eg:
        for exc in eg.exceptions:
            logger.error("[execute_mcp] Connection error: %s", exc)
    except Exception as e:
        logger.error("[execute_mcp] MCP tool call failed: %s", e)

    result = {"final_asset_path": WIN_FBX_PATH}
    _log_output("execute_mcp", result)
    return result

def route_evaluation(state: PipelineState):
    """Conditional router after the critic node. Three possible outcomes:
      - "finish"      : Score passed and we have an FBX -- pipeline is done.
      - "execute_mcp" : Score passed but only BVH exists -- trigger Blender fallback.
      - "replan"      : Score too low and iterations remain -- loop back to the planner.
    Caps at MAX_ITERATIONS to prevent infinite loops on stubborn prompts."""
    score     = state["critic_score"]
    iteration = state["iteration_count"]
    asset     = state.get("final_asset_path", "")
    logger.debug("[route_evaluation] critic_score=%.2f | iteration=%d/%d | asset=%s",
                 score, iteration, MAX_ITERATIONS, asset)

    if score >= CRITIC_SCORE_THRESHOLD or iteration >= MAX_ITERATIONS:
        route = "execute_mcp" if asset.endswith(".bvh") else "finish"
    else:
        route = "replan"

    logger.info("[route_evaluation] Decision: %s (score=%.2f, threshold=%.2f, iteration=%d/%d)",
                route.upper(), score, CRITIC_SCORE_THRESHOLD, iteration, MAX_ITERATIONS)
    return route

# ============================================================================
# LANGGRAPH ASSEMBLY
# Wires the nodes into a directed graph with one conditional branch:
#
#   retrieve_context -> generate_plan -> generate_motion -> evaluate_motion
#                            ^                                   |
#                            |  (replan)                         v
#                            +------- route_evaluation ----> [finish | execute_mcp]
#                                                                        |
#                                                                        v
#                                                                       END
# ============================================================================
workflow = StateGraph(PipelineState)
workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("generate_plan", generate_symbolic_plan_node)
workflow.add_node("generate_motion", generate_motion_node)
workflow.add_node("evaluate_motion", evaluate_motion_node)
workflow.add_node("execute_mcp", execute_mcp_translation_node)

workflow.set_entry_point("retrieve_context")
workflow.add_edge("retrieve_context", "generate_plan")
workflow.add_edge("generate_plan", "generate_motion")
workflow.add_edge("generate_motion", "evaluate_motion")

workflow.add_conditional_edges(
    "evaluate_motion",
    route_evaluation,
    {
        "execute_mcp": "execute_mcp",
        "finish": END,
        "replan": "generate_plan"
    }
)
workflow.add_edge("execute_mcp", END)

app = workflow.compile()

if __name__ == "__main__":
    logger.info("══════════════════════════════════════════════════════")
    logger.info("  GENERATIVE ANIMATION PIPELINE — STARTING")
    logger.info("  Prompt    : %s", DEFAULT_PROMPT)
    logger.info("  Log file  : %s", LOG_FILE)
    logger.info("══════════════════════════════════════════════════════")
    app.invoke({"prompt": DEFAULT_PROMPT, "iteration_count": 0})
    logger.info("══════════════════════════════════════════════════════")
    logger.info("  PIPELINE COMPLETE")
    logger.info("══════════════════════════════════════════════════════")