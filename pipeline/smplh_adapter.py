"""
pipeline/smplh_adapter.py — SMPL-H NPZ adapter and Forward Kinematics helpers.

When HY-Motion outputs FBX directly (no BVH intermediate), the critic cascade
reads the co-generated .npz file through SmplhMocap, which exposes the same
API as bvh.Bvh so all FK helpers and stage functions work unchanged.

Can be run standalone to validate a real .npz file:
    python -m pipeline.smplh_adapter [path/to/file.npz]
Or with a synthetic dummy to verify the math:
    python -m pipeline.smplh_adapter
"""

import math
import os

import numpy as np

from pipeline.shared import logger

# ── SMPL-H skeleton definition ────────────────────────────────────────────────

# 22 body joints (the 30 hand joints are ignored for critic purposes).
SMPLH_BODY_JOINTS = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
    'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck',
    'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
]

# Parent index for each joint (-1 = root).
SMPLH_PARENT_IDX = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]

# Approximate rest-pose bone offsets (metres) for a neutral SMPL body (~1.7 m).
# Pelvis offset is (0,0,0) — its world position comes from the NPZ trans channels.
SMPLH_OFFSETS_M = [
    (0.0, 0.0, 0.0),       # 0  Pelvis
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

SMPLH_SCALE = 100.0  # metres → centimetres (matches BVH conventions)


# ── Internal hierarchy node ───────────────────────────────────────────────────

class _MockBvhNode:
    """Minimal stand-in for bvh.BvhNode used by the FK hierarchy walker."""
    def __init__(self, name: str, node_type: str = 'JOINT', parent=None):
        self.value = [node_type, name]
        self.parent = parent
        self.children = []


# ── Rotation math ─────────────────────────────────────────────────────────────

def axis_angle_to_rotation_matrix(aa: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: axis-angle vector (3,) → rotation matrix (3×3)."""
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


def rotation_matrix_to_euler_zxy(R: np.ndarray):
    """Decompose rotation matrix into ZXY intrinsic Euler angles (degrees).

    Channel order [Zrotation, Xrotation, Yrotation] matches the FK code in
    compute_joint_world_position which multiplies Rz @ Rx @ Ry.
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


# ── SmplhMocap adapter ────────────────────────────────────────────────────────

class SmplhMocap:
    """Adapter: reads an SMPL-H .npz and exposes a bvh.Bvh-compatible API.

    Allows the critic cascade to evaluate HY-Motion NPZ output using the same
    FK helpers and stage functions written for BVH files. All spatial values
    are scaled to centimetres so the kinematic thresholds remain valid.

    Supported bvh.Bvh attributes/methods used by the critic:
        .nframes
        .frame_time
        .get_joint_names()
        .get_joint(name)
        .joint_offset(name)
        .joint_channels(name)
        .frame_joint_channel(frame, name, channel)
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        raw_poses = np.array(data['poses'], dtype=np.float64)
        self._trans = np.array(data['trans'], dtype=np.float64) * SMPLH_SCALE
        self._nframes = int(data.get('num_frames', raw_poses.shape[0]))
        self._framerate = float(data.get('mocap_framerate', 30))

        # (N, 156) → (N, 52, 3); keep only 22 body joints
        all_joints = raw_poses.reshape(self._nframes, -1, 3)
        body_poses = all_joints[:, :len(SMPLH_BODY_JOINTS), :]

        # Pre-convert every frame/joint from axis-angle to ZXY Euler degrees
        self._euler = np.zeros((self._nframes, len(SMPLH_BODY_JOINTS), 3))
        for f in range(self._nframes):
            for j in range(len(SMPLH_BODY_JOINTS)):
                R = axis_angle_to_rotation_matrix(body_poses[f, j])
                self._euler[f, j] = rotation_matrix_to_euler_zxy(R)

        # Build mock hierarchy nodes
        self._nodes = {}
        for idx, name in enumerate(SMPLH_BODY_JOINTS):
            pidx = SMPLH_PARENT_IDX[idx]
            ntype = 'ROOT' if pidx == -1 else 'JOINT'
            pnode = self._nodes[SMPLH_BODY_JOINTS[pidx]] if pidx >= 0 else None
            self._nodes[name] = _MockBvhNode(name, ntype, pnode)

        self._name_to_idx = {n: i for i, n in enumerate(SMPLH_BODY_JOINTS)}

    @property
    def nframes(self) -> int:
        return self._nframes

    @property
    def frame_time(self) -> float:
        return 1.0 / self._framerate

    def get_joint_names(self) -> list:
        return list(SMPLH_BODY_JOINTS)

    def get_joint(self, name: str) -> _MockBvhNode:
        return self._nodes[name]

    def joint_offset(self, name: str):
        ox, oy, oz = SMPLH_OFFSETS_M[self._name_to_idx[name]]
        return (ox * SMPLH_SCALE, oy * SMPLH_SCALE, oz * SMPLH_SCALE)

    def joint_channels(self, name: str) -> list:
        if name == SMPLH_BODY_JOINTS[0]:
            return ['Xposition', 'Yposition', 'Zposition',
                    'Zrotation', 'Xrotation', 'Yrotation']
        return ['Zrotation', 'Xrotation', 'Yrotation']

    def frame_joint_channel(self, frame: int, name: str, channel: str) -> float:
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


# ── FK helpers (used by critic stages) ───────────────────────────────────────

def axis_rotation_3x3(axis: str, angle_rad: float) -> np.ndarray:
    """Single-axis rotation matrix (3×3) for forward kinematics."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    if axis == 'X':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'Y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'Z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.eye(3)


def get_joint_chain(mocap, joint_name: str) -> list:
    """Walk up the hierarchy to build the kinematic chain from root to joint."""
    joint = mocap.get_joint(joint_name)
    chain = []
    current = joint
    while current is not None:
        if hasattr(current, 'value') and current.value and current.value[0] in ('ROOT', 'JOINT'):
            chain.append(current.value[1])
        current = getattr(current, 'parent', None)
    chain.reverse()
    return chain


def compute_joint_world_position(mocap, joint_name: str, frame: int) -> np.ndarray:
    """Forward kinematics: compute world-space (x, y, z) of a joint at a given frame."""
    chain = get_joint_chain(mocap, joint_name)
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
            R = R @ axis_rotation_3x3(axis, angle)

        R4 = np.eye(4)
        R4[:3, :3] = R
        transform = transform @ T_offset @ T_pos @ R4

    return transform[:3, 3].copy()


def find_joint_name(mocap, candidates: list):
    """Match a joint name from a list of alternatives against the skeleton."""
    available = set(mocap.get_joint_names())
    for name in candidates:
        if name in available:
            return name
    lower_map = {n.lower(): n for n in available}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def get_all_joint_positions(mocap, frame: int) -> dict:
    """Return {joint_name: np.array([x, y, z])} for every joint at a given frame."""
    positions = {}
    for jname in mocap.get_joint_names():
        try:
            positions[jname] = compute_joint_world_position(mocap, jname, frame)
        except Exception:
            continue
    return positions


def get_bone_connections(mocap) -> list:
    """Build a list of (parent_name, child_name) bone pairs from the hierarchy."""
    connections = []
    for jname in mocap.get_joint_names():
        joint = mocap.get_joint(jname)
        parent = getattr(joint, 'parent', None)
        if parent and hasattr(parent, 'value') and parent.value and parent.value[0] in ('ROOT', 'JOINT'):
            connections.append((parent.value[1], jname))
    return connections


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile

    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    else:
        # Generate a minimal synthetic NPZ for smoke-testing without real data.
        print("No .npz path given — using synthetic dummy data.")
        n_frames = 30
        n_joints = 52  # SMPL-H full body (22 body + 30 hands)
        poses = np.zeros((n_frames, n_joints * 3), dtype=np.float32)
        trans = np.zeros((n_frames, 3), dtype=np.float32)
        trans[:, 1] = 1.0  # keep pelvis 1 m above ground
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        np.savez(tmp.name, poses=poses, trans=trans, mocap_framerate=30)
        npz_path = tmp.name
        print(f"Dummy NPZ written to: {npz_path}")

    mocap = SmplhMocap(npz_path)
    print(f"Loaded: {npz_path}")
    print(f"  Frames     : {mocap.nframes}")
    print(f"  Frame time : {mocap.frame_time:.4f}s  ({1/mocap.frame_time:.1f} fps)")
    print(f"  Joints     : {mocap.get_joint_names()}")

    pos = compute_joint_world_position(mocap, "Pelvis", 0)
    print(f"  Pelvis pos @ frame 0 : {pos}")

    left_ankle = find_joint_name(mocap, ["L_Ankle", "LeftAnkle", "left_ankle"])
    if left_ankle:
        pos_ankle = compute_joint_world_position(mocap, left_ankle, 0)
        print(f"  {left_ankle} pos @ frame 0 : {pos_ankle}")
    else:
        print("  Could not resolve left ankle joint.")
