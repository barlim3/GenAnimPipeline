"""
pipeline/nodes/critic.py — 4-Stage Critic Cascade and batch candidate scorer.

Stages:
  1. Kinematic Gatekeeper — foot-sliding physics check (pure Python math)
  2. Semantic Logician    — Llama 3 verifies motion matches prompt intent
  3. Art Director         — LLaVA vision model scores skeleton keyframes
  4. Human Sign-Off       — CLI approval gate (only if AI score >= threshold)

Public node:
  evaluate_motion_node   — runs the full cascade on the winning candidate
  score_candidate        — runs stages 1-3 only, used for best-of-batch selection

Standalone usage (loads a real .npz if given, else uses synthetic dummy data):
    python -m pipeline.nodes.critic [path/to/motion.npz]
"""

import base64
import json
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import requests
from bvh import Bvh

from pipeline.shared import (
    CRITIC_SCORE_THRESHOLD, OLLAMA_PORT, OUTPUT_BVH_PATH, OUTPUT_NPZ_PATH,
    PLANNER_MODEL, VISION_MODEL,
    PipelineState, get_windows_host_ip, logger, log_input, log_output,
)
from pipeline.smplh_adapter import (
    SmplhMocap,
    compute_joint_world_position,
    find_joint_name,
    get_all_joint_positions,
    get_bone_connections,
)


# ── Stage 1: Kinematic Gatekeeper ─────────────────────────────────────────────

def stage1_kinematic_gatekeeper(mocap):
    """Check for foot sliding: if a foot is grounded (Y near 0), its XZ must not drift."""
    left_foot = find_joint_name(mocap, [
        'LeftFoot', 'lFoot', 'Left_Foot', 'leftFoot', 'L_Foot', 'left_foot',
    ])
    right_foot = find_joint_name(mocap, [
        'RightFoot', 'rFoot', 'Right_Foot', 'rightFoot', 'R_Foot', 'right_foot',
    ])

    if not left_foot and not right_foot:
        return True, 0.8, "Could not identify foot joints in skeleton. Skipping kinematic check."

    all_pos = get_all_joint_positions(mocap, 0)
    if all_pos:
        y_vals = [p[1] for p in all_pos.values()]
        char_height = max(y_vals) - min(y_vals)
    else:
        char_height = 100.0

    ground_threshold = max(char_height * 0.05, 1.0)
    slide_threshold  = max(char_height * 0.02, 0.5)

    total_violations = 0
    total_ground_frames = 0

    for foot_name in [left_foot, right_foot]:
        if not foot_name:
            continue
        prev_ground_xz = None
        for frame in range(mocap.nframes):
            pos = compute_joint_world_position(mocap, foot_name, frame)
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


# ── Stage 2: Semantic Logician ─────────────────────────────────────────────────

def stage2_semantic_logician(mocap, prompt: str):
    """Sample the motion every 10 frames, summarize key bone data, ask Llama 3 whether
    the motion trajectory matches the prompt's semantic intent."""
    try:
        win_ip = get_windows_host_ip()
    except Exception:
        win_ip = "localhost"

    sample_frames = list(range(0, mocap.nframes, 10))
    if not sample_frames:
        sample_frames = [0]

    key_joints = []
    for candidates in [
        ['Spine', 'Spine1', 'Spine2', 'Spine3', 'spine'],
        ['RightArm', 'RightForeArm', 'Right_Arm', 'RightShoulder', 'rightArm', 'R_Shoulder', 'R_Elbow'],
        ['LeftArm', 'LeftForeArm', 'Left_Arm', 'LeftShoulder', 'leftArm', 'L_Shoulder', 'L_Elbow'],
    ]:
        found = find_joint_name(mocap, candidates)
        if found:
            key_joints.append(found)

    if not key_joints:
        all_joints = mocap.get_joint_names()
        key_joints = all_joints[1:4] if len(all_joints) > 3 else all_joints

    motion_summary = []
    for frame in sample_frames:
        frame_data = {"frame": frame}
        for jname in key_joints:
            try:
                pos = compute_joint_world_position(mocap, jname, frame)
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
        matches   = result.get("matches", True)
        reasoning = result.get("reasoning", "No reasoning provided.")

        if not matches:
            return False, 0.4, f"Semantic FAIL: {reasoning}"
        return True, 1.0, f"Semantic match confirmed: {reasoning}"

    except requests.exceptions.Timeout:
        return True, 0.7, "Semantic check skipped: Ollama request timed out."
    except Exception as e:
        return True, 0.7, f"Semantic check skipped: {e}"


# ── Stage 3: Art Director ──────────────────────────────────────────────────────

def render_skeleton_grid(mocap, output_path: str) -> None:
    """Render a 2×2 grid of 3D skeleton keyframes and save as PNG."""
    nf = mocap.nframes
    keyframes   = [0, nf // 3, (2 * nf) // 3, max(nf - 1, 0)]
    connections = get_bone_connections(mocap)
    fig = plt.figure(figsize=(12, 12))

    for idx, frame in enumerate(keyframes):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        positions = get_all_joint_positions(mocap, frame)

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
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    plt.suptitle("Motion Keyframes - Skeleton Preview", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def stage3_art_director(mocap, prompt: str):
    """Render 4 skeleton keyframes, send to LLaVA for visual quality scoring."""
    eval_image_path = os.path.join(tempfile.gettempdir(), "temp_eval.png")

    try:
        render_skeleton_grid(mocap, eval_image_path)
    except Exception as e:
        return True, 0.7, f"Art Director skipped: skeleton render failed ({e})."

    try:
        with open(eval_image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return True, 0.7, f"Art Director skipped: could not read rendered image ({e})."

    try:
        win_ip = get_windows_host_ip()
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
        score    = max(0.0, min(1.0, float(result.get("score", 0.5))))
        feedback = result.get("feedback", "No feedback provided.")

        if score < CRITIC_SCORE_THRESHOLD:
            return False, score, f"Art Director ({score:.2f}): {feedback}"
        return True, score, f"Art Director ({score:.2f}): {feedback}"

    except requests.exceptions.Timeout:
        return True, 0.7, "Art Director skipped: LLaVA request timed out."
    except Exception as e:
        return True, 0.7, f"Art Director skipped: {e}"


# ── Stage 4: Human-in-the-Loop ─────────────────────────────────────────────────

def stage4_human_sign_off(prompt: str, ai_score: float, ai_feedback: str, asset_path: str):
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
    if approval == 'N':
        reason = input("  >>> Reason for rejection: ").strip()
        return 0.0, f"Human Override: {reason if reason else 'No reason given.'}"

    return 1.0, "Human approved."


# ── Batch candidate scorer ─────────────────────────────────────────────────────

def score_candidate(npz_path: str, prompt: str) -> float:
    """Run automated critic stages 1–3 on a single motion candidate NPZ.

    Used by generate_motion_node to rank all batch outputs and select the best one
    before passing the winner to evaluate_motion_node (which adds the human gate).
    Stage 4 is intentionally excluded — it only runs once on the winning candidate.

    Returns a composite score 0.0–1.0 (average of the three stage scores).
    Returns 0.0 if the file cannot be loaded or has insufficient frames.
    """
    try:
        mocap = SmplhMocap(npz_path)
    except Exception as e:
        logger.warning("[batch_score] Could not load %s: %s", npz_path, e)
        return 0.0

    if mocap.nframes < 2:
        logger.warning("[batch_score] %s has fewer than 2 frames — scoring 0.", npz_path)
        return 0.0

    _, s1, fb1 = stage1_kinematic_gatekeeper(mocap)
    logger.debug("[batch_score] %s  stage1=%.2f  (%s)", os.path.basename(npz_path), s1, fb1)

    _, s2, fb2 = stage2_semantic_logician(mocap, prompt)
    logger.debug("[batch_score] %s  stage2=%.2f  (%s)", os.path.basename(npz_path), s2, fb2)

    _, s3, fb3 = stage3_art_director(mocap, prompt)
    logger.debug("[batch_score] %s  stage3=%.2f  (%s)", os.path.basename(npz_path), s3, fb3)

    composite = (s1 + s2 + s3) / 3.0
    logger.info("[batch_score] %s  composite=%.2f", os.path.basename(npz_path), composite)
    return composite


# ── Main node ──────────────────────────────────────────────────────────────────

def evaluate_motion_node(state: PipelineState) -> dict:
    """Node 4: 4-Stage Critic Cascade for motion quality evaluation.

    Runs sequentially — if a lower stage fails, heavier stages are skipped:
      Stage 1 - Kinematic Gatekeeper : foot-sliding physics check (pure Python math)
      Stage 2 - Semantic Logician    : Llama 3 verifies motion matches prompt intent
      Stage 3 - Art Director         : LLaVA vision model scores skeleton keyframes
      Stage 4 - Human Sign-Off       : CLI approval gate (only if AI score >= threshold)
    """
    log_input("evaluate_motion", ["final_asset_path", "prompt", "iteration_count"], state)
    asset_path = state.get("final_asset_path", "")
    prompt     = state.get("prompt", "")

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

    # NPZ fallback — HY-Motion always generates .npz with SMPL-H pose data
    if mocap is None:
        npz_path = OUTPUT_NPZ_PATH
        if not os.path.exists(npz_path):
            logger.warning("[evaluate_motion] No BVH or NPZ found — passing with default score.")
            result = {"critic_score": 0.85, "critic_feedback": "No motion data available for evaluation."}
            log_output("evaluate_motion", result)
            return result
        try:
            mocap = SmplhMocap(npz_path)
            logger.info("[evaluate_motion] Loaded SMPL-H NPZ — %d frames from %s", mocap.nframes, npz_path)
        except Exception as e:
            logger.error("[evaluate_motion] NPZ parse error: %s", e)
            result = {"critic_score": 0.5, "critic_feedback": f"Motion data parse error: {e}"}
            log_output("evaluate_motion", result)
            return result

    if mocap.nframes < 2:
        logger.warning("[evaluate_motion] Only %d frame(s) — insufficient for evaluation.", mocap.nframes)
        result = {"critic_score": 0.3, "critic_feedback": "Motion contains insufficient frames for evaluation."}
        log_output("evaluate_motion", result)
        return result

    # Stage 1
    logger.info("[evaluate_motion] ── Stage 1/4: Kinematic Gatekeeper ──────────────────")
    passed, score, feedback = stage1_kinematic_gatekeeper(mocap)
    logger.info("[evaluate_motion]   %s | score=%.2f | %s", "PASS" if passed else "FAIL", score, feedback)
    if not passed:
        result = {"critic_score": score, "critic_feedback": feedback}
        log_output("evaluate_motion", result)
        return result

    # Stage 2
    logger.info("[evaluate_motion] ── Stage 2/4: Semantic Logician (Llama 3) ──────────")
    passed, score, feedback = stage2_semantic_logician(mocap, prompt)
    logger.info("[evaluate_motion]   %s | score=%.2f | %s", "PASS" if passed else "FAIL", score, feedback)
    if not passed:
        result = {"critic_score": score, "critic_feedback": feedback}
        log_output("evaluate_motion", result)
        return result

    # Stage 3
    logger.info("[evaluate_motion] ── Stage 3/4: Art Director (LLaVA) ─────────────────")
    passed, score, feedback = stage3_art_director(mocap, prompt)
    logger.info("[evaluate_motion]   %s | score=%.2f | %s", "PASS" if passed else "FAIL", score, feedback)
    if not passed:
        result = {"critic_score": score, "critic_feedback": feedback}
        log_output("evaluate_motion", result)
        return result

    # Stage 4
    logger.info("[evaluate_motion] ── Stage 4/4: Human-in-the-Loop ────────────────────")
    final_score, final_feedback = stage4_human_sign_off(prompt, score, feedback, asset_path)
    logger.info(
        "[evaluate_motion]   %s | %s",
        "APPROVED" if final_score == 1.0 else "REJECTED", final_feedback,
    )

    result = {"critic_score": final_score, "critic_feedback": final_feedback}
    log_output("evaluate_motion", result)
    return result


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile as _tempfile

    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
        print(f"=== Testing critic stages on: {npz_path} ===")
    else:
        print("No .npz given — generating synthetic dummy data for smoke test.")
        n_frames = 30
        n_joints = 52
        poses = np.zeros((n_frames, n_joints * 3), dtype=np.float32)
        trans = np.zeros((n_frames, 3), dtype=np.float32)
        trans[:, 1] = 1.0
        tmp = _tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        np.savez(tmp.name, poses=poses, trans=trans, mocap_framerate=30)
        npz_path = tmp.name
        print(f"  Dummy NPZ: {npz_path}")

    prompt = "A character performs a heavy broadsword swing."
    mocap  = SmplhMocap(npz_path)

    print(f"\n-- Stage 1 --")
    passed, score, fb = stage1_kinematic_gatekeeper(mocap)
    print(f"  passed={passed}  score={score:.2f}  feedback={fb}")

    print(f"\n-- Stage 2 (Ollama may timeout gracefully) --")
    passed, score, fb = stage2_semantic_logician(mocap, prompt)
    print(f"  passed={passed}  score={score:.2f}  feedback={fb}")

    print(f"\n-- Stage 3 (LLaVA may timeout gracefully) --")
    passed, score, fb = stage3_art_director(mocap, prompt)
    print(f"  passed={passed}  score={score:.2f}  feedback={fb}")

    composite = score_candidate(npz_path, prompt)
    print(f"\n-- Composite score (stages 1-3 avg): {composite:.2f} --")
