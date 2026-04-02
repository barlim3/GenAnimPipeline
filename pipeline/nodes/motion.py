"""
pipeline/nodes/motion.py — Motion generation node (HY-Motion-1.0 runner).

Node:
  generate_motion_node — Node 3: invoke HY-Motion-1.0, score batch candidates,
                         copy the best output to the configured output paths.

Standalone usage (dry-run — prints the command that would be executed, no inference):
    python -m pipeline.nodes.motion
    python -m pipeline.nodes.motion --dry-run      (same, explicit flag)
"""

import glob
import os
import shutil
import subprocess

from pipeline.nodes.critic import score_candidate
from pipeline.shared import (
    GLOBAL_DURATION, MOTION_BATCH_SIZE, MOTION_ENGINE_CHECKPOINT,
    MOTION_ENGINE_DIR, MOTION_ENGINE_SCRIPT, MOTION_SEED,
    OLLAMA_PORT, PLANNER_MODEL, PROMPT_REWRITER,
    OUTPUT_BVH_PATH, OUTPUT_FBX_PATH, OUTPUT_NPZ_PATH,
    PipelineState, logger, log_input, log_output,
)

# Import helpers from planner (rewrite logic lives there)
from pipeline.nodes.planner import log_fallback_alert, rewrite_prompt_ollama


def generate_motion_node(state: PipelineState) -> dict:
    """Node 3: Run HY-Motion-1.0 diffusion transformer to produce animation data.

    Workflow:
      1. (Optional) Enrich prompt via Ollama or delegate to HY-Motion's Qwen3 rewriter.
      2. Write the prompt file in HY-Motion's expected format: text#duration#seed.
      3. Run local_infer.py with --num_seeds=MOTION_BATCH_SIZE.
      4. Score all candidates (stages 1-3); copy winner to output paths.

    Output preference:
      FBX found → copy to final_animation.fbx + co-copy .npz for critic.
      BVH only  → copy to temp_motion.bvh (triggers Blender MCP fallback).
      Nothing   → log critical error; asset_path stays empty.
    """
    log_input("generate_motion", ["prompt", "laban_plan", "iteration_count"], state)
    logger.info(
        "[generate_motion] Starting HY-Motion-1.0 inference (duration=%ss, seed=%s)...",
        GLOBAL_DURATION, MOTION_SEED,
    )
    prompt = state["prompt"]

    # Clear previous inference outputs to avoid stale file collisions.
    os.system(f"rm -rf {MOTION_ENGINE_DIR}/output/local_infer/*")
    logger.debug("[generate_motion] Cleared previous inference outputs.")

    # ── Prompt rewriter selection ─────────────────────────────────────────────
    use_hymotion_rewriter = False
    final_prompt   = prompt
    final_duration = GLOBAL_DURATION

    if PROMPT_REWRITER == "ollama":
        logger.info("[generate_motion] Rewriter: Ollama (%s) — enriching prompt...", PLANNER_MODEL)
        try:
            final_prompt, final_duration = rewrite_prompt_ollama(prompt)
            logger.info("[generate_motion] Rewritten prompt : %s", final_prompt)
            logger.info("[generate_motion] Inferred duration: %.1fs", final_duration)
        except Exception as e:
            log_fallback_alert(str(e))
            use_hymotion_rewriter = True
            final_prompt   = prompt
            final_duration = GLOBAL_DURATION
    else:
        logger.info("[generate_motion] Rewriter: HY-Motion internal (Qwen3) — delegating to engine.")
        use_hymotion_rewriter = True

    # ── Write prompt file ─────────────────────────────────────────────────────
    input_text_dir = os.path.join(MOTION_ENGINE_DIR, "prompt_inputs")
    os.makedirs(input_text_dir, exist_ok=True)
    prompt_file_path = os.path.join(input_text_dir, "task.txt")
    with open(prompt_file_path, "w") as fh:
        fh.write(f"{final_prompt}#{final_duration}#{MOTION_SEED}\n")
    logger.debug("[generate_motion] Wrote prompt file: %s", prompt_file_path)

    # ── Build command ─────────────────────────────────────────────────────────
    command = [
        "python", MOTION_ENGINE_SCRIPT,
        "--model_path", MOTION_ENGINE_CHECKPOINT,
        "--input_text_dir", "prompt_inputs",
        "--num_seeds", str(MOTION_BATCH_SIZE),
    ]
    if not use_hymotion_rewriter:
        command.append("--disable_rewrite")

    logger.debug("[generate_motion] Command: %s (cwd=%s)", " ".join(command), MOTION_ENGINE_DIR)
    logger.info(
        "[generate_motion] Batch size: %d candidate(s) will be scored; best is kept.",
        MOTION_BATCH_SIZE,
    )

    asset_path = ""

    try:
        subprocess.run(command, cwd=MOTION_ENGINE_DIR, check=True)
        logger.info("[generate_motion] Inference complete. Scanning output directory...")

        output_dir = os.path.join(MOTION_ENGINE_DIR, "output")
        fbx_files  = sorted(glob.glob(f"{output_dir}/**/*.fbx", recursive=True))
        bvh_files  = sorted(glob.glob(f"{output_dir}/**/*.bvh", recursive=True))
        npz_files  = sorted(glob.glob(f"{output_dir}/**/*.npz", recursive=True))
        logger.debug(
            "[generate_motion] Found — FBX: %s | BVH: %s | NPZ: %s",
            fbx_files, bvh_files, npz_files,
        )

        if fbx_files:
            if len(fbx_files) == 1:
                best_fbx = fbx_files[0]
                best_npz = npz_files[0] if npz_files else None
                logger.info("[generate_motion] Single candidate — skipping batch scoring.")
            else:
                best_fbx, best_npz, best_score = fbx_files[0], None, -1.0
                for fbx in fbx_files:
                    stem = fbx.rsplit(".", 1)[0]
                    npz  = stem + ".npz"
                    if not os.path.exists(npz):
                        npz_match = [n for n in npz_files if n.startswith(stem)]
                        npz = npz_match[0] if npz_match else None
                    candidate_score = score_candidate(npz, prompt) if npz else 0.0
                    logger.info(
                        "[generate_motion] Candidate %s → score %.2f",
                        os.path.basename(fbx), candidate_score,
                    )
                    if candidate_score > best_score:
                        best_score, best_fbx, best_npz = candidate_score, fbx, npz
                logger.info(
                    "[generate_motion] Winner: %s (score=%.2f)",
                    os.path.basename(best_fbx), best_score,
                )

            shutil.copy(best_fbx, OUTPUT_FBX_PATH)
            asset_path = OUTPUT_FBX_PATH
            logger.info("[generate_motion] Best FBX copied to %s.", OUTPUT_FBX_PATH)

            if bvh_files:
                shutil.copy(bvh_files[0], OUTPUT_BVH_PATH)
                logger.debug("[generate_motion] Co-generated BVH copied to %s.", OUTPUT_BVH_PATH)
            if best_npz and os.path.exists(best_npz):
                shutil.copy(best_npz, OUTPUT_NPZ_PATH)
                logger.debug("[generate_motion] Winning NPZ copied to %s.", OUTPUT_NPZ_PATH)

        elif bvh_files:
            logger.info(
                "[generate_motion] No FBX found — falling back to BVH. Copying to %s.",
                OUTPUT_BVH_PATH,
            )
            shutil.copy(bvh_files[0], OUTPUT_BVH_PATH)
            asset_path = OUTPUT_BVH_PATH
        else:
            logger.error("[generate_motion] CRITICAL — no animation files were produced by the engine.")

    except subprocess.CalledProcessError as e:
        logger.error("[generate_motion] HY-Motion subprocess failed: %s", e)

    result = {"final_asset_path": asset_path}
    log_output("generate_motion", result)
    return result


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone test for the motion generation node")
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Print the inference command without executing it (default: True).",
    )
    args = parser.parse_args()

    dummy_state: PipelineState = {
        "prompt": "A character performs a heavy broadsword swing.",
        "historical_context": [],
        "laban_plan": {"action_type": "strike", "spatial_level": "high", "speed": "fast"},
        "motion_tensor_path": "",
        "critic_score": 0.0,
        "critic_feedback": "",
        "iteration_count": 1,
        "final_asset_path": "",
    }

    if args.dry_run:
        print("=== generate_motion_node DRY RUN ===")
        print(f"  PROMPT_REWRITER : {PROMPT_REWRITER}")
        print(f"  MOTION_ENGINE_DIR : {MOTION_ENGINE_DIR}")
        print(f"  MOTION_ENGINE_SCRIPT : {MOTION_ENGINE_SCRIPT}")
        print(f"  MOTION_BATCH_SIZE : {MOTION_BATCH_SIZE}")
        print(f"  GLOBAL_DURATION : {GLOBAL_DURATION}s  MOTION_SEED : {MOTION_SEED}")
        print()
        command = [
            "python", MOTION_ENGINE_SCRIPT,
            "--model_path", MOTION_ENGINE_CHECKPOINT,
            "--input_text_dir", "prompt_inputs",
            "--num_seeds", str(MOTION_BATCH_SIZE),
        ]
        if PROMPT_REWRITER != "hymotion":
            command.append("--disable_rewrite")
        print(f"  Command (cwd={MOTION_ENGINE_DIR}):")
        print(f"    {' '.join(command)}")
        print()
        print("  To run a live inference, remove --dry-run (or pass --no-dry-run).")
    else:
        print("=== Running generate_motion_node (live inference) ===")
        result = generate_motion_node(dummy_state)
        print("  final_asset_path:", result["final_asset_path"])
