"""
pipeline/nodes/motion.py — Motion generation node.

Dispatches to either HY-Motion-1.0 or Kimodo depending on the active engine
configured in pipeline_config.json (motion_engine.active) or the --engine CLI flag.

Node:
  generate_motion_node — Node 3: invoke the selected engine, score batch
                         candidates where supported, copy the best output
                         to the configured output paths.

Standalone usage (dry-run — prints the command that would be executed):
    python -m pipeline.nodes.motion
    python -m pipeline.nodes.motion --dry-run
"""

import glob
import os
import shutil
import subprocess

from pipeline.nodes.critic import score_candidate
from pipeline.shared import (
    ACTIVE_ENGINE,
    GLOBAL_DURATION, MOTION_BATCH_SIZE, MOTION_ENGINE_CHECKPOINT,
    MOTION_ENGINE_DIR, MOTION_ENGINE_SCRIPT, MOTION_SEED,
    OLLAMA_PORT, PLANNER_MODEL, PROMPT_REWRITER,
    KIMODO_MODEL, KIMODO_DIFFUSION_STEPS, KIMODO_CFG_WEIGHT, KIMODO_OUTPUT_BVH,
    OUTPUT_BVH_PATH, OUTPUT_FBX_PATH, OUTPUT_NPZ_PATH,
    PipelineState, logger, log_input, log_output,
)
from pipeline.nodes.planner import log_fallback_alert, rewrite_prompt_ollama


# ── Engine-level mutable aliases (overridden by graph.py --engine) ─────────────
# graph.py writes to pipeline.shared.ACTIVE_ENGINE; this module re-reads at call
# time so a single import is sufficient.


# ── HY-Motion-1.0 runner ──────────────────────────────────────────────────────

def _run_hymotion(prompt: str, laban_plan: dict) -> str:
    """Run HY-Motion-1.0 and return the path of the best output asset.

    Steps:
      1. Optionally rewrite the prompt via Ollama or delegate to Qwen3.
      2. Write the HY-Motion prompt file (text#duration#seed).
      3. Run local_infer.py with --num_seeds=MOTION_BATCH_SIZE.
      4. Score all candidates (stages 1-3); copy winner to output paths.

    Returns the absolute path of the selected asset (.fbx or .bvh), or "" on failure.
    """
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

    # Write prompt file
    input_text_dir   = os.path.join(MOTION_ENGINE_DIR, "prompt_inputs")
    os.makedirs(input_text_dir, exist_ok=True)
    prompt_file_path = os.path.join(input_text_dir, "task.txt")
    with open(prompt_file_path, "w") as fh:
        fh.write(f"{final_prompt}#{final_duration}#{MOTION_SEED}\n")
    logger.debug("[generate_motion] Wrote HY-Motion prompt file: %s", prompt_file_path)

    # Clear stale outputs
    os.system(f"rm -rf {MOTION_ENGINE_DIR}/output/local_infer/*")

    command = [
        "python", MOTION_ENGINE_SCRIPT,
        "--model_path", MOTION_ENGINE_CHECKPOINT,
        "--input_text_dir", "prompt_inputs",
        "--num_seeds", str(MOTION_BATCH_SIZE),
    ]
    if not use_hymotion_rewriter:
        command.append("--disable_rewrite")

    logger.debug("[generate_motion] HY-Motion command: %s (cwd=%s)", " ".join(command), MOTION_ENGINE_DIR)
    logger.info("[generate_motion] Batch size: %d candidate(s); best will be kept.", MOTION_BATCH_SIZE)

    asset_path = ""
    try:
        subprocess.run(command, cwd=MOTION_ENGINE_DIR, check=True)
        logger.info("[generate_motion] HY-Motion inference complete. Scanning output...")

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
                "[generate_motion] No FBX — falling back to BVH. Copying to %s.", OUTPUT_BVH_PATH
            )
            shutil.copy(bvh_files[0], OUTPUT_BVH_PATH)
            asset_path = OUTPUT_BVH_PATH
        else:
            logger.error("[generate_motion] CRITICAL — HY-Motion produced no animation files.")

    except subprocess.CalledProcessError as e:
        logger.error("[generate_motion] HY-Motion subprocess failed: %s", e)

    return asset_path


# ── Kimodo runner ──────────────────────────────────────────────────────────────

def _run_kimodo(prompt: str) -> str:
    """Run Kimodo (NVIDIA) and return the path of the best output asset.

    Kimodo has no native FBX exporter. When KIMODO_OUTPUT_BVH is True (default)
    it produces BVH alongside the NPZ; the pipeline's existing Blender MCP
    fallback then converts BVH → FBX.

    Batch generation (MOTION_BATCH_SIZE > 1) is supported: Kimodo writes
    motion_00.npz … motion_N.npz into a temp output folder.  Because Kimodo's
    NPZ schema (posed_joints / local_rot_mats) differs from HY-Motion's SMPL-H
    schema, the SmplhMocap-based score_candidate() cannot rank candidates here.
    Instead we pick the first one (motion_00) — future work can add a
    Kimodo-native scorer.

    Returns the absolute path of the selected asset (.bvh or .npz), or "" on failure.
    """
    # Kimodo reads the prompt directly from CLI; no prompt file needed.
    # Use the Ollama rewriter if configured, same as HY-Motion.
    final_prompt   = prompt
    final_duration = float(GLOBAL_DURATION)

    if PROMPT_REWRITER == "ollama":
        logger.info("[generate_motion] Rewriter: Ollama (%s) — enriching prompt...", PLANNER_MODEL)
        try:
            final_prompt, final_duration = rewrite_prompt_ollama(prompt)
            logger.info("[generate_motion] Rewritten prompt : %s", final_prompt)
            logger.info("[generate_motion] Inferred duration: %.1fs", final_duration)
        except Exception as e:
            log_fallback_alert(str(e))
            final_prompt   = prompt
            final_duration = float(GLOBAL_DURATION)
    else:
        logger.info("[generate_motion] Rewriter: none (Kimodo has no internal rewriter).")

    # Kimodo writes outputs to a folder when num_samples > 1, or a single file stem otherwise.
    kimodo_output_dir = os.path.join(os.path.dirname(OUTPUT_NPZ_PATH), "kimodo_output")
    os.makedirs(kimodo_output_dir, exist_ok=True)
    # Clear stale outputs
    os.system(f"rm -rf {kimodo_output_dir}/*")

    output_stem = os.path.join(kimodo_output_dir, "motion")

    command = [
        "kimodo_gen", final_prompt,
        "--model", KIMODO_MODEL,
        "--duration", str(final_duration),
        "--num_samples", str(MOTION_BATCH_SIZE),
        "--diffusion_steps", str(KIMODO_DIFFUSION_STEPS),
        "--cfg_weight", str(KIMODO_CFG_WEIGHT),
        "--seed", str(MOTION_SEED),
        "--output", output_stem,
    ]
    if KIMODO_OUTPUT_BVH:
        command.append("--bvh")

    logger.debug("[generate_motion] Kimodo command: %s", " ".join(command))
    logger.info(
        "[generate_motion] Kimodo model=%s | duration=%.1fs | samples=%d | steps=%d",
        KIMODO_MODEL, final_duration, MOTION_BATCH_SIZE, KIMODO_DIFFUSION_STEPS,
    )

    asset_path = ""
    try:
        subprocess.run(command, check=True)
        logger.info("[generate_motion] Kimodo inference complete. Scanning output...")

        # Kimodo writes motion_00.npz, motion_01.npz… (or motion.npz for a single sample)
        # and motion_00.bvh… when --bvh is passed.
        if MOTION_BATCH_SIZE > 1:
            npz_files = sorted(glob.glob(os.path.join(kimodo_output_dir, "*.npz")))
            bvh_files = sorted(glob.glob(os.path.join(kimodo_output_dir, "*.bvh")))
        else:
            npz_files = sorted(glob.glob(output_stem + "*.npz"))
            bvh_files = sorted(glob.glob(output_stem + "*.bvh"))

        logger.debug(
            "[generate_motion] Kimodo found — BVH: %s | NPZ: %s", bvh_files, npz_files
        )

        # Batch scoring is not yet supported for Kimodo's NPZ schema — take motion_00.
        best_npz = npz_files[0] if npz_files else None
        best_bvh = bvh_files[0] if bvh_files else None

        if MOTION_BATCH_SIZE > 1 and len(npz_files) > 1:
            logger.info(
                "[generate_motion] Kimodo produced %d candidates — selecting motion_00 "
                "(per-candidate scoring not yet supported for Kimodo NPZ schema).",
                len(npz_files),
            )

        if best_bvh:
            shutil.copy(best_bvh, OUTPUT_BVH_PATH)
            asset_path = OUTPUT_BVH_PATH
            logger.info(
                "[generate_motion] Best BVH copied to %s (will route through Blender MCP).",
                OUTPUT_BVH_PATH,
            )
        if best_npz:
            shutil.copy(best_npz, OUTPUT_NPZ_PATH)
            logger.debug("[generate_motion] Best NPZ copied to %s.", OUTPUT_NPZ_PATH)

        if not asset_path:
            logger.error("[generate_motion] CRITICAL — Kimodo produced no animation files.")

    except FileNotFoundError:
        logger.error(
            "[generate_motion] 'kimodo_gen' not found — is the kimodo package installed? "
            "Run: pip install kimodo[all]"
        )
    except subprocess.CalledProcessError as e:
        logger.error("[generate_motion] Kimodo subprocess failed: %s", e)

    return asset_path


# ── LangGraph node ─────────────────────────────────────────────────────────────

def generate_motion_node(state: PipelineState) -> dict:
    """Node 3: Run the selected motion engine to produce animation data.

    Reads pipeline.shared.ACTIVE_ENGINE at call time so that a --engine CLI
    override applied after import is respected.
    """
    import pipeline.shared as _shared
    active = _shared.ACTIVE_ENGINE

    log_input("generate_motion", ["prompt", "laban_plan", "iteration_count"], state)
    logger.info("[generate_motion] Active engine: %s", active)

    prompt = state["prompt"]

    if active == "kimodo":
        asset_path = _run_kimodo(prompt)
    else:
        if active != "hy-motion":
            logger.warning(
                "[generate_motion] Unknown engine '%s' — defaulting to hy-motion.", active
            )
        asset_path = _run_hymotion(prompt, state.get("laban_plan", {}))

    result = {"final_asset_path": asset_path}
    log_output("generate_motion", result)
    return result


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pipeline.shared as _shared

    parser = argparse.ArgumentParser(description="Standalone test for the motion generation node")
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Print the inference command without executing it (default: True).",
    )
    parser.add_argument(
        "--engine", type=str, choices=["hy-motion", "kimodo"], default=None,
        help="Override the active engine for this dry-run.",
    )
    args = parser.parse_args()

    active = args.engine or _shared.ACTIVE_ENGINE

    print("=== generate_motion_node DRY RUN ===")
    print(f"  ACTIVE_ENGINE   : {active}")
    print(f"  PROMPT_REWRITER : {PROMPT_REWRITER}")
    print(f"  GLOBAL_DURATION : {GLOBAL_DURATION}s  MOTION_SEED : {MOTION_SEED}")
    print(f"  MOTION_BATCH_SIZE: {MOTION_BATCH_SIZE}")
    print()

    if active == "kimodo":
        print(f"  KIMODO_MODEL          : {KIMODO_MODEL}")
        print(f"  KIMODO_DIFFUSION_STEPS: {KIMODO_DIFFUSION_STEPS}")
        print(f"  KIMODO_CFG_WEIGHT     : {KIMODO_CFG_WEIGHT}")
        print(f"  KIMODO_OUTPUT_BVH     : {KIMODO_OUTPUT_BVH}")
        print()
        cmd = [
            "kimodo_gen", "<prompt>",
            "--model", KIMODO_MODEL,
            "--duration", str(GLOBAL_DURATION),
            "--num_samples", str(MOTION_BATCH_SIZE),
            "--diffusion_steps", str(KIMODO_DIFFUSION_STEPS),
            "--cfg_weight", str(KIMODO_CFG_WEIGHT),
            "--seed", str(MOTION_SEED),
            "--output", "<kimodo_output/motion>",
        ]
        if KIMODO_OUTPUT_BVH:
            cmd.append("--bvh")
        print(f"  Command: {' '.join(cmd)}")
    else:
        print(f"  MOTION_ENGINE_DIR    : {MOTION_ENGINE_DIR}")
        print(f"  MOTION_ENGINE_SCRIPT : {MOTION_ENGINE_SCRIPT}")
        cmd = [
            "python", MOTION_ENGINE_SCRIPT,
            "--model_path", MOTION_ENGINE_CHECKPOINT,
            "--input_text_dir", "prompt_inputs",
            "--num_seeds", str(MOTION_BATCH_SIZE),
            "--disable_rewrite",
        ]
        print(f"  Command (cwd={MOTION_ENGINE_DIR}): {' '.join(cmd)}")

    print()
    print("  To run live inference, remove --dry-run.")
