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

All configuration lives in pipeline_config.json.
All node implementations live in pipeline/nodes/.
See SETUP.md for full environment setup instructions.

CLI usage:
    python graph.py
    python graph.py --prompt "A ninja performs a spinning kick."
    python graph.py --batch-size 8
    python graph.py --rewriter hymotion
"""

from langgraph.graph import StateGraph, END

# ── Shared config and state ────────────────────────────────────────────────────
import pipeline.shared as _cfg
from pipeline.shared import (
    DEFAULT_PROMPT, MOTION_BATCH_SIZE,
    PipelineState, logger,
)

# ── Pipeline nodes ─────────────────────────────────────────────────────────────
from pipeline.nodes.memory  import retrieve_context_node, update_memory_node
from pipeline.nodes.planner import generate_symbolic_plan_node
from pipeline.nodes.motion  import generate_motion_node
from pipeline.nodes.critic  import evaluate_motion_node
from pipeline.nodes.mcp     import execute_mcp_translation_node


# ── Conditional router ────────────────────────────────────────────────────────

def route_evaluation(state: PipelineState) -> str:
    """Conditional router after the critic node. Three possible outcomes:
      - 'finish'      : Score passed and we have an FBX — pipeline is done.
      - 'execute_mcp' : Score passed but only BVH exists — trigger Blender fallback.
      - 'replan'      : Score too low and iterations remain — loop back to planner.
    Caps at MAX_ITERATIONS to prevent infinite loops on stubborn prompts.
    """
    score     = state["critic_score"]
    iteration = state["iteration_count"]
    asset     = state.get("final_asset_path", "")

    from pipeline.shared import CRITIC_SCORE_THRESHOLD, MAX_ITERATIONS
    logger.debug(
        "[route_evaluation] critic_score=%.2f | iteration=%d/%d | asset=%s",
        score, iteration, MAX_ITERATIONS, asset,
    )

    if score >= CRITIC_SCORE_THRESHOLD or iteration >= MAX_ITERATIONS:
        route = "execute_mcp" if asset.endswith(".bvh") else "finish"
    else:
        route = "replan"

    logger.info(
        "[route_evaluation] Decision: %s (score=%.2f, threshold=%.2f, iteration=%d/%d)",
        route.upper(), score, CRITIC_SCORE_THRESHOLD, iteration, MAX_ITERATIONS,
    )
    return route


# ── LangGraph assembly ─────────────────────────────────────────────────────────
#
#   retrieve_context → generate_plan → generate_motion → evaluate_motion
#                           ^                                    |
#                           |                                    v
#                     update_memory  ←── route_evaluation ──────+──→ [finish | execute_mcp]
#                           |                                                  |
#                           └──────────────────────────────────────────────── END

workflow = StateGraph(PipelineState)

workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("generate_plan",    generate_symbolic_plan_node)
workflow.add_node("generate_motion",  generate_motion_node)
workflow.add_node("evaluate_motion",  evaluate_motion_node)
workflow.add_node("update_memory",    update_memory_node)
workflow.add_node("execute_mcp",      execute_mcp_translation_node)

workflow.set_entry_point("retrieve_context")
workflow.add_edge("retrieve_context", "generate_plan")
workflow.add_edge("generate_plan",    "generate_motion")
workflow.add_edge("generate_motion",  "evaluate_motion")

workflow.add_conditional_edges(
    "evaluate_motion",
    route_evaluation,
    {
        "execute_mcp": "execute_mcp",
        "finish":      END,
        "replan":      "update_memory",
    },
)
workflow.add_edge("update_memory", "generate_plan")
workflow.add_edge("execute_mcp",   END)

app = workflow.compile()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GenAnimPipeline — Text-to-Motion Orchestrator")
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=(
            "Number of motion candidates HY-Motion generates per run. "
            "The best-scoring candidate is kept; the rest are discarded. "
            f"Defaults to motion_batch_size in pipeline_config.json (currently {MOTION_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Motion description to generate. Defaults to default_prompt in pipeline_config.json.",
    )
    parser.add_argument(
        "--rewriter", type=str, default=None, choices=["ollama", "hymotion"],
        help=(
            "Prompt rewriter to use before motion generation. "
            "'ollama' (default) enriches the prompt via Llama3 on the Windows host and "
            "falls back to HY-Motion's internal Qwen3 rewriter if Ollama is unreachable. "
            "'hymotion' delegates entirely to HY-Motion's internal rewriter (no fallback alert). "
            "When --engine kimodo is set, 'hymotion' is treated as no rewriter."
        ),
    )
    parser.add_argument(
        "--engine", type=str, default=None, choices=["hy-motion", "kimodo"],
        help=(
            "Motion generation backend to use. Overrides motion_engine.active in "
            "pipeline_config.json for this run only. "
            "'hy-motion' (default) uses HY-Motion-1.0 (outputs FBX + NPZ). "
            "'kimodo' uses NVIDIA Kimodo (outputs BVH + NPZ; routes through Blender MCP for FBX). "
            "Requires 'pip install kimodo[all]' and ~17 GB VRAM."
        ),
    )
    args = parser.parse_args()

    # Apply CLI overrides to the shared config so all modules see the updated values.
    if args.batch_size is not None:
        _cfg.MOTION_BATCH_SIZE = args.batch_size
        import pipeline.nodes.motion as _motion_node
        _motion_node.MOTION_BATCH_SIZE = args.batch_size

    if args.rewriter is not None:
        _cfg.PROMPT_REWRITER = args.rewriter
        import pipeline.nodes.planner as _planner_node
        import pipeline.nodes.motion  as _motion_node
        _planner_node.PROMPT_REWRITER = args.rewriter
        _motion_node.PROMPT_REWRITER  = args.rewriter

    if args.engine is not None:
        _cfg.ACTIVE_ENGINE = args.engine

    prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPT

    logger.info("══════════════════════════════════════════════════════")
    logger.info("  GENERATIVE ANIMATION PIPELINE — STARTING")
    logger.info("  Prompt     : %s", prompt)
    logger.info("  Engine     : %s", _cfg.ACTIVE_ENGINE)
    logger.info("  Rewriter   : %s", _cfg.PROMPT_REWRITER)
    logger.info("  Batch size : %d", _cfg.MOTION_BATCH_SIZE)
    logger.info("  Log file   : %s", _cfg.LOG_FILE)
    logger.info("══════════════════════════════════════════════════════")

    app.invoke({"prompt": prompt, "iteration_count": 0})

    logger.info("══════════════════════════════════════════════════════")
    logger.info("  PIPELINE COMPLETE")
    logger.info("══════════════════════════════════════════════════════")
