"""
pipeline/nodes/planner.py — Symbolic motion planner and prompt rewriter.

Nodes:
  generate_symbolic_plan_node — Node 2: produce a Laban-style motion plan.

Helpers:
  rewrite_prompt_ollama  — Enrich the user prompt via Llama3 on the Windows host.
  log_fallback_alert     — Emit a red ANSI alert when Ollama falls back unexpectedly.

Standalone usage (calls a real Ollama endpoint if available, else shows fallback):
    python -m pipeline.nodes.planner
    python -m pipeline.nodes.planner --rewriter hymotion
"""

import json

import requests

from pipeline.shared import (
    ANSI_BOLD, ANSI_RED, ANSI_RESET,
    GLOBAL_DURATION, OLLAMA_PORT, PLANNER_MODEL, PROMPT_REWRITER,
    PipelineState, get_windows_host_ip, logger, log_input, log_output,
)


# ── Fallback alert ────────────────────────────────────────────────────────────

def log_fallback_alert(reason: str) -> None:
    """Emit a highly visible red alert to both console and the rotating log file.

    Called ONLY when PROMPT_REWRITER='ollama' but Ollama is unreachable, so the
    pipeline falls back to HY-Motion's internal rewriter unexpectedly.
    Never called when PROMPT_REWRITER='hymotion' is explicitly chosen.
    """
    border  = "!" * 70
    message = (
        f"\n{ANSI_RED}{ANSI_BOLD}"
        f"{border}\n"
        f"  PROMPT REWRITER FALLBACK TRIGGERED\n"
        f"  Configured rewriter : Ollama ({PLANNER_MODEL})\n"
        f"  Reason              : {reason}\n"
        f"  Falling back to     : HY-Motion internal rewriter (Qwen3)\n"
        f"  Action required     : Check Ollama is running on the Windows host\n"
        f"                        and that no_proxy includes {OLLAMA_PORT}.\n"
        f"{border}"
        f"{ANSI_RESET}"
    )
    # Print directly so ANSI codes reach the terminal even if the log handler strips them.
    print(message, flush=True)
    logger.warning(
        "PROMPT REWRITER FALLBACK — Ollama unreachable (%s). "
        "HY-Motion internal rewriter (Qwen3) will be used instead.", reason
    )


# ── Ollama prompt rewriter ────────────────────────────────────────────────────

def rewrite_prompt_ollama(prompt: str) -> tuple:
    """Call Llama3 via Ollama to enrich the motion prompt and infer duration.

    Returns (rewritten_prompt, duration_seconds) on success.
    Raises RuntimeError on any failure so the caller can trigger the fallback.
    """
    system_prompt = (
        "You are a motion capture expert. Given a natural-language description of a "
        "character action, produce a detailed motion prompt optimised for a 3D motion "
        "diffusion model and estimate how long the action takes in seconds.\n"
        "Respond ONLY with a JSON object using these exact keys:\n"
        '  {"rewritten_prompt": "<enriched description>", "duration_seconds": <float>}'
    )
    win_ip = get_windows_host_ip()
    response = requests.post(
        f"http://{win_ip}:{OLLAMA_PORT}/api/generate",
        json={
            "model": PLANNER_MODEL,
            "prompt": f"{system_prompt}\n\nUser prompt: {prompt}",
            "stream": False,
            "format": "json",
            "keep_alive": 0,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = json.loads(response.json().get("response", "{}"))

    rewritten = data.get("rewritten_prompt", "").strip()
    duration  = float(data.get("duration_seconds", GLOBAL_DURATION))

    if not rewritten:
        raise RuntimeError("Ollama returned an empty rewritten_prompt.")

    duration = max(1.0, min(duration, GLOBAL_DURATION * 2))
    return rewritten, duration


# ── Node ──────────────────────────────────────────────────────────────────────

def generate_symbolic_plan_node(state: PipelineState) -> dict:
    """Node 2: Produce a structured Laban-style motion plan for the generator.

    When PROMPT_REWRITER='ollama': calls Llama-3 on the Windows host to decompose
    the prompt into a JSON plan (action_type, spatial_level, speed). Falls back to
    a safe default plan if the LLM is unreachable.

    When PROMPT_REWRITER='hymotion': skips the Ollama call entirely — HY-Motion's
    internal Qwen3 rewriter handles all prompt decomposition and duration inference
    inside the engine. A passthrough plan is logged in the same format for consistency.
    """
    log_input("generate_plan", ["prompt", "historical_context", "iteration_count", "critic_feedback"], state)
    prompt = state["prompt"]

    if PROMPT_REWRITER == "hymotion":
        plan = {"action_type": "delegated", "spatial_level": "delegated", "speed": "delegated"}
        logger.info(
            "[generate_plan] Rewriter=hymotion — skipping Ollama. "
            "Motion parameters will be inferred internally by HY-Motion (Qwen3). Plan: %s", plan
        )
    else:
        logger.info(
            "[generate_plan] Generating symbolic motion plan with %s (iteration %s)...",
            PLANNER_MODEL, state.get("iteration_count", 0),
        )
        try:
            win_ip = get_windows_host_ip()
            logger.debug("[generate_plan] Resolved Windows host IP: %s", win_ip)
            system_prompt = (
                "Output a JSON object for motion planning. "
                "Keys: 'action_type', 'spatial_level', 'speed'."
            )
            response = requests.post(
                f"http://{win_ip}:{OLLAMA_PORT}/api/generate",
                json={
                    "model": PLANNER_MODEL,
                    "prompt": f"{system_prompt}\nUser: {prompt}",
                    "stream": False,
                    "format": "json",
                    "keep_alive": 0,
                },
                timeout=30,
            )
            plan = json.loads(response.json().get("response", "{}"))
            logger.info("[generate_plan] Plan received: %s", plan)
        except Exception as e:
            logger.warning("[generate_plan] LLM unreachable — using fallback plan. Error: %s", e)
            plan = {"action_type": "idle", "spatial_level": "mid", "speed": "normal"}

    result = {"laban_plan": plan, "iteration_count": state.get("iteration_count", 0) + 1}
    log_output("generate_plan", result)
    return result


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone test for the planner node")
    parser.add_argument("--rewriter", choices=["ollama", "hymotion"], default=PROMPT_REWRITER)
    args = parser.parse_args()

    # Override the module-level setting so generate_symbolic_plan_node picks it up.
    import pipeline.shared as _shared
    import pipeline.nodes.planner as _this
    _shared.PROMPT_REWRITER = args.rewriter
    _this.PROMPT_REWRITER   = args.rewriter  # local reference used directly in the function

    dummy_state: PipelineState = {
        "prompt": "A character performs a heavy broadsword swing.",
        "historical_context": [],
        "laban_plan": {},
        "motion_tensor_path": "",
        "critic_score": 0.0,
        "critic_feedback": "",
        "iteration_count": 0,
        "final_asset_path": "",
    }

    print(f"=== generate_symbolic_plan_node (rewriter={args.rewriter}) ===")
    result = generate_symbolic_plan_node(dummy_state)
    print("  laban_plan      :", result["laban_plan"])
    print("  iteration_count :", result["iteration_count"])
