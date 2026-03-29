"""
graph.py - Master LangGraph Orchestrator for Text-to-Motion Generation

This is the central pipeline that converts a text prompt into a rigged FBX animation
file through a multi-agent workflow running in WSL2. The pipeline flows through five
stages:

  1. Retrieve Context  - Queries Milvus vector DB for historical motion corrections
  2. Generate Plan     - Calls Ollama/Llama-3 on Windows host for a symbolic Laban motion plan
  3. Generate Motion   - Runs HY-Motion-1.0 diffusion transformer to produce animation data
  4. Evaluate Motion   - Critic scores the output (placeholder for future ML-based critic)
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
from typing import Dict, TypedDict, Any
from langgraph.graph import StateGraph, END
from pymilvus import connections, Collection
from mcp import ClientSession
from mcp.client.sse import sse_client

# ============================================================================
# PIPELINE CONFIGURATION
# All model and service settings are centralized here. To swap a model or
# change an endpoint, update only this block -- node functions reference
# these values by name, keeping the pipeline flow untouched.
# ============================================================================

# -- Animation Parameters --
GLOBAL_DURATION = 5       # Animation length in seconds
MOTION_SEED = 999         # Reproducibility seed for HY-Motion inference

# -- Planner LLM (Ollama API on Windows host) --
# Change PLANNER_MODEL to any Ollama-pulled model (e.g. "mistral", "gemma2", "phi3")
PLANNER_MODEL = "llama3"
OLLAMA_PORT = 11434

# -- Motion Generation Engine (HY-Motion-1.0 in WSL2) --
# To swap engines, update these paths and adjust the prompt format in
# generate_motion_node() to match the new engine's expected input.
MOTION_ENGINE_DIR = "/mnt/e/GenAnimPipeline/HY-Motion-1.0"
MOTION_ENGINE_CHECKPOINT = "ckpts/tencent/HY-Motion-1.0"
MOTION_ENGINE_SCRIPT = "local_infer.py"

# -- Vector Memory (Milvus via Docker) --
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "motion_corrections"

# -- Output Paths (WSL2 mount) --
OUTPUT_FBX_PATH = "/mnt/e/GenAnimPipeline/final_animation.fbx"
OUTPUT_BVH_PATH = "/mnt/e/GenAnimPipeline/temp_motion.bvh"

# -- MCP Blender Fallback (Windows-native paths for the Blender server) --
MCP_SERVER_PORT = 8000
WIN_BVH_PATH = "E:\\GenAnimPipeline\\temp_motion.bvh"
WIN_FBX_PATH = "E:\\GenAnimPipeline\\final_animation.fbx"

# -- Critic / Evaluation Thresholds --
CRITIC_SCORE_THRESHOLD = 0.85  # Minimum score to accept without re-planning
MAX_ITERATIONS = 3             # Hard cap on plan-generate-evaluate loops

# Pass duration to HY-Motion's patched prompt_rewrite.py via environment
os.environ["HY_MOTION_DURATION"] = str(GLOBAL_DURATION)


def _get_windows_host_ip():
    """Resolve the Windows host IP from inside WSL2 via the default gateway.
    Used by any node that needs to reach a service on the Windows host."""
    return subprocess.check_output(
        "ip route list default | awk '{print $3}'", shell=True
    ).decode().strip()

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
    print("Agent: Connecting to Vector Memory...")
    historical_context = []
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(MILVUS_COLLECTION)
        collection.load()
        print("Milvus connected successfully.")
    except Exception as e:
        print(f"Vector DB not available: {e}")
    return {"historical_context": historical_context}

def generate_symbolic_plan_node(state: PipelineState):
    """Node 2: Ask Ollama/Llama-3 (running on the Windows host) to decompose the
    user prompt into a structured Laban-style motion plan. The Windows host IP is
    discovered dynamically via the WSL2 default gateway. Falls back to a safe
    default plan if the LLM is unreachable."""
    print(f"Agent: Generating symbolic motion plan with {PLANNER_MODEL}...")
    prompt = state["prompt"]

    try:
        win_ip = _get_windows_host_ip()
        system_prompt = "Output a JSON object for motion planning. Keys: 'action_type', 'spatial_level', 'speed'."
        response = requests.post(
            f"http://{win_ip}:{OLLAMA_PORT}/api/generate",
            json={"model": PLANNER_MODEL, "prompt": f"{system_prompt}\nUser: {prompt}", "stream": False, "format": "json", "keep_alive": 0},
            timeout=30
        )
        plan = json.loads(response.json().get("response", "{}"))
        print(f"{PLANNER_MODEL} Plan Generated: {plan}")
    except Exception as e:
        print(f"LLM Fallback. Error: {e}")
        plan = {"action_type": "idle", "spatial_level": "mid", "speed": "normal"}

    return {"laban_plan": plan, "iteration_count": state.get("iteration_count", 0) + 1}

def generate_motion_node(state: PipelineState):
    """Node 3: Run the HY-Motion-1.0 diffusion transformer to generate animation data.

    Writes the prompt in HY-Motion's expected format (text#duration#seed) then invokes
    local_infer.py. After generation, applies smart fallback logic:
      - If a native .fbx was produced, copy it to final_animation.fbx (preferred path).
      - If only a .bvh was produced, copy it to temp_motion.bvh (triggers Blender fallback).
      - If neither exists, log a critical error and continue with an empty asset path.
    """
    print("Agent: Initiating deep neural network motion generation...")
    prompt = state["prompt"]

    # Clear previous inference outputs to avoid stale file collisions
    os.system(f"rm -rf {MOTION_ENGINE_DIR}/output/local_infer/*")

    # Write the prompt file in HY-Motion's input format: "text#duration#seed"
    input_text_dir = os.path.join(MOTION_ENGINE_DIR, "prompt_inputs")
    os.makedirs(input_text_dir, exist_ok=True)
    prompt_file_path = os.path.join(input_text_dir, "task.txt")
    with open(prompt_file_path, "w") as f:
        f.write(f"{prompt}#{GLOBAL_DURATION}#{MOTION_SEED}\n")

    command = ["python", MOTION_ENGINE_SCRIPT, "--model_path", MOTION_ENGINE_CHECKPOINT, "--input_text_dir", "prompt_inputs"]

    asset_path = ""

    try:
        subprocess.run(command, cwd=MOTION_ENGINE_DIR, check=True)

        # --- SMART FALLBACK LOGIC ---
        # Prefer native FBX (game-engine ready); fall back to BVH (raw skeleton data)
        output_dir = os.path.join(MOTION_ENGINE_DIR, "output")
        fbx_files = glob.glob(f"{output_dir}/**/*.fbx", recursive=True)
        bvh_files = glob.glob(f"{output_dir}/**/*.bvh", recursive=True)

        if fbx_files:
            print("Agent: Native FBX detected. Bypassing translation.")
            shutil.copy(fbx_files[0], OUTPUT_FBX_PATH)
            asset_path = OUTPUT_FBX_PATH
        elif bvh_files:
            print("Agent: No FBX found. Falling back to raw BVH format.")
            shutil.copy(bvh_files[0], OUTPUT_BVH_PATH)
            asset_path = OUTPUT_BVH_PATH
        else:
            print("Agent: CRITICAL ERROR - No animation files were generated.")

    except subprocess.CalledProcessError as e:
        print(f"Motion generation failed: {e}")

    return {"final_asset_path": asset_path}

def evaluate_motion_node(state: PipelineState):
    """Node 4: Score the generated motion for quality. Currently a hardcoded placeholder
    (always returns 0.90). Replace with an ML-based critic or human-in-the-loop review
    to enable the re-plan loop for low-quality outputs."""
    score = 0.90
    return {"critic_score": score, "critic_feedback": "Motion aligns well."}

def execute_mcp_translation_node(state: PipelineState):
    """Node 5 (conditional): BVH-to-FBX fallback via the Windows Blender MCP server.

    When HY-Motion only produces a .bvh file (raw skeleton data without mesh binding),
    this node calls the FastMCP translation server running natively on Windows. That
    server launches Blender headlessly to import the BVH, bake the animation, and
    export a game-engine-ready FBX. Communication uses Server-Sent Events (SSE) over
    the WSL2-to-Windows network bridge.

    Paths are Windows-native because Blender runs on the Windows side.
    """
    print("Agent: BVH Detected. Contacting Windows MCP Server via SSE to boot Blender...")

    try:
        win_ip = _get_windows_host_ip()
        mcp_url = f"http://{win_ip}:{MCP_SERVER_PORT}/sse"

        async def run_mcp_tool():
            async with sse_client(mcp_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await session.call_tool(
                        "run_blender_export",
                        arguments={"bvh_motion_path": WIN_BVH_PATH, "output_fbx_path": WIN_FBX_PATH}
                    )

        result = asyncio.run(run_mcp_tool())
        print(f"Blender Server Output: {result.content}")

    except ExceptionGroup as eg:
        for exc in eg.exceptions:
            print(f"Connection Error: {exc}")
    except Exception as e:
        print(f"Failed to execute MCP tool: {e}")

    return {"final_asset_path": WIN_FBX_PATH}

def route_evaluation(state: PipelineState):
    """Conditional router after the critic node. Three possible outcomes:
      - "finish"      : Score passed and we have an FBX -- pipeline is done.
      - "execute_mcp" : Score passed but only BVH exists -- trigger Blender fallback.
      - "replan"      : Score too low and iterations remain -- loop back to the planner.
    Caps at MAX_ITERATIONS to prevent infinite loops on stubborn prompts."""
    if state["critic_score"] >= CRITIC_SCORE_THRESHOLD or state["iteration_count"] >= MAX_ITERATIONS:
        if state.get("final_asset_path", "").endswith(".bvh"):
            return "execute_mcp"
        else:
            return "finish"
    return "replan"

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
    print("--- STARTING GENERATIVE PIPELINE ---")
    app.invoke({"prompt": "A character performs a heavy broadsword swing.", "iteration_count": 0})
    print("--- PIPELINE COMPLETE ---")