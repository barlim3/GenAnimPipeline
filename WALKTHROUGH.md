# Multi-Agent Text-to-Motion Generative Pipeline: Deployment Guide

This architecture utilizes a hybrid system: AI inference and multi-agent orchestration run within a Windows Subsystem for Linux (WSL2) environment, while native Windows acts as a hardware-accelerated fallback server for asset translation via Blender and the Model Context Protocol (MCP).

All model-specific settings (LLM name, motion engine paths, service ports, etc.) are centralized in configuration blocks at the top of each source file. To swap a model or change an endpoint, update only the config block -- the pipeline flow stays untouched.

## System Requirements

Due to the heavy use of local Diffusion Transformers and Large Language Models, this pipeline requires a robust, modern workstation.

### Minimum Specifications

*(Capable of running the pipeline, but generation times will be slower and background apps must be closed).*

* **OS:** Windows 11 (build 22000 or higher) to properly support WSL2 GUI/GPU passthrough.
* **CPU:** 8-Core Processor (Intel i7 10th Gen / AMD Ryzen 7 5000 series).
* **System RAM:** 16GB (requires manual memory management).
* **GPU VRAM:** 12GB (NVIDIA RTX 3060 or 4070).
* **Storage:** 50GB SSD space.

### Recommended Specifications

*(Optimized for seamless orchestration, fast diffusion rendering, and background multi-tasking).*

* **OS:** Windows 11.
* **CPU:** 12-Core Processor or higher (Intel i9 13th Gen / AMD Ryzen 9 7000+).
* **System RAM:** 32GB+ (Allows for a generous 24GB WSL2 allocation).
* **GPU VRAM:** 16GB+ (NVIDIA RTX 4080 Super or RTX 5070 Ti).
* **Storage:** 100GB NVMe SSD.
* **GPU Note (RTX 50-Series):** The Blackwell (sm_120) architecture strictly requires CUDA 13.0+ and the absolute latest Windows NVIDIA Game Ready/Studio drivers to function.

---

## Phase 1: Core System Preparation

### 1. Enable WSL2 and Optimize Memory

1. Open PowerShell as Administrator on Windows.
2. Execute:
   ```
   wsl --install
   ```
3. Restart your computer and follow the prompts to create your Ubuntu credentials.
4. **Expand WSL2 RAM Limit:** Open Windows File Explorer, navigate to `%userprofile%` (e.g., `C:\Users\YourName`), and create a file named `.wslconfig`. Add the following to give WSL2 enough breathing room for staging AI weights:
   ```ini
   [wsl2]
   memory=24GB
   ```
5. Open PowerShell and run `wsl --shutdown` to apply.

### 2. Install Native Windows Dependencies

* [**Python 3.10/3.11 for Windows**](https://www.python.org/downloads/windows/): Ensure you check the "Add python.exe to PATH" box during installation.
* [**Blender 4.0+**](https://www.blender.org/download/): Install natively on your Windows drive (acting as the safety net for mesh binding).

---

## Phase 2: Vector Memory (Milvus) Deployment

1. Install [**Docker Desktop for Windows**](https://www.docker.com/products/docker-desktop/). Ensure the "Use the WSL 2 based engine" setting is checked.
2. Open your Ubuntu (WSL2) terminal and run:
   ```bash
   wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
   sudo docker compose up -d
   ```
3. Create `init_milvus.py` in your WSL2 environment (or use the existing file at `E:\GenAnimPipeline\init_milvus.py`):
   ```python
   """
   init_milvus.py - One-Time Milvus Vector Database Initialization

   Creates the 'motion_corrections' collection in the local Milvus instance.
   This collection stores historical motion correction rules as 384-dim embeddings
   so the pipeline's retrieve_context_node (graph.py) can query for past failures
   and apply learned fixes during motion planning.

   Prerequisites:
     - Milvus running via Docker Compose (docker-compose.yml) on localhost:19530
     - pip install pymilvus

   Run once after first deploying Milvus:
       python init_milvus.py
   """

   from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

   connections.connect("default", host="localhost", port="19530")

   if not utility.has_collection("motion_corrections"):
       # Collection schema: auto-ID primary key, 384-dim float vector for semantic
       # similarity search, and a text field storing the human-readable correction rule.
       fields = [
           FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
           FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
           FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
       ]

       schema = CollectionSchema(fields, description="Stores motion correction rules")
       collection = Collection("motion_corrections", schema)

       # IVF_FLAT index with L2 distance for fast approximate nearest-neighbor search
       index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
       collection.create_index(field_name="embedding", index_params=index_params)

       print("Success! The 'motion_corrections' collection is ready.")
   else:
       print("Collection already exists!")
   ```
4. Run `pip install pymilvus` followed by `python init_milvus.py`.

---

## Phase 3: Semantic Planner (Ollama & Llama 3)

1. Install [**Ollama for Windows**](https://ollama.com/).
2. Open Windows "Environment Variables" and add a new System Variable:
   * **Variable name:** `OLLAMA_HOST`
   * **Variable value:** `0.0.0.0`
3. Quit Ollama from your system tray, reopen it, and pull the two models the pipeline requires:
   ```
   ollama pull llama3
   ollama pull llava
   ```
   `llama3` is the planner (Node 2) and the critic's semantic logician (Stage 2). `llava` is the critic's art director (Stage 3) -- it receives skeleton keyframe images for visual quality scoring. Pull `llava` once and Ollama serves it on demand via the same endpoint.

   > **LLaVA VRAM:** LLaVA (7B) requires ~4-8 GB VRAM. Like the planner, the critic passes `"keep_alive": 0` so Ollama releases it immediately after scoring. If VRAM is tight, set `OLLAMA_NUM_GPU=0` to run LLaVA on CPU -- inference will be slower (~10-30s per evaluation) but the result is the same.
   >
   > **Swapping the vision model:** To use a larger or different multimodal model (e.g., `llava:13b`, `bakllava`), pull it and update `VISION_MODEL` in `graph.py`.

> **Swapping the planner model:** To use a different LLM (e.g., Mistral, Gemma2, Phi3), pull it with `ollama pull <model>` and update `PLANNER_MODEL` in the configuration block at the top of `graph.py`. No other code changes are needed.

> **VRAM isolation (optional):** The planner and HY-Motion share the same physical GPU. By default, `graph.py` passes `"keep_alive": 0` in the Ollama request to immediately free VRAM after planning. For full GPU isolation, add a second system environment variable alongside `OLLAMA_HOST`:
> * **Variable name:** `OLLAMA_NUM_GPU`
> * **Variable value:** `0`
>
> This forces Ollama to run entirely on CPU. The planner task is lightweight (one short JSON response), so CPU inference adds only 2-3 seconds -- negligible compared to the diffusion pass.

---

## Phase 4: Deep Learning Environment (WSL2)

1. In your Ubuntu terminal, install Miniconda:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
2. Create and activate the environment:
   ```bash
   conda create -n text2motion python=3.10 -y
   conda activate text2motion
   ```
3. Install build essentials and Git LFS:
   ```bash
   sudo apt update && sudo apt install build-essential git-lfs -y
   git lfs install
   ```
4. **CRITICAL HARDWARE PATCH:** Install PyTorch Nightly specifically built for CUDA 13.0 to support RTX 50-series Blackwell architectures, and downgrade NumPy to prevent file-saving crashes:
   ```bash
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130 --upgrade --force-reinstall
   pip install "numpy<2.0"
   pip install transformers tokenizers accelerate bitsandbytes triton huggingface_hub langgraph langchain-community requests mcp
   pip install bvh matplotlib
   ```

---

## Phase 5: Motion Engine Deployment & Surgical Patching

### 1. Clone and Pull Real Assets

1. Clone the HY-Motion 1.0 [Repository](https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git) to your Windows drive via WSL2 (e.g., `/mnt/e/GenAnimPipeline/HY-Motion-1.0`).
2. Navigate into the folder: `cd /mnt/e/GenAnimPipeline/HY-Motion-1.0`.
3. Pull the actual 3D mesh files to replace Git LFS text pointers:
   ```bash
   git lfs pull
   ```
4. Download the checkpoint weights directly to bypass CLI linking issues:
   ```bash
   pip install huggingface_hub
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='tencent/HY-Motion-1.0', local_dir='ckpts/tencent/HY-Motion-1.0', local_dir_use_symlinks=False)"
   pip install -U "huggingface_hub[cli]"
   # Example for Standard version
   huggingface-cli download tencent/HY-Motion-1.0 --include "HY-Motion-1.0/*" --local-dir ckpts/tencent
   # Example for Lite version
   huggingface-cli download tencent/HY-Motion-1.0 --include "HY-Motion-1.0-Lite/*" --local-dir ckpts/tencent
   ```

### 2. The VRAM Bypass Patch

To prevent VRAM explosion, we must bypass HY-Motion's internal 40GB LLM and set it up to read duration instructions from our master `graph.py` configuration.

1. On Windows, open `E:\GenAnimPipeline\HY-Motion-1.0\hymotion\prompt_engineering\prompt_rewrite.py`.
2. **Comment out the model loader (around line 267):**
   ```python
           # self._init_prompter()
           # self._load_model()
   ```
3. **Overwrite the `__call__` function** to intercept the text and parse the duration dial from the `HY_MOTION_DURATION` environment variable (set by `graph.py`'s config block). Replace the entire function with:
   ```python
       def __call__(self, text, *args, **kwargs):
           import os
           # Read duration from graph.py's GLOBAL_DURATION (passed via env var)
           env_string = os.environ.get("HY_MOTION_DURATION", "2.0")
           dynamic_duration = float(env_string)
           # Return tuple in correct order (duration first, then text)
           return dynamic_duration, text
   ```

> **Swapping the motion engine:** To use a different motion generation model, update `MOTION_ENGINE_DIR`, `MOTION_ENGINE_CHECKPOINT`, and `MOTION_ENGINE_SCRIPT` in the configuration block at the top of `graph.py`. You will also need to adjust the prompt format in `generate_motion_node()` to match the new engine's expected input.

---

## Phase 6: Windows Translation Server (FastMCP Safety Net)

This natively runs Blender on Windows to bind the mesh if HY-Motion's native .fbx exporter fails and falls back to .bvh.

1. Create a directory on your Windows drive (e.g., `E:\GenAnimPipeline\mcp_server`).
2. Open a Windows Command Prompt, navigate to the folder, and setup the environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install fastmcp
   ```
3. Create `translate_to_fbx.py` in this folder (or use the existing file):
   ```python
   """
   translate_to_fbx.py - FastMCP Blender Translation Server (Windows Native)

   Runs natively on the Windows host as a safety-net service for the pipeline.
   When HY-Motion produces only a .bvh file (raw skeleton data), the LangGraph
   orchestrator (graph.py) calls this server over SSE to convert it into a
   game-engine-ready .fbx via headless Blender.

   Start this server BEFORE running the pipeline:
       cd E:\GenAnimPipeline\mcp_server
       venv\Scripts\activate
       python translate_to_fbx.py

   Listens on 0.0.0.0:8000 so WSL2 can reach it via the host gateway IP.
   """

   from fastmcp import FastMCP
   import subprocess

   # -- SERVER CONFIGURATION --
   # Update these paths if Blender or the project directory moves.
   BLENDER_EXE = r"D:\Blender Foundation\Blender 5.0\blender.exe"
   BLENDER_SCRIPT = r"E:\GenAnimPipeline\mcp_server\blender_retarget.py"
   SERVER_PORT = 8000

   mcp = FastMCP("Blender-Translation-Engine")

   @mcp.tool()
   def run_blender_export(bvh_motion_path: str, output_fbx_path: str) -> str:
       """Launches Blender headlessly to import a BVH motion file, bake the
       animation onto the skeleton, and export the result as FBX."""

       command = [
           BLENDER_EXE,
           "-b",                   # Headless mode (no UI)
           "-P", BLENDER_SCRIPT,   # Run the retarget script inside Blender's Python
           "--",                   # Separator: everything after this goes to the script
           bvh_motion_path,
           output_fbx_path
       ]

       try:
           result = subprocess.run(command, capture_output=True, text=True, check=True)
           return f"Success! FBX exported to {output_fbx_path}. Log: {result.stdout}"
       except subprocess.CalledProcessError as e:
           return f"Blender Execution Failed: {e.stderr}"

   if __name__ == "__main__":
       print("Blender Translation Server is ONLINE. Speaking pure SSE...")
       mcp.run(transport="sse", host="0.0.0.0", port=SERVER_PORT)
   ```
4. Create `blender_retarget.py` in the same folder (or use the existing file):
   ```python
   """
   blender_retarget.py - Blender Python Script for BVH-to-FBX Conversion

   Executed inside Blender's embedded Python interpreter by translate_to_fbx.py.
   NOT meant to be run standalone -- Blender injects the 'bpy' module at runtime.

   Workflow:
     1. Clear the default scene (cube, camera, light)
     2. Import the raw BVH skeleton animation
     3. Bake all bone keyframes and export as FBX

   Custom rig binding (loading a character mesh and retargeting the BVH onto it)
   can be added in import_and_export() where noted.

   Usage (called automatically by the MCP server):
       blender.exe -b -P blender_retarget.py -- <input.bvh> <output.fbx>
   """

   import bpy
   import sys


   def clear_scene():
       """Remove all objects from the default Blender scene to start clean."""
       bpy.ops.object.select_all(action='SELECT')
       bpy.ops.object.delete(use_global=False)


   def import_and_export(bvh_path, fbx_path):
       """Import a BVH motion file, bake the animation, and export as FBX."""
       clear_scene()

       # Import the raw BVH skeleton animation data
       try:
           bpy.ops.import_anim.bvh(filepath=bvh_path, filter_glob="*.bvh", global_scale=1.0, use_fps_scale=False)
       except Exception as e:
           print(f"Error importing BVH: {e}")
           sys.exit(1)

       # (Optional) Load a custom character rig and retarget the BVH onto it:
       # bpy.ops.wm.append(filepath="character_rig.blend/Object/Armature")

       # Export the scene as FBX with all bone animations baked
       try:
           bpy.ops.export_scene.fbx(
               filepath=fbx_path,
               use_selection=False,
               bake_anim=True,
               bake_anim_use_all_bones=True,
               bake_anim_use_nla_strips=False,
               bake_anim_use_all_actions=False,
               bake_anim_force_startend_keying=True
           )
           print(f"Successfully exported FBX to {fbx_path}")
       except Exception as e:
           print(f"Error exporting FBX: {e}")
           sys.exit(1)


   if __name__ == "__main__":
       # Blender passes custom arguments after the '--' separator
       if "--" in sys.argv:
           argv = sys.argv[sys.argv.index("--") + 1:]
           if len(argv) >= 2:
               bvh_file = argv[0]
               fbx_file = argv[1]
               import_and_export(bvh_file, fbx_file)
           else:
               print("Error: Missing file paths.")
   ```

---

## Phase 7: The Master LangGraph Orchestrator (WSL2)

Create `graph.py` in `/mnt/e/GenAnimPipeline` (or use the existing file). This contains the fault-tolerant router that prefers native .fbx but seamlessly fails over to .bvh + Blender.

### Configuration Block

All model and service settings live in the configuration block at the top of `graph.py`. To swap any model or change any endpoint, edit only this section:

```python
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
VISION_MODEL = "llava"    # LLaVA vision model for critic Stage 3 art direction
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
OUTPUT_NPZ_PATH = "/mnt/e/GenAnimPipeline/temp_motion.npz"

# -- MCP Blender Fallback (Windows-native paths for the Blender server) --
MCP_SERVER_PORT = 8000
WIN_BVH_PATH = "E:\\GenAnimPipeline\\temp_motion.bvh"
WIN_FBX_PATH = "E:\\GenAnimPipeline\\final_animation.fbx"

# -- Critic / Evaluation Thresholds --
CRITIC_SCORE_THRESHOLD = 0.85  # Minimum score to accept without re-planning
MAX_ITERATIONS = 3             # Hard cap on plan-generate-evaluate loops
```

### Pipeline Flow

The LangGraph workflow connects five nodes in a directed graph with one conditional branch:

```
retrieve_context -> generate_plan -> generate_motion -> evaluate_motion
                         ^                                   |
                         |  (replan)                         v
                         +------- route_evaluation ----> [finish | execute_mcp]
                                                                     |
                                                                     v
                                                                    END
```

| Node | Purpose |
|------|---------|
| `retrieve_context` | Connects to Milvus and loads historical motion correction embeddings. Degrades gracefully if offline. |
| `generate_plan` | Asks the planner LLM (via Ollama on the Windows host) to decompose the prompt into a Laban-style motion plan (action_type, spatial_level, speed). Falls back to a safe default plan if the LLM is unreachable. |
| `generate_motion` | Runs the motion diffusion transformer. Writes the prompt in the engine's expected format, invokes inference, then applies smart fallback logic: prefers native FBX, falls back to BVH, or logs an error if neither exists. Always copies the co-generated `.npz` file (SMPL-H pose data) to `temp_motion.npz` so the critic cascade can evaluate the motion via the `SmplhMocap` adapter. |
| `evaluate_motion` | **4-Stage Critic Cascade.** Runs sequentially -- an earlier failure skips heavier stages: **(1) Kinematic Gatekeeper** checks for foot sliding via forward kinematics (pure Python/numpy); **(2) Semantic Logician** sends a sampled joint trajectory JSON to Llama 3 to verify semantic intent; **(3) Art Director** renders 4 skeleton keyframes with matplotlib and sends the PNG to LLaVA for visual quality scoring; **(4) Human Sign-Off** prints a CLI block and waits for `Y`/`N` input. Requires `bvh`, `matplotlib`, `numpy` (WSL2) and `llava` pulled in Ollama (Windows). |
| `execute_mcp` | (Conditional) When only a BVH was produced, contacts the Windows FastMCP Blender server via SSE to convert it to FBX. |

### Full Source

See the `graph.py` file for the complete implementation. The key architectural features:

* **`_get_windows_host_ip()`** -- Shared helper that resolves the Windows host IP from inside WSL2 via the default gateway. Used by any node that reaches the Windows host (Ollama, MCP server).
* **`PipelineState`** -- TypedDict shared across all nodes. Each node reads what it needs and returns a partial dict to merge back.
* **`route_evaluation()`** -- Conditional router: passes to finish (FBX), Blender fallback (BVH), or loops back to re-plan.

---

## Daily Execution: Startup Sequence

When sitting down to use the pipeline, follow this exact boot sequence to ensure the virtual networks bridge correctly.

### Step 1: Ignite the Background Services (Windows/WSL)

1. Ensure the **Ollama** app is running in your Windows system tray and run `ollama run llama3` in a Windows Command Prompt.
   ```bash
   ollama run llama3
   ollama run llava
   ```
2. Open a WSL2 (Ubuntu) terminal and start your Milvus memory database:
   ```bash
   sudo docker compose up -d
   ```

### Step 2: Boot the Translation Bridge (Windows Native)

Open a standard Windows Command Prompt to start the FastMCP safety net. *Leave this window open in the background.*

```
cd E:\GenAnimPipeline\mcp_server
venv\Scripts\activate
python translate_to_fbx.py
```

### Step 3: Trigger the Orchestrator (WSL2)

Open your WSL2 terminal, activate the PyTorch cu130 environment, and launch the workflow. Ensure the `GLOBAL_DURATION` and other settings in the configuration block at the top of `graph.py` are set to your desired values before running.

```bash
conda activate text2motion
cd /mnt/e/GenAnimPipeline
python graph.py
```

---

## Quick Reference: Swapping Models

| What to swap | Config location | Variables to change |
|---|---|---|
| Planner LLM | `graph.py` config block | `PLANNER_MODEL`, `OLLAMA_PORT` |
| Critic vision model (Stage 3) | `graph.py` config block | `VISION_MODEL` (e.g. `"llava:13b"`, `"bakllava"`) |
| Critic score threshold | `graph.py` config block | `CRITIC_SCORE_THRESHOLD` (default `0.85`) |
| Motion engine | `graph.py` config block | `MOTION_ENGINE_DIR`, `MOTION_ENGINE_CHECKPOINT`, `MOTION_ENGINE_SCRIPT` (+ prompt format in `generate_motion_node()`) |
| Vector database | `graph.py` config block | `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_COLLECTION` |
| Blender version | `mcp_server/translate_to_fbx.py` config block | `BLENDER_EXE` |
| MCP server port | Both `graph.py` and `translate_to_fbx.py` | `MCP_SERVER_PORT` / `SERVER_PORT` |
