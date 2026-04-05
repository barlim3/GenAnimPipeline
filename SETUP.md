# Multi-Agent Text-to-Motion Generative Pipeline: Deployment Guide

This pipeline supports two deployment environments. Follow the path that matches your OS.

| | **Windows + WSL2** *(recommended for Windows users)* | **Native Linux** |
|---|---|---|
| AI inference & orchestrator | WSL2 (Ubuntu) | Linux host |
| Docker / Milvus | WSL2 via Docker Desktop | Native Docker |
| Ollama + LLMs | Windows host | Linux host |
| Blender + MCP server | Windows host | Linux host |
| Dashboard | Any browser on Windows | Any browser on Linux |
| Networking note | Ollama/MCP reached via Windows host gateway IP from WSL2 | All services on `localhost` |

All model-specific settings are centralized in `pipeline_config.json`. Node implementations live in `pipeline/nodes/`; shared infrastructure lives in `pipeline/shared.py`.

> **Switching motion engines:** `pipeline_config.json → motion_engine.active` controls whether HY-Motion-1.0 or Kimodo is used. Override for a single run with `--engine hy-motion` or `--engine kimodo`.

---

## System Requirements

Due to the heavy use of local Diffusion Transformers and Large Language Models, this pipeline requires a robust, modern workstation.

### Minimum Specifications

*(Capable of running the pipeline, but generation times will be slower and background apps must be closed).*

* **OS:** Windows 11 (build 22000 or higher) **or** Ubuntu 20.04+ / Debian 11+.
* **CPU:** 8-Core Processor (Intel i7 10th Gen / AMD Ryzen 7 5000 series).
* **System RAM:** 16GB (requires manual memory management).
* **GPU VRAM:** 12GB (NVIDIA RTX 3060 or 4070).
* **Storage:** 50GB SSD space.

### Recommended Specifications

*(Optimized for seamless orchestration, fast diffusion rendering, and background multi-tasking).*

* **OS:** Windows 11 **or** Ubuntu 22.04 LTS.
* **CPU:** 12-Core Processor or higher (Intel i9 13th Gen / AMD Ryzen 9 7000+).
* **System RAM:** 32GB+ (Windows: allows a generous 24GB WSL2 allocation; Linux: direct allocation).
* **GPU VRAM:** 16GB+ (NVIDIA RTX 4080 Super or RTX 5070 Ti).
* **Storage:** 100GB NVMe SSD.
* **GPU Note (RTX 50-Series):** The Blackwell (sm_120) architecture strictly requires CUDA 13.0+ and the latest NVIDIA drivers. On Linux, use the latest production driver from `graphics-drivers/ppa` or the NVIDIA `.run` installer.

---

## Phase 1: Core System Preparation

### Windows + WSL2

1. Open PowerShell as Administrator and install WSL2:
   ```
   wsl --install
   ```
2. Restart and follow the prompts to create your Ubuntu credentials.
3. **Expand WSL2 RAM limit:** Navigate to `%userprofile%` in Explorer and create `.wslconfig`:
   ```ini
   [wsl2]
   memory=24GB
   ```
4. Apply with `wsl --shutdown` in PowerShell.
5. Install native Windows dependencies:
   * [**Python 3.10/3.11 for Windows**](https://www.python.org/downloads/windows/) — check "Add python.exe to PATH".
   * [**Blender 4.0+**](https://www.blender.org/download/) — installs on Windows; used headlessly by the MCP server.

### Native Linux

1. Install Python 3.10 or 3.11 and Blender:
   ```bash
   sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip blender
   ```
   Or install Blender via the [official `.tar` release](https://www.blender.org/download/) for the latest version.
2. Note the Blender executable path — you will need it in Phase 6:
   ```bash
   which blender   # typically /usr/bin/blender
   ```
3. No WSL2 or RAM limit configuration is needed — resource allocation is managed natively.

---

## Phase 2: Vector Memory (Milvus) Deployment

#### Windows + WSL2
1. Install [**Docker Desktop for Windows**](https://www.docker.com/products/docker-desktop/). Ensure the "Use the WSL 2 based engine" setting is checked.
2. Open your Ubuntu (WSL2) terminal and run:

#### Native Linux
1. Install Docker Engine:
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER   # log out and back in after this
   ```
2. In your Linux terminal, run:

---

*(Both environments continue from here with the same commands.)*


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
4. Run `pip install langchain langgraph pymilvus` followed by `python init_milvus.py`.

---

## Phase 3: Semantic Planner (Ollama & Llama 3)

#### Windows + WSL2

1. Install [**Ollama for Windows**](https://ollama.com/).
2. Open Windows "Environment Variables" and add a new System Variable so WSL2 can reach the Windows host:
   * **Variable name:** `OLLAMA_HOST`
   * **Variable value:** `0.0.0.0`
3. Quit Ollama from your system tray, reopen it, and pull the required models in a Windows Command Prompt:
   ```
   ollama pull llama3
   ollama pull llava
   ```

#### Native Linux

1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
   Ollama runs as a systemd service and listens on `localhost:11434` — no `OLLAMA_HOST` variable needed.
2. Pull the required models:
   ```bash
   ollama pull llama3
   ollama pull llava
   ```

---

`llama3` is used by the planner (Node 2) and the critic's semantic logician (Stage 2). `llava` is the critic's art director (Stage 3) — it receives rendered skeleton keyframe images for visual quality scoring.

> **LLaVA VRAM:** LLaVA (7B) requires ~4-8 GB VRAM. The critic passes `"keep_alive": 0` so Ollama releases it immediately after scoring. If VRAM is tight, set `OLLAMA_NUM_GPU=0` to run LLaVA on CPU — inference will be slower (~10-30s per evaluation) but the result is the same.

> **Swapping models:** Pull any Ollama-compatible model and update `ollama.planner_model` or `ollama.vision_model` in `pipeline_config.json`. No code changes needed.

> **VRAM isolation (optional):** Set the environment variable `OLLAMA_NUM_GPU=0` to force CPU-only Ollama inference. The planner task is lightweight (one short JSON response), so CPU mode adds only ~2-3 seconds per run.

---

## Phase 4: Deep Learning Environment

*(Both environments follow the same steps — WSL2 users run these in their Ubuntu terminal; native Linux users run them directly.)*

1. Install Miniconda:
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
   pip install transformers tokenizers "accelerate>=0.29.0" bitsandbytes triton huggingface_hub langgraph langchain-community requests mcp
   pip install bvh matplotlib sentence-transformers
   ```

   > **`accelerate` version requirement:** `sentence_transformers` pulls in `peft` as an indirect dependency. `peft>=0.7` requires `accelerate>=0.29.0` for `clear_device_cache`. Pinning `accelerate>=0.29.0` above prevents an `ImportError: cannot import name 'clear_device_cache'` on pipeline startup. If you see this error in an existing environment, fix it with `pip install --upgrade accelerate`.

   > **RTX 30/40-series:** Use the stable PyTorch release instead: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

---

## Phase 5: Motion Engine Deployment & Surgical Patching

### 1. Clone and Pull Real Assets

Clone into the directory that matches your environment:

| Environment | Recommended clone path |
|---|---|
| Windows + WSL2 | `/mnt/e/GenAnimPipeline/HY-Motion-1.0` (on the Windows drive, accessible from both sides) |
| Native Linux | `~/GenAnimPipeline/HY-Motion-1.0` |

1. Clone the HY-Motion 1.0 [Repository](https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git) to your chosen path.
2. Navigate into the folder (adjust for your environment):
   ```bash
   # Windows + WSL2
   cd /mnt/e/GenAnimPipeline/HY-Motion-1.0

   # Native Linux
   cd ~/GenAnimPipeline/HY-Motion-1.0
   ```
3. Pull the actual 3D mesh files to replace Git LFS text pointers:
   ```bash
   git lfs pull
   ```
4. Download the checkpoint weights directly to bypass CLI linking issues:
   ```bash
   pip install -r HY-Motion-1.0/requirements.txt
   pip install huggingface_hub
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='tencent/HY-Motion-1.0', local_dir='ckpts/tencent/HY-Motion-1.0', local_dir_use_symlinks=False)"
   pip install -U "huggingface_hub[cli]"

   # Example for Standard version
   huggingface-cli download tencent/HY-Motion-1.0 --include "HY-Motion-1.0/*" --local-dir ckpts/tencent

   # Example for Lite version
   huggingface-cli download tencent/HY-Motion-1.0 --include "HY-Motion-1.0-Lite/*" --local-dir ckpts/tencent

   # CLIP Large
   huggingface-cli download openai/clip-vit-large-patch14 --local-dir ckpts/clip-vit-large-patch14/

   # Qwen Text Encoder
   huggingface-cli download Qwen/Qwen3-8B --local-dir ckpts/Qwen3-8B

   # Optional
   huggingface-cli download Text2MotionPrompter/Text2MotionPrompter --local-dir ckpts/Text2MotionPrompter
   ```

### 2. The VRAM Bypass Patch

To prevent VRAM explosion, we must bypass HY-Motion's internal 40GB LLM and set it up to read duration instructions from our master `graph.py` configuration.

1. On Windows, open `E:\GenAnimPipeline\HY-Motion-1.0\hymotion\prompt_engineering\prompt_rewrite.py`.
2. **Comment out the model loader (around line 267):**
   ```python
           # self._init_prompter()
           # self._load_model()
   ```
3. **Overwrite the `rewrite_prompt_and_infer_time` function** to intercept the text and parse the duration dial from the `HY_MOTION_DURATION` environment variable (set by `graph.py`'s config block). Replace the entire function with:
   ```python
       def __call__(self, text, *args, **kwargs):
           import os
           # Read duration from graph.py's GLOBAL_DURATION (passed via env var)
           env_string = os.environ.get("HY_MOTION_DURATION", "2.0")
           dynamic_duration = float(env_string)
           # Return tuple in correct order (duration first, then text)
           return dynamic_duration, text
   ```

> **Swapping the motion engine:** Update `dir`, `checkpoint`, and `script` under `motion_engine` in `pipeline_config.json`. You will also need to adjust the prompt format in `pipeline/nodes/motion.py` (`generate_motion_node`) to match the new engine's expected input.

---

## Phase 6: BVH-to-FBX Translation Server (FastMCP Safety Net)

This phase sets up the FastMCP Blender server that converts `.bvh` files to `.fbx` when the motion engine doesn't produce a native FBX (e.g., when using Kimodo, or when HY-Motion's FBX exporter fails).

#### Windows + WSL2

1. Create a directory on your Windows drive (e.g., `E:\GenAnimPipeline\mcp_server`).
2. Open a **Windows** Command Prompt, navigate to the folder, and set up the environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install fastmcp
   ```

> **Note for WSL2 users:** The MCP server runs on Windows so that Blender can access the GPU natively. The WSL2 orchestrator reaches it via the Windows host gateway IP, resolved at runtime by `get_windows_host_ip()` in `pipeline/shared.py`.

#### Native Linux

1. Navigate to `~/GenAnimPipeline/mcp_server` (or create it).
2. Set up the environment in a Linux terminal:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install fastmcp
   ```
3. In `translate_to_fbx.py`, update the configuration block to use your Linux Blender path:
   ```python
   BLENDER_EXE = "/usr/bin/blender"          # or full path from `which blender`
   BLENDER_SCRIPT = "/home/user/GenAnimPipeline/mcp_server/blender_retarget.py"
   SERVER_PORT = 8000
   ```
4. The server and orchestrator both run on the same machine, so `mcp.server_port` in `pipeline_config.json` points to `localhost:8000` — no host IP resolution needed.

> **Note for Linux users:** `pipeline/nodes/mcp.py` builds the server URL using `get_windows_host_ip()` under WSL2 and `localhost` on native Linux. If you're on native Linux and the MCP node fails to connect, confirm the server is running and that `mcp.server_port` in `pipeline_config.json` matches `SERVER_PORT` in `translate_to_fbx.py`.

---

*(Both environments continue below with the same server files.)*

1. In the `mcp_server` directory, set up the Python environment:
   ```bash
   # Windows: venv\Scripts\activate
   # Linux:   source venv/bin/activate
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

## Phase 7: The Master LangGraph Orchestrator

The pipeline entry point is `graph.py`, but it is now a thin orchestrator — it only wires nodes into the LangGraph graph and handles CLI arguments. All node logic lives in the `pipeline/` package.

> **WSL2 users:** run all commands below in your Ubuntu terminal with the `text2motion` conda environment active.
> **Native Linux users:** run them in any terminal in the same conda environment.

### Project Layout

```
GenAnimPipeline/
  graph.py                     # Entry point: LangGraph assembly + argparse
  pipeline_config.json         # All defaults — models, ports, paths, thresholds
  pipeline/
    shared.py                  # Config loader, logger, PipelineState, path helpers
    smplh_adapter.py           # SMPL-H NPZ adapter + FK helpers
    nodes/
      memory.py                # retrieve_context_node, update_memory_node
      planner.py               # generate_symbolic_plan_node + rewriter helpers
      motion.py                # generate_motion_node
      critic.py                # Stages 1-4, score_candidate, evaluate_motion_node
      mcp.py                   # execute_mcp_translation_node
```

### Configuration

All model and service settings live in `pipeline_config.json`. To swap any model or change any endpoint, edit only this file — no source code changes needed:

```json
{
  "default_prompt": "A character performs a heavy broadsword swing.",

  "animation": {
    "global_duration": 5,
    "motion_seed": 999,
    "motion_batch_size": 4
  },

  "ollama": {
    "port": 11434,
    "planner_model": "llama3:latest",
    "vision_model": "llava:latest"
  },

  "prompt_rewriter": "ollama",

  "motion_engine": {
    "active": "hy-motion",
    "hy-motion": {
      "dir": "HY-Motion-1.0",
      "checkpoint": "ckpts/tencent/HY-Motion-1.0",
      "script": "local_infer.py"
    },
    "kimodo": {
      "model": "soma-rp-v1",
      "diffusion_steps": 100,
      "cfg_weight": 7.5,
      "output_bvh": true
    }
  },

  "milvus": {
    "host": "localhost",
    "port": "19530",
    "collection": "motion_corrections"
  },

  "output": {
    "fbx_path": "final_animation.fbx",
    "bvh_path": "temp_motion.bvh",
    "npz_path": "temp_motion.npz"
  },

  "mcp": { "server_port": 8000 },

  "critic": {
    "score_threshold": 0.85,
    "max_iterations": 3
  },

  "logging": {
    "level": "DEBUG",
    "file": "pipeline.log"
  }
}
```

All output paths are relative to the project root. `pipeline/shared.py` resolves them to absolute paths at import time, so the pipeline works regardless of where the repo is cloned.

**Environment-specific notes for `pipeline_config.json`:**

| Setting | Windows + WSL2 | Native Linux |
|---|---|---|
| `motion_engine.hy-motion.dir` | Relative to project root — no change needed | Same |
| `milvus.host` | `"localhost"` — Docker Desktop bridges WSL2 | `"localhost"` |
| `mcp.server_port` | Must match `SERVER_PORT` in `mcp_server/translate_to_fbx.py` on Windows | Same, but server runs on Linux localhost |
| `ollama.port` | `11434` — Windows host IP resolved at runtime | `11434` on localhost |

> **WSL2 only:** `pipeline/shared.py` calls `get_windows_host_ip()` (via `ip route`) to reach Ollama and the MCP server running on the Windows host. This is automatic — no config change needed. On native Linux all services are on `localhost` and this helper is not invoked.

### Pipeline Flow

The LangGraph workflow connects five nodes in a directed graph with one conditional branch:

```
retrieve_context -> generate_plan -> generate_motion -> evaluate_motion
                         ^                                   |
                         |                                   v
                   update_memory <--(replan)-- route_evaluation --> [finish | execute_mcp]
                         |                                                       |
                         v                                                       v
                    generate_plan                                               END
```

| Node | Purpose |
|------|---------|
| `retrieve_context` | Connects to Milvus and loads historical motion correction embeddings. Degrades gracefully if offline. |
| `generate_plan` | Asks the planner LLM (via Ollama on the Windows host) to decompose the prompt into a Laban-style motion plan (action_type, spatial_level, speed). Falls back to a safe default plan if the LLM is unreachable. |
| `generate_motion` | Runs the motion diffusion transformer. Writes the prompt in the engine's expected format, invokes inference, then applies smart fallback logic: prefers native FBX, falls back to BVH, or logs an error if neither exists. Always copies the co-generated `.npz` file (SMPL-H pose data) to `temp_motion.npz` so the critic cascade can evaluate the motion via the `SmplhMocap` adapter. |
| `evaluate_motion` | **4-Stage Critic Cascade.** Runs sequentially -- an earlier failure skips heavier stages: **(1) Kinematic Gatekeeper** checks for foot sliding via forward kinematics (pure Python/numpy); **(2) Semantic Logician** sends a sampled joint trajectory JSON to Llama 3 to verify semantic intent; **(3) Art Director** renders 4 skeleton keyframes with matplotlib and sends the PNG to LLaVA for visual quality scoring; **(4) Human Sign-Off** prints a CLI block and waits for `Y`/`N` input. Requires `bvh`, `matplotlib`, `numpy` (WSL2) and `llava` pulled in Ollama (Windows). |
| `update_memory` | **(Continuous learning)** Intercepts every rejection before replanning. Encodes the critic feedback into a 384-dim vector using `all-MiniLM-L6-v2` (sentence-transformers) and inserts it into the `motion_corrections` Milvus collection so future runs can retrieve it as historical context. Skips gracefully if Milvus is offline. |
| `execute_mcp` | (Conditional) When only a BVH was produced, contacts the Windows FastMCP Blender server via SSE to convert it to FBX. |

### Key architectural features

* **`pipeline/shared.py`** — Loaded by every module. Reads `pipeline_config.json`, sets up the rotating logger, exposes `PipelineState`, `get_windows_host_ip()`, `log_input()`, `log_output()`, and derived path constants like `WIN_BVH_PATH`.
* **`PipelineState`** — TypedDict shared across all nodes. Each node reads what it needs and returns a partial dict to merge back into the shared state.
* **`route_evaluation()`** — Conditional router in `graph.py`: passes to finish (FBX), Blender fallback (BVH), or loops back to re-plan.
* **`pipeline/smplh_adapter.py`** — Runs standalone for FK/adapter testing: `python -m pipeline.smplh_adapter`.

### Standalone Module Testing

Each node module has a built-in test runner so stages can be developed and debugged without running the full pipeline:

```bash
python -m pipeline.smplh_adapter                        # FK math with synthetic NPZ
python -m pipeline.nodes.memory                         # Milvus retrieval (graceful if offline)
python -m pipeline.nodes.planner                        # Ollama planner or hymotion passthrough
python -m pipeline.nodes.planner --rewriter hymotion
python -m pipeline.nodes.critic                         # Stages 1-3 with synthetic NPZ
python -m pipeline.nodes.critic path/to/motion.npz      # Stages 1-3 with real file
python -m pipeline.nodes.motion                         # Dry-run: prints inference command
python -m pipeline.nodes.mcp                            # SSE connectivity check
```

---

## Daily Execution: Startup Sequence

Follow the sequence for your environment. All steps that are identical between environments are listed once.

---

### Windows + WSL2

Open **three separate terminal windows** — one Windows Command Prompt for services, one for the orchestrator (WSL2 Ubuntu), and optionally one for the dashboard.

**Terminal 1 — Windows Command Prompt (Ollama + MCP server)**

```
:: 1. Ensure Ollama is running in the system tray, then warm up the models:
ollama run llama3
ollama run llava

:: 2. Start the Blender MCP translation server (leave this running):
cd E:\GenAnimPipeline\mcp_server
venv\Scripts\activate
python translate_to_fbx.py
```

**Terminal 2 — Windows Command Prompt (Dashboard — Optional)**

```
cd E:\GenAnimPipeline\dashboard
python server.py
```

Open `http://localhost:8080` in your browser.

**Terminal 3 — WSL2 Ubuntu (Milvus + Orchestrator)**

```bash
# Start Milvus vector database
sudo docker compose up -d

# Activate the AI environment and run the pipeline
conda activate text2motion
cd /mnt/e/GenAnimPipeline
python graph.py
```

---

### Native Linux

Open **two terminal windows** — one for background services, one for the orchestrator.

**Terminal 1 — Background Services (Ollama + Milvus + MCP server)**

```bash
# 1. Ollama (if not already running as a systemd service)
ollama serve &

# 2. Warm up models
ollama run llama3
ollama run llava

# 3. Start Milvus vector database
sudo docker compose up -d

# 4. Start the Blender MCP translation server (leave this running)
cd ~/GenAnimPipeline/mcp_server
source venv/bin/activate
python translate_to_fbx.py
```

**Terminal 2 — Dashboard (Optional)**

```bash
cd ~/GenAnimPipeline/dashboard
python server.py
```

Open `http://localhost:8080` in your browser.

**Terminal 3 — Orchestrator**

```bash
conda activate text2motion
cd ~/GenAnimPipeline
python graph.py
```

---

**CLI arguments** — all are optional and override the corresponding `pipeline_config.json` value for that run only:

| Argument | Values | Default | Description |
|---|---|---|---|
| `--prompt "..."` | any string | `default_prompt` | Motion description to generate |
| `--batch-size N` | integer | `motion_batch_size` (4) | Candidates generated; best scorer is kept (HY-Motion only; Kimodo takes first) |
| `--engine` | `hy-motion` \| `kimodo` | `motion_engine.active` | Select motion generation backend for this run |
| `--rewriter` | `ollama` \| `hymotion` | `prompt_rewriter` | Prompt enrichment strategy. `ollama` uses Llama3 with Qwen3 fallback; `hymotion` delegates to HY-Motion's internal rewriter. Kimodo ignores `hymotion` and uses no rewriter |

Examples:
```bash
# Single candidate, fast iteration
python graph.py --batch-size 1

# Switch to Kimodo for one run without changing the config
python graph.py --engine kimodo --prompt "A person waves hello"

# Use HY-Motion's internal Qwen3 rewriter instead of Ollama
python graph.py --rewriter hymotion

# Full override
python graph.py --prompt "A character vaults over an obstacle" --batch-size 8 --engine hy-motion --rewriter ollama
```

---

## Quick Reference: Swapping Models & Configuration

| What to change | Config location | Key(s) |
|---|---|---|
| **Active motion engine** | `pipeline_config.json → motion_engine.active` or `--engine` CLI | `"hy-motion"` or `"kimodo"` |
| Kimodo model variant | `pipeline_config.json → motion_engine.kimodo` | `model` (e.g. `"soma-rp-v1"`, `"g1-rp-v1"`, `"smplx-rp-v1"`) |
| Kimodo quality vs speed | `pipeline_config.json → motion_engine.kimodo` | `diffusion_steps` (fewer = faster, lower quality) |
| HY-Motion checkpoint | `pipeline_config.json → motion_engine.hy-motion` | `checkpoint` (e.g. `"ckpts/tencent/HY-Motion-1.0-Lite"` for lower VRAM) |
| Planner LLM | `pipeline_config.json → ollama` | `planner_model`, `port` |
| Prompt rewriter | `pipeline_config.json` or `--rewriter` | `prompt_rewriter` (`"ollama"` or `"hymotion"`) |
| Motion batch size | `pipeline_config.json → animation` or `--batch-size` | `motion_batch_size` (default `4`; applies to HY-Motion only) |
| Critic vision model (Stage 3) | `pipeline_config.json → ollama` | `vision_model` (e.g. `"llava:13b"`, `"bakllava"`) |
| Critic score threshold | `pipeline_config.json → critic` | `score_threshold` (default `0.85`) |
| Vector database | `pipeline_config.json → milvus` | `host`, `port`, `collection` |
| Blender executable | `mcp_server/translate_to_fbx.py` | `BLENDER_EXE` (Windows: full `.exe` path; Linux: `/usr/bin/blender` or `which blender`) |
| MCP server port | `pipeline_config.json → mcp` **and** `translate_to_fbx.py` | `server_port` / `SERVER_PORT` — must match in both files |
