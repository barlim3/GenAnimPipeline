"""
pipeline/shared.py — Shared infrastructure for the GenAnimPipeline.

Loads pipeline_config.json and exposes all config values as module-level
constants, initialises the structured logger, and defines the PipelineState
TypedDict that flows through every LangGraph node.

Any module inside this package can do:
    from pipeline.shared import cfg, logger, PipelineState, SCRIPT_DIR
"""

import json
import logging
import os
import subprocess
from logging.handlers import RotatingFileHandler
from typing import Any, TypedDict

# ── Project Root ─────────────────────────────────────────────────────────────
# All relative paths are resolved from here so the pipeline works regardless
# of where the repo is cloned.
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Config Loader ─────────────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(SCRIPT_DIR, "pipeline_config.json")

def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)

cfg = _load_config()

# ── Flat config accessors (mirrors the old module-level constants) ────────────

DEFAULT_PROMPT: str        = cfg["default_prompt"]
GLOBAL_DURATION: int       = cfg["animation"]["global_duration"]
MOTION_SEED: int           = cfg["animation"]["motion_seed"]
MOTION_BATCH_SIZE: int     = cfg["animation"]["motion_batch_size"]

PLANNER_MODEL: str         = cfg["ollama"]["planner_model"]
VISION_MODEL: str          = cfg["ollama"]["vision_model"]
OLLAMA_PORT: int           = cfg["ollama"]["port"]

PROMPT_REWRITER: str       = cfg["prompt_rewriter"]

# ── Motion engine selection ───────────────────────────────────────────────────
# Set active_engine to "hy-motion" or "kimodo" in pipeline_config.json, or
# override at runtime via graph.py --engine.  All engine-specific constants are
# read here so every module stays in sync after a CLI override.

ACTIVE_ENGINE: str         = cfg["motion_engine"]["active"]   # "hy-motion" | "kimodo"

_hym = cfg["motion_engine"]["hy-motion"]
MOTION_ENGINE_DIR: str        = os.path.join(SCRIPT_DIR, _hym["dir"])
MOTION_ENGINE_CHECKPOINT: str = _hym["checkpoint"]
MOTION_ENGINE_SCRIPT: str     = _hym["script"]

_kim = cfg["motion_engine"]["kimodo"]
KIMODO_MODEL: str          = _kim["model"]
KIMODO_DIFFUSION_STEPS: int = int(_kim["diffusion_steps"])
KIMODO_CFG_WEIGHT: float   = float(_kim["cfg_weight"])
KIMODO_OUTPUT_BVH: bool    = bool(_kim["output_bvh"])

MILVUS_HOST: str           = cfg["milvus"]["host"]
MILVUS_PORT: str           = cfg["milvus"]["port"]
MILVUS_COLLECTION: str     = cfg["milvus"]["collection"]

OUTPUT_FBX_PATH: str       = os.path.join(SCRIPT_DIR, cfg["output"]["fbx_path"])
OUTPUT_BVH_PATH: str       = os.path.join(SCRIPT_DIR, cfg["output"]["bvh_path"])
OUTPUT_NPZ_PATH: str       = os.path.join(SCRIPT_DIR, cfg["output"]["npz_path"])

MCP_SERVER_PORT: int       = cfg["mcp"]["server_port"]

CRITIC_SCORE_THRESHOLD: float = cfg["critic"]["score_threshold"]
MAX_ITERATIONS: int           = cfg["critic"]["max_iterations"]

_LOG_LEVEL_STR: str = cfg["logging"]["level"]
LOG_LEVEL: int      = getattr(logging, _LOG_LEVEL_STR.upper(), logging.DEBUG)
LOG_FILE: str       = os.path.join(SCRIPT_DIR, cfg["logging"]["file"])

# ── WSL2 / Windows path helpers ───────────────────────────────────────────────

def _wsl_to_win(wsl_path: str) -> str:
    """Convert a WSL2 /mnt/X/... path to its Windows equivalent X:\\..."""
    if wsl_path.startswith("/mnt/") and len(wsl_path) > 6:
        drive = wsl_path[5].upper()
        rest = wsl_path[6:].replace("/", "\\")
        return f"{drive}:{rest}"
    return wsl_path.replace("/", "\\")


WIN_BVH_PATH: str = _wsl_to_win(OUTPUT_BVH_PATH)
WIN_FBX_PATH: str = _wsl_to_win(OUTPUT_FBX_PATH)


def get_windows_host_ip() -> str:
    """Resolve the Windows host IP from inside WSL2 via the default gateway."""
    return subprocess.check_output(
        "ip route list default | awk '{print $3}'", shell=True
    ).decode().strip()

# ── Logger ─────────────────────────────────────────────────────────────────────

_LOG_FMT      = "[%(asctime)s] [%(levelname)-8s] %(message)s"
_LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("pipeline")
logger.setLevel(LOG_LEVEL)

if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter(_LOG_FMT, _LOG_DATE_FMT))

    _file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _file_handler.setFormatter(logging.Formatter(_LOG_FMT, _LOG_DATE_FMT))

    logger.addHandler(_console_handler)
    logger.addHandler(_file_handler)

# ── I/O trace helpers ─────────────────────────────────────────────────────────

def _fmt_val(v: Any) -> str:
    """Truncate long values so I/O dumps stay readable in the log."""
    s = repr(v)
    return s if len(s) <= 200 else s[:197] + "..."


def log_input(node: str, keys: list, state: dict) -> None:
    logger.debug("[%s] ── INPUT  ─────────────────────────────────────", node)
    for k in keys:
        logger.debug("[%s]   %-22s = %s", node, k, _fmt_val(state.get(k)))


def log_output(node: str, result: dict) -> None:
    logger.debug("[%s] ── OUTPUT ─────────────────────────────────────", node)
    for k, v in result.items():
        logger.debug("[%s]   %-22s = %s", node, k, _fmt_val(v))

# ── LangGraph pipeline state ──────────────────────────────────────────────────

class PipelineState(TypedDict):
    prompt: str                 # User's natural-language motion description
    historical_context: list    # Past correction rules retrieved from Milvus
    laban_plan: dict            # Symbolic motion plan (action_type, spatial_level, speed)
    motion_tensor_path: str     # (Reserved) path to raw motion tensor if needed
    critic_score: float         # Quality score from the evaluation node (0.0–1.0)
    critic_feedback: str        # Human-readable feedback from the critic
    iteration_count: int        # Number of plan-generate-evaluate loops completed
    final_asset_path: str       # Absolute path to the output file (.fbx or .bvh)

# ── ANSI terminal helpers ─────────────────────────────────────────────────────

ANSI_RED   = "\033[91m"
ANSI_BOLD  = "\033[1m"
ANSI_RESET = "\033[0m"

# ── Pass duration to HY-Motion's patched prompt_rewrite.py ───────────────────
# Only meaningful when the active engine is hy-motion, but harmless to always set.
os.environ["HY_MOTION_DURATION"] = str(GLOBAL_DURATION)
