"""
pipeline/nodes/mcp.py — BVH-to-FBX fallback via the Windows Blender MCP server.

Node:
  execute_mcp_translation_node — Node 5 (conditional): send a BVH file to the
  FastMCP Blender server running on Windows for headless FBX export.

Standalone usage (tests SSE connectivity without real animation data):
    python -m pipeline.nodes.mcp
"""

import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client

from pipeline.shared import (
    MCP_SERVER_PORT, WIN_BVH_PATH, WIN_FBX_PATH,
    PipelineState, get_windows_host_ip, logger, log_input, log_output,
)


def execute_mcp_translation_node(state: PipelineState) -> dict:
    """Node 5 (conditional): BVH-to-FBX via the Windows Blender MCP server.

    When HY-Motion only produces a .bvh file, this node calls the FastMCP
    translation server running natively on Windows. That server launches Blender
    headlessly to import the BVH, bake the animation, and export a game-engine-
    ready FBX. Communication uses Server-Sent Events (SSE) over the WSL2-to-
    Windows network bridge.

    Paths are Windows-native because Blender runs on the Windows side.
    """
    log_input("execute_mcp", ["final_asset_path", "critic_score", "critic_feedback"], state)
    logger.info("[execute_mcp] BVH detected — contacting Windows Blender MCP server...")
    logger.debug("[execute_mcp] BVH input  (Windows): %s", WIN_BVH_PATH)
    logger.debug("[execute_mcp] FBX output (Windows): %s", WIN_FBX_PATH)

    try:
        win_ip  = get_windows_host_ip()
        mcp_url = f"http://{win_ip}:{MCP_SERVER_PORT}/sse"
        logger.debug("[execute_mcp] MCP SSE endpoint: %s", mcp_url)

        async def run_mcp_tool():
            async with sse_client(mcp_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await session.call_tool(
                        "run_blender_export",
                        arguments={
                            "bvh_motion_path": WIN_BVH_PATH,
                            "output_fbx_path": WIN_FBX_PATH,
                        },
                    )

        mcp_result = asyncio.run(run_mcp_tool())
        logger.info("[execute_mcp] Blender server response: %s", mcp_result.content)

    except ExceptionGroup as eg:
        for exc in eg.exceptions:
            logger.error("[execute_mcp] Connection error: %s", exc)
    except Exception as e:
        logger.error("[execute_mcp] MCP tool call failed: %s", e)

    result = {"final_asset_path": WIN_FBX_PATH}
    log_output("execute_mcp", result)
    return result


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== execute_mcp_translation_node (connectivity check) ===")
    print(f"  MCP_SERVER_PORT : {MCP_SERVER_PORT}")
    print(f"  WIN_BVH_PATH    : {WIN_BVH_PATH}")
    print(f"  WIN_FBX_PATH    : {WIN_FBX_PATH}")

    try:
        win_ip = get_windows_host_ip()
        print(f"  Windows host IP : {win_ip}")
        mcp_url = f"http://{win_ip}:{MCP_SERVER_PORT}/sse"
        print(f"  MCP endpoint    : {mcp_url}")
    except Exception as e:
        print(f"  Could not resolve Windows host IP: {e}")
        win_ip = "localhost"
        mcp_url = f"http://{win_ip}:{MCP_SERVER_PORT}/sse"
        print(f"  Fallback endpoint: {mcp_url}")

    dummy_state: PipelineState = {
        "prompt": "A character performs a heavy broadsword swing.",
        "historical_context": [],
        "laban_plan": {},
        "motion_tensor_path": "",
        "critic_score": 0.85,
        "critic_feedback": "Art Director: poses look dynamic.",
        "iteration_count": 1,
        "final_asset_path": "/tmp/temp_motion.bvh",
    }

    print("\n  Calling node (will fail gracefully if Blender server is offline)...")
    result = execute_mcp_translation_node(dummy_state)
    print(f"  final_asset_path: {result['final_asset_path']}")
