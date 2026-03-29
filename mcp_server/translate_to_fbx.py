"""
translate_to_fbx.py - FastMCP Blender Translation Server (Windows Native)

Runs natively on the Windows host as a safety-net service for the pipeline.
When HY-Motion produces only a .bvh file (raw skeleton data), the LangGraph
orchestrator (graph.py) calls this server over SSE to convert it into a
game-engine-ready .fbx via headless Blender.

Start this server BEFORE running the pipeline:
    cd E:\\GenAnimPipeline\\mcp_server
    venv\\Scripts\\activate
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