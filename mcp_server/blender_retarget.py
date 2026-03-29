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