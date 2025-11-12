#!/usr/bin/env python3
"""
Visualize MeshCoder-generated Blender code locally.
Uses minimal bpy_lib (no torch/heavy dependencies).

Usage:
    # Visualize in Blender GUI:
    /Applications/Blender.app/Contents/MacOS/Blender --python visualize_meshcoder_local.py -- code.py

    # Export to GLB/OBJ (headless):
    /Applications/Blender.app/Contents/MacOS/Blender --background --python visualize_meshcoder_local.py -- code.py output.glb
"""

import sys
import os
import json

def main():
    # Must import bpy AFTER Blender starts
    import bpy

    # Add bpy_lib_minimal to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Import our minimal bpy_lib functions into globals
    import bpy_lib_minimal
    for name in dir(bpy_lib_minimal):
        if not name.startswith('_'):
            globals()[name] = getattr(bpy_lib_minimal, name)

    # Parse arguments
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:] if len(sys.argv) > 1 else []

    if len(argv) < 1:
        print("Usage:")
        print("  Visualize: blender --python visualize_meshcoder_local.py -- code.py [--no-yup]")
        print("  Export:    blender --background --python visualize_meshcoder_local.py -- code.py output.glb [--no-yup]")
        print("")
        print("Options:")
        print("  --no-yup   Disable orientation correction (keeps original, usually wrong)")
        print("")
        print("Note: By default, objects are rotated 90° to appear upright (MeshCoder outputs on their back)")
        sys.exit(1)

    input_file = os.path.abspath(argv[0])
    output_file = os.path.abspath(argv[1]) if len(argv) > 1 and not argv[1].startswith('--') else None
    export_mode = output_file is not None

    # Check for flags
    # Note: By default we DO convert to Y-up (MeshCoder output is oriented wrong otherwise)
    convert_to_yup = '--no-yup' not in argv

    print(f"\n{'='*60}")
    print(f"MeshCoder Local Visualization")
    print(f"{'='*60}")
    print(f"Input:  {input_file}")
    if export_mode:
        print(f"Output: {output_file}")
        print(f"Orient: {'Corrected (Y-up conversion)' if convert_to_yup else 'Original (likely wrong)'}")
    else:
        print(f"Mode:   Interactive (will open in Blender)")
        print(f"Orient: {'Corrected (90° rotation)' if convert_to_yup else 'Original (likely wrong)'}")
    print()

    # Read the generated code
    with open(input_file, 'r') as f:
        data = json.load(f)
        code = data['code']

    print(f"✓ Code loaded (success={data.get('success')})")
    print(f"✓ Generated from {data.get('num_points_used')} points")

    # Replace the import statement with a comment (we already imported functions into globals)
    code = code.replace('from bpy_lib import *', '# bpy_lib functions loaded')

    # Execute the code
    print(f"\n{'='*60}")
    print("Executing generated code...")
    print(f"{'='*60}\n")

    try:
        # Execute in global namespace so bpy_lib functions are available
        exec(code, globals())
        print("✓ Code executed successfully!")

        # Count what was created
        mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        curve_objects = [obj for obj in bpy.data.objects if obj.type == 'CURVE']

        print(f"✓ Created {len(mesh_objects)} mesh objects")
        if curve_objects:
            print(f"✓ Created {len(curve_objects)} curve objects")

        # Calculate stats
        total_verts = sum(len(obj.data.vertices) for obj in mesh_objects)
        total_faces = sum(len(obj.data.polygons) for obj in mesh_objects)
        print(f"✓ Total: {total_verts} vertices, {total_faces} faces")

        # Apply orientation correction if requested (for interactive mode)
        if not export_mode and convert_to_yup:
            import math
            print(f"\n✓ Applying orientation correction (rotating all objects 90° around X)")
            for obj in bpy.data.objects:
                # Rotate 90 degrees around X axis to convert Z-up to Y-up
                obj.rotation_euler[0] += math.radians(90)
            print(f"  (Objects rotated to appear upright in Blender)")

    except Exception as e:
        print(f"\n✗ Error executing code:")
        print(f"  {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Export if requested (headless mode)
    if export_mode:
        print(f"\n{'='*60}")
        print("Exporting...")
        print(f"{'='*60}\n")

        try:
            # Export to GLB with optional Y-up conversion
            # Blender uses Z-up, but most game engines/viewers use Y-up
            # Setting export_yup=True converts the coordinate system automatically
            bpy.ops.export_scene.gltf(
                filepath=output_file,
                export_format='GLB',
                use_selection=False,
                export_apply=True,
                export_yup=convert_to_yup  # Convert Z-up to Y-up for compatibility
            )
            print(f"✓ Exported to: {output_file}")
            if convert_to_yup:
                print(f"  (orientation corrected for Y-up viewers)")
            else:
                print(f"  (keeping original orientation - may appear wrong)")

            # Also try OBJ
            if output_file.endswith('.glb'):
                obj_file = output_file.replace('.glb', '.obj')
                try:
                    bpy.ops.wm.obj_export(
                        filepath=obj_file,
                        export_selected_objects=False,
                        export_materials=False
                    )
                    print(f"✓ Also exported to: {obj_file}")
                except:
                    pass

            print(f"\n{'='*60}")
            print("Success! View your model:")
            print(f"  • Blender:  /Applications/Blender.app/Contents/MacOS/Blender {output_file}")
            print(f"  • Online:   https://gltf-viewer.donmccurdy.com/")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"✗ Error exporting: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("Opening in Blender GUI...")
        print("(The scene is now loaded - you can interact with it)")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
