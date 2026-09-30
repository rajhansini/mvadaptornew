import trimesh
import numpy as np

# Absolute path to avoid the "string is not a file" error
obj_path = "/net/projects/ranalab/rajhansini/MV-Adapter-Fresh/outputs/sanity_check/conversiontoobj/demo_test_shaded.obj"

# Load with process=False to see the RAW data before Trimesh 'fixes' it
mesh = trimesh.load(obj_path, process=False)

print(f"--- DATA ANALYSIS ---")
print(f"Vertices (v):  {len(mesh.vertices)}")
print(f"Faces (f):     {len(mesh.faces)}")

if hasattr(mesh.visual, 'uv'):
    uv_count = len(mesh.visual.uv)
    print(f"UVs (vt):      {uv_count}")
    
    if uv_count != len(mesh.vertices):
        print("\n❌ MISMATCH DETECTED")
        print(f"You have {uv_count} UVs for {len(mesh.vertices)} vertices.")
        print("RESULT: The renderer is guessing which UV goes where = Camouflage.")
    else:
        print("\n✅ INDEX MATCH")
        print("The number of UVs matches vertices. The issue is likely a V-FLIP or image sampling.")
else:
    print("\n❌ NO UVs FOUND: The loader isn't seeing the 'vt' lines.")