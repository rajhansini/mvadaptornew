import trimesh
from pygltflib import GLTF2
import os
import sys

def absolute_safe_extraction(frame_folder):
    # Dynamically build paths based on the frame folder provided
    base_path = f"/net/projects/ranalab/rajhansini/MV-Adapter-Fresh/outputs/monster_distillation_from_video_model/monster_lava_luma_ray_3.14/{frame_folder}"
    glb_path = os.path.join(base_path, f"{frame_folder}_textured_shaded.glb")
    out_dir = os.path.join(base_path, "obj_information")
    
    if not os.path.exists(glb_path):
        print(f"Skipping {frame_folder}: GLB not found.")
        return

    os.makedirs(out_dir, exist_ok=True)
    
    base_name = f"{frame_folder}_textured_shaded"
    obj_out = os.path.join(out_dir, f"{base_name}.obj")
    mtl_out = os.path.join(out_dir, f"{base_name}.mtl")
    tex_out = os.path.join(out_dir, f"{base_name}.png")

    # --- YOUR ORIGINAL LOGIC START ---
    scene = trimesh.load(glb_path, process=False)
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    
    with open(obj_out, 'w') as f:
        f.write(f"mtllib {base_name}.mtl\n")
        f.write(f"usemtl material_0\n")
        f.write(trimesh.exchange.obj.export_obj(mesh, include_texture=True, include_normals=True))

    with open(mtl_out, 'w') as f:
        f.write("newmtl material_0\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write(f"map_Kd {base_name}.png\n") 

    gltf = GLTF2.load(glb_path)
    blob = gltf.binary_blob()
    img_meta = gltf.images[0]
    bv = gltf.bufferViews[img_meta.bufferView]
    
    with open(tex_out, "wb") as f:
        f.write(blob[bv.byteOffset : bv.byteOffset + bv.byteLength])
    # --- YOUR ORIGINAL LOGIC END ---

    print(f"✅ SUCCESS: {frame_folder} processed.")

if __name__ == "__main__":
    # Get the frame folder from the command line argument
    absolute_safe_extraction(sys.argv[1])