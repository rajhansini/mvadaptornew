import trimesh
from pygltflib import GLTF2
import os

def absolute_safe_extraction():
    glb_path = "/net/projects/ranalab/rajhansini/MV-Adapter-Fresh/outputs/monster_distillation_from_video_model/monster_lava_luma_ray_3.14/frame_0028/frame_0028_textured_shaded.glb"
    out_dir = "/net/projects/ranalab/rajhansini/MV-Adapter-Fresh/outputs/monster_distillation_from_video_model/monster_lava_luma_ray_3.14/frame_0028/obj_information"
    os.makedirs(out_dir, exist_ok=True) # Ensure dir exists
    
    base_name = os.path.splitext(os.path.basename(glb_path))[0]
    obj_out = os.path.join(out_dir, f"{base_name}.obj")
    mtl_out = os.path.join(out_dir, f"{base_name}.mtl")
    tex_out = os.path.join(out_dir, f"{base_name}.png")

    # 1. GEOMETRY: bake transformations to prevent "shattered" parts
    scene = trimesh.load(glb_path, process=False)
    # concatenate=True ensures multiple parts are merged WITH their correct positions
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    
    # Export without trimesh's automatic (and often broken) MTL logic
    mesh.export(obj_out, include_texture=True) 

    # 2. FIXED OBJ/MTL LINK (More robust than manual string injection)
    with open(obj_out, 'w') as f:
        f.write(f"mtllib {base_name}.mtl\n")
        f.write(f"usemtl material_0\n")
        # Export actual geometry data
        f.write(trimesh.exchange.obj.export_obj(mesh, include_texture=True, include_normals=True))

    # 3. CREATE MTL
    with open(mtl_out, 'w') as f:
        f.write("newmtl material_0\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n") # Disable specular highlights for "flat" look
        f.write(f"map_Kd {base_name}.png\n") 

    # 4. BIT-PERFECT TEXTURE RIP (Unchanged, this part was solid)
    gltf = GLTF2.load(glb_path)
    blob = gltf.binary_blob()
    img_meta = gltf.images[0]
    bv = gltf.bufferViews[img_meta.bufferView]
    
    with open(tex_out, "wb") as f:
        f.write(blob[bv.byteOffset : bv.byteOffset + bv.byteLength])

    print(f"✅ SUCCESS: {base_name}.obj is hard-locked to {base_name}.png")

if __name__ == "__main__":
    absolute_safe_extraction()