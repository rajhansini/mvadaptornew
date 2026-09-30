import trimesh

def check_materials(path):
    # Load the GLB
    scene = trimesh.load(path)
    
    # trimesh.load can return a Scene or a single Trimesh
    if isinstance(scene, trimesh.Scene):
        geometries = scene.geometry.values()
    else:
        geometries = [scene]

    unique_material_names = set()

    for mesh in geometries:
        # Check if the mesh has a visual material attribute
        if hasattr(mesh.visual, 'material'):
            # The material itself has a name, or we check the object ID
            mat = mesh.visual.material
            # Some materials don't have a string name, so we use their hash/id
            name = getattr(mat, 'name', None) or str(id(mat))
            unique_material_names.add(name)

    print("-" * 30)
    print(f"File: {path}")
    print(f"Total Unique Materials Found: {len(unique_material_names)}")
    print(f"Material List: {unique_material_names}")
    
    if len(unique_material_names) <= 1:
        print("✅ RESULT: Single Material. Your 'material_0' OBJ script is SAFE.")
    else:
        print("❌ RESULT: Multiple Materials. Your current OBJ script will OVERWRITE/BREAK textures.")
    print("-" * 30)

if __name__ == "__main__":
    glb_path = "/net/projects/ranalab/rajhansini/MV-Adapter-Fresh/outputs/sanity_check/demo_test_shaded.glb"
    check_materials(glb_path)