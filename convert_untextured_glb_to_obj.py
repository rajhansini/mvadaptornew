import trimesh
import os

def blunt_convert(input_path, output_dir, output_name):
    # Load GLB (Read-only)
    mesh = trimesh.load(input_path, force='mesh')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define path
    base_dir = os.path.dirname(input_path)
    obj_path = os.path.join(output_dir, f"{output_name}.obj")

    # Export Geometry only
    mesh.export(obj_path)
    print(f"DONE: {obj_path}")

if __name__ == "__main__":
    # Change these paths
    target = "./assets/demo/ig2mv/1ccd5c1563ea4f5fb8152eac59dabd5c.glb"
    output_directory = "./untextured_obj"
    name = "1ccd5c1563ea4f5fb8152eac59dabd5c"
    blunt_convert(target, output_directory, name)