"""
Simple GLB renderer that matches MV-Adapter camera orientation exactly.
Uses MV-Adapter's load_mesh function for 100% consistency.
"""
import torch
import torchvision
from pathlib import Path
import sys
import argparse

# Add MV-Adapter to path
sys.path.insert(0, '/net/projects/ranalab/rajhansini/MV-Adapter-Fresh')
sys.path.insert(0, '/net/projects/ranalab/rajhansini/dynamic_texture/src')

from mvadapter.utils.mesh_utils.mesh import load_mesh
from models.render import Renderer


def generate_mvadapter_renders(mesh_path: str, output_dir: str = "./reference_renders", render_size: int = 1024):
    """
    Generate 6 reference renders matching MV-Adapter's exact camera configuration.
    Uses MV-Adapter's load_mesh for perfect consistency.

    Args:
        mesh_path: Path to .glb or .obj file
        output_dir: Directory to save renders
        render_size: Resolution of rendered images
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup output directory
    mesh_name = Path(mesh_path).stem
    output_path = Path(output_dir) / mesh_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading mesh: {mesh_name}")

    # Load mesh using MV-Adapter's load_mesh function (same as MV-Adapter uses)
    textured_mesh = load_mesh(mesh_path, rescale=True, device=device)

    # Extract vertices and create simple Mesh-like object
    class SimpleMesh:
        def __init__(self, vertices, faces):
            self.vertices = vertices
            self.faces = faces

    mesh = SimpleMesh(
        vertices=textured_mesh.v_pos,
        faces=textured_mesh.t_pos_idx
    )

    print(f"Mesh loaded: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

    # Initialize renderer
    renderer = Renderer(device=device, dim=(render_size, render_size))

    # Match MV-Adapter's exact camera configuration
    # From inference_ig2mv_sdxl.py:
    # elevation_deg=[0, 0, 0, 0, 89.99, -89.99]
    # azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]]
    elevations = torch.tensor([0.0, 0.0, 0.0, 0.0, 89.99, -89.99], device=device)
    azimuths = torch.tensor([270.0, 0.0, 90.0, 180.0, 90.0, 90.0], device=device)
    view_names = ["left", "front", "right", "back", "top", "bottom"]

    radius = torch.tensor([1.8], device=device)  # MV-Adapter uses 1.8

    # Create uniform gray vertex colors
    num_vertices = mesh.vertices.shape[0]
    # Change this in your script to use the GLB's actual colors:
    if hasattr(textured_mesh, 'v_rgb') and textured_mesh.v_rgb is not None:
        vertex_colors = textured_mesh.v_rgb
    else:
        # Fallback to gray only if the mesh has no colors
        vertex_colors = torch.tensor([0.7, 0.7, 0.7], device=device).unsqueeze(0).repeat(num_vertices, 1)

    print(f"Generating 6 orthogonal reference renders at {render_size}x{render_size}...")
    print("Camera configuration matches MV-Adapter exactly")
    print(f"Using MV-Adapter's load_mesh for consistent normalization")

    # Render each view
    for i, (elev, azim, view_name) in enumerate(zip(elevations, azimuths, view_names)):
        # Render mesh with vertex colors
        pred_features, mask = renderer.render_mesh(
            mesh,
            vertex_colors,
            elev.unsqueeze(0),
            azim.unsqueeze(0),
            radius,
            look_at_height=0.0  # MV-Adapter centers at 0 after normalization
        )

        # Composite with white background
        mask = mask.detach()
        pred_map = (pred_features * mask) + (1 * (1 - mask))

        # Save render
        save_path = output_path / f"{mesh_name}_v{i}_{view_name}_ref.png"
        torchvision.utils.save_image(pred_map, save_path)
        print(f"  Saved v{i} {view_name} (elev={elev:.2f}°, azim={azim:.2f}°): {save_path}")

    print(f"✓ All renders saved to {output_path}")
    print(f"  Files: {mesh_name}_v0_left_ref.png through {mesh_name}_v5_bottom_ref.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 6 orthogonal renders matching MV-Adapter exactly")
    parser.add_argument("--mesh", type=str, required=True, help="Path to .glb or .obj file")
    parser.add_argument("--output", type=str, default="./reference_renders", help="Output directory")
    parser.add_argument("--size", type=int, default=1024, help="Render resolution")

    args = parser.parse_args()

    generate_mvadapter_renders(args.mesh, args.output, args.size)