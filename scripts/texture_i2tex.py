import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from mvadapter.pipelines.pipeline_texture import ModProcessConfig, TexturePipeline
from mvadapter.utils import make_image_grid
from mvadapter.utils.mesh_utils import (
    NVDiffRastContextWrapper,
    get_orthogonal_camera,
    load_mesh,
    render,
)

# MV-Adapter's 6 canonical camera positions
ELEVATION_DEG = [0, 0, 0, 0, 89.99, -89.99]
AZIMUTH_DEG   = [x - 90 for x in [0, 90, 180, 270, 180, 180]]
DISTANCE      = [1.8] * 6
VIEW_NAMES    = ["front", "right", "back", "left", "top", "bottom"]


def render_geometry_reference(mesh_path, height, width, device, save_dir, save_name):
    """Render greyscale normal-shaded reference views of the raw mesh geometry."""
    print("\n--- Rendering geometry reference views ---")

    ctx = NVDiffRastContextWrapper(device=device)
    mesh = load_mesh(mesh_path, rescale=True, device=device)
    cameras = get_orthogonal_camera(
        elevation_deg=ELEVATION_DEG,
        distance=DISTANCE,
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=AZIMUTH_DEG,
        device=device,
    )
    render_out = render(ctx, mesh, cameras, height=height, width=width,
                        render_attr=False, normal_background=0.0)

    ref_dir = os.path.join(save_dir, "geometry_reference")
    os.makedirs(ref_dir, exist_ok=True)

    grey_imgs = []
    for i in range(6):
        view_name = VIEW_NAMES[i]

        # Normal map [-1,1] -> [0,1], then greyscale
        normal = (render_out.normal[i] / 2 + 0.5).clamp(0, 1)  # (H, W, 3)
        grey = 0.299 * normal[..., 0] + 0.587 * normal[..., 1] + 0.114 * normal[..., 2]

        # White background where mesh is absent
        mask = (render_out.normal[i].abs().sum(dim=-1) > 1e-3).float()
        img_tensor = grey * mask + 1.0 * (1 - mask)
        img_tensor = img_tensor.unsqueeze(-1).repeat(1, 1, 3)  # (H, W, 3)

        arr = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(os.path.join(ref_dir, f"{save_name}_ref_{i:02d}_{view_name}.png"))
        grey_imgs.append(img)
        print(f"  View {i} ({view_name}): elev={ELEVATION_DEG[i]:.1f}  azim={AZIMUTH_DEG[i]:.1f}")

    make_image_grid(grey_imgs, rows=1).save(
        os.path.join(ref_dir, f"{save_name}_ref_grid.png")
    )
    print(f"Geometry reference renders saved to: {ref_dir}")
    print("-----------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--variant", type=str, default="sdxl", choices=["sdxl", "sd21"])
    # I/O
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, default="high quality")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--save_dir", type=str, default="./output")
    parser.add_argument("--save_name", type=str, default="i2tex_sample")
    # Extra
    parser.add_argument("--reference_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--preprocess_mesh", action="store_true")
    parser.add_argument("--remove_bg", action="store_true")
    parser.add_argument("--paintbrush_target_dir", type=str, default=None,
                        help="If set, also copy MV-Adapter views as {mesh_name}_v{i}.png here for paintbrush training")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.variant == "sdxl":
        from scripts.inference_ig2mv_sdxl import prepare_pipeline, remove_bg, run_pipeline
        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_model = "madebyollin/sdxl-vae-fp16-fix"
        height = width = 768
        uv_size = 4096
    elif args.variant == "sd21":
        from scripts.inference_ig2mv_sd import prepare_pipeline, remove_bg, run_pipeline
        base_model = "stabilityai/stable-diffusion-2-1-base"
        vae_model = None
        height = width = 512
        uv_size = 2048
    else:
        raise ValueError(f"Invalid variant: {args.variant}")

    device = args.device
    num_views = 6

    # Step 0: render raw geometry reference views
    render_geometry_reference(
        mesh_path=args.mesh,
        height=1024, width=1024,
        device=device,
        save_dir=args.save_dir,
        save_name=args.save_name,
    )

    # Step 1: load MV-Adapter pipeline
    pipe = prepare_pipeline(
        base_model=base_model,
        vae_model=vae_model,
        unet_model=None,
        lora_model=None,
        adapter_path="huanngzh/mv-adapter",
        scheduler=None,
        num_views=num_views,
        device=device,
        dtype=torch.float16,
    )

    remove_bg_fn = None
    if args.remove_bg:
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        birefnet.to(device)
        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        remove_bg_fn = lambda x: remove_bg(x, birefnet, transform_image, device)

    texture_pipe = TexturePipeline(
        upscaler_ckpt_path="./checkpoints/RealESRGAN_x2plus.pth",
        inpaint_ckpt_path="./checkpoints/big-lama.pt",
        device=device,
    )
    print("Pipeline ready.")

    # Step 2: run MV-Adapter to generate 6 multi-view images
    images, _, _, _ = run_pipeline(
        pipe,
        mesh_path=args.mesh,
        num_views=num_views,
        text=args.text,
        image=args.image,
        height=height,
        width=width,
        num_inference_steps=50,
        guidance_scale=3.0,
        seed=args.seed,
        reference_conditioning_scale=args.reference_conditioning_scale,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        device=device,
        remove_bg_fn=remove_bg_fn,
    )

    mv_path = os.path.join(args.save_dir, f"{args.save_name}.png")
    make_image_grid(images, rows=1).save(mv_path)
    for i, img in enumerate(images):
        img.save(os.path.join(args.save_dir, f"{args.save_name}_view_{i:02d}.png"))
        print(f"  Saved view {i}: {args.save_name}_view_{i:02d}.png")

    # Optionally copy views to paintbrush target directory
    if args.paintbrush_target_dir is not None:
        import shutil
        mesh_name = os.path.splitext(os.path.basename(args.mesh))[0]
        os.makedirs(args.paintbrush_target_dir, exist_ok=True)
        for i in range(num_views):
            src = os.path.join(args.save_dir, f"{args.save_name}_view_{i:02d}.png")
            dst = os.path.join(args.paintbrush_target_dir, f"{mesh_name}_v{i}.png")
            shutil.copy(src, dst)
            print(f"  Copied to paintbrush target: {dst}")

    torch.cuda.empty_cache()

    # Step 3: back-project and complete UV texture
    out = texture_pipe(
        mesh_path=args.mesh,
        save_dir=args.save_dir,
        save_name=args.save_name,
        uv_unwarp=True,
        preprocess_mesh=args.preprocess_mesh,
        uv_size=uv_size,
        rgb_path=mv_path,
        rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
        camera_azimuth_deg=AZIMUTH_DEG,
    )
    print(f"Output saved to {out.shaded_model_save_path}")