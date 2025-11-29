import os
import torch
import imageio
import trimesh

from pathlib import Path

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    HardPhongShader,
    BlendParams,
    PointLights
)
from pytorch3d.renderer.mesh import TexturesVertex

def normalize_mesh(mesh):
    verts = mesh.verts_packed()
    center = verts.mean(0)
    verts = verts - center
    scale = verts.norm(p=2, dim=1).max()
    verts = verts / scale
    return Meshes(verts=[verts], faces=[mesh.faces_packed()], textures=mesh.textures)

def setup_silhouette_renderer(image_size=512):
    raster_settings = RasterizationSettings(image_size=image_size)
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    return MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )

def setup_rgb_renderer(device, image_size=512):
    raster_settings = RasterizationSettings(image_size=image_size)
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    return MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights)
    )

def load_and_prepare_mesh(stl_path, device):
    tmp_obj_path = stl_path.replace(".stl", ".obj")
    tm = trimesh.load(stl_path)
    tm.export(tmp_obj_path)
    mesh = load_objs_as_meshes([tmp_obj_path], device=device)
    verts = mesh.verts_packed()
    mesh.textures = TexturesVertex(verts_features=torch.ones_like(verts)[None])
    mesh = normalize_mesh(mesh)
    return mesh, tmp_obj_path

def render_all_stl_to_masks_and_rgbs(stl_folder, output_root, views=4, dist=1.2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_root, exist_ok=True)

    silhouette_renderer = setup_silhouette_renderer()
    rgb_renderer = setup_rgb_renderer(device)

    stl_files = sorted([
        f for f in os.listdir(stl_folder)
        if f.endswith(".stl") and "_heart_ventricle_left" in f
    ])

    for stl_file in stl_files:
        name = os.path.splitext(stl_file)[0]
        input_path = os.path.join(stl_folder, stl_file)
        output_dir = os.path.join(output_root, name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            mesh, tmp_obj_path = load_and_prepare_mesh(input_path, device)

            for i in range(views):
                elev = 15 * i
                azim = 45 * i
                R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
                cameras = OpenGLOrthographicCameras(device=device, R=R, T=T)

                # mask
                rendered_sil = silhouette_renderer(mesh, cameras=cameras)[0]
                alpha = rendered_sil[..., 3].cpu().numpy()
                mask = (alpha * 255).astype('uint8')
                imageio.imwrite(os.path.join(output_dir, f"mask_{i:02d}.png"), mask)

                # rgb
                rendered_rgb = rgb_renderer(mesh, cameras=cameras)[0]
                rgb = rendered_rgb[..., :3].cpu().numpy()
                rgb_img = (rgb * 255).astype('uint8')
                imageio.imwrite(os.path.join(output_dir, f"rgb_{i:02d}.png"), rgb_img)

            os.remove(tmp_obj_path)
            print(f"[✓] Rendered {stl_file} to {output_dir}")
        except Exception as e:
            print(f"[✗] Failed on {stl_file}: {e}")

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    DATA_DIR = ROOT / "data"

    render_all_stl_to_masks_and_rgbs(
        stl_folder=DATA_DIR / "output_new",
        output_root=DATA_DIR / "rendered_masks",
        views=4,
        dist=1.2
    )
