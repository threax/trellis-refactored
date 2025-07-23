from typing import *

import numpy as np
import torch
import xatlas
from tqdm import tqdm
import pyvista as pv

from PIL import Image
import trimesh

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    TexturesUV,
)

import cv2
import csv

import os, csv, json, datetime
from pathlib import Path

from .fill_holes import _fill_holes

from functools import wraps
import inspect
from numbers import Number
import torch.nn.functional as F
from .render_utils import render_multiview

import math

def parametrize_mesh(vertices: np.array, faces: np.array):
    """
    Parametrize a mesh to a texture space, using xatlas.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
    """

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    vertices = vertices[vmapping]
    faces = indices

    return vertices, faces, uvs


def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = True,
    simplify_ratio: float = 0.9,
    verbose: bool = False,
):
    """
    Postprocess a mesh by simplifying, removing invisible faces, and removing isolated pieces.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        simplify (bool): Whether to simplify the mesh, using quadric edge collapse.
        simplify_ratio (float): Ratio of faces to keep after simplification.
        verbose (bool): Whether to print progress.
    """

    if verbose:
        tqdm.write(f'Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    # Simplify
    if simplify and simplify_ratio > 0:
        mesh = pv.PolyData(vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1))
        mesh = mesh.decimate(simplify_ratio, progress_bar=verbose)
        vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        if verbose:
            tqdm.write(f'After decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces')
        if verbose:
            tqdm.write(f'After remove invisible faces: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    return vertices, faces


# Blue represents raster ouput, green represents mask

def _centroid_bool(mask: torch.Tensor):
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None
    return torch.stack([ys.float().mean(), xs.float().mean()])  # (y,x)

def _principal_axis(mask: torch.Tensor, centre: torch.Tensor):
    """2-D PCA on binary blob → unit vector (dx,dy) in image coords."""
    ys, xs = torch.where(mask)
    if ys.numel() < 10:
        return None
    pts = torch.stack([xs.float() - centre[1],  # X
                       ys.float() - centre[0]], # Y
                      dim=1)
    cov = pts.T @ pts / (pts.shape[0] - 1)
    _, vecs = torch.linalg.eigh(cov)
    axis = vecs[:, 1]
    return axis / axis.norm()

class AlignmentDebugger:
    """
    Creates a new CSV + annotated overlay images that compare
    PyTorch3D's raster silhouette (`rast_vis`) against the
    Replicator mask (`mask`).
    """

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.csv  = open(root / "alignment_metrics.csv", "w", newline="")
        self.w    = csv.writer(self.csv)
        self.w.writerow([
            "view", "rast_px", "mask_px", "overlap_px",
            "dx_px", "dy_px", "axis_deg"
        ])

    def log_view(
        self,
        view_idx: int,
        rast_vis: torch.Tensor,   # H×W bool
        mask: torch.Tensor,       # H×W bool
        overlay_img: np.ndarray   # H×W×3 uint8, will be annotated
    ):
        # centroids
        c_rast = _centroid_bool(rast_vis)
        c_mask = _centroid_bool(mask)

        dx = dy = ang = float("nan")
        if c_rast is not None and c_mask is not None:
            dx = (c_rast[1] - c_mask[1]).item()
            dy = (c_rast[0] - c_mask[0]).item()

            # principal axes
            axis_r = _principal_axis(rast_vis, c_rast)
            axis_m = _principal_axis(mask,      c_mask)

            if axis_r is not None and axis_m is not None:
                dot_val  = (axis_r @ axis_m).clamp(-1.0, 1.0).item()
                cross_z  = axis_r[0] * axis_m[1] - axis_r[1] * axis_m[0]
                sign     = 1.0 if cross_z >= 0 else -1.0
                ang      = math.degrees(math.acos(dot_val)) * sign

                # draw arrows (blue=raster, green=mask)
                def _arrow(p, a, col):
                    q = (int(p[0] + 60 * a[0].item()),
                         int(p[1] + 60 * a[1].item()))
                    cv2.arrowedLine(overlay_img, p, q, col, 2, tipLength=0.15)

                p_r = (int(c_rast[1].item()), int(c_rast[0].item()))
                p_m = (int(c_mask[1].item()), int(c_mask[0].item()))
                cv2.circle(overlay_img, p_r, 3, (255,0,0), -1)  # blue dot
                cv2.circle(overlay_img, p_m, 3, (0,255,0), -1)  # green dot
                _arrow(p_r, axis_r, (255,0,0))
                _arrow(p_m, axis_m, (0,255,0))

        # CSV row
        self.w.writerow([
            view_idx,
            int(rast_vis.sum()), int(mask.sum()), int((rast_vis & mask).sum()),
            dx, dy, ang
        ])

        # write overlay
        cv2.imwrite(str(self.root / f"align_{view_idx:03d}.png"), overlay_img)

    def close(self):
        self.csv.close()

def bake_texture_and_return_mesh(
    app_rep,
    mesh,
    simplify: float = 0.90,
    texture_size: int = 1024,
    verbose: bool = False,
    debug: bool = False,
):
    """
    Texture-bake with optional per-view diagnostics.

    When `debug=True` the function writes, per view:

        overlay_NNN.png   blue=raster, green=mask
        camera_NNN.json   R/T + intrinsics
        pointcloud_NNN.ply mesh vertices in that view (first & last)

    plus run-wide artefacts:

        view_stats.csv
        uv_hits.png
        baked_texture.png
    """

    # ---------- 0. debug dirs ----------
    if debug:
        ts       = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dbg_root = Path("debug_bake") / ts
        dbg_root.mkdir(parents=True, exist_ok=True)

        align_dbg = AlignmentDebugger(dbg_root / "alignment")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 1. mesh ----------
    Vn, Fn = mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy()
    Vn, Fn = postprocess_mesh(Vn, Fn, simplify_ratio=simplify)
    Vn, Fn, UVn = parametrize_mesh(Vn, Fn)

    V  = torch.tensor(Vn,  dtype=torch.float32, device=dev)
    F  = torch.tensor(Fn,  dtype=torch.int64,   device=dev)
    UV = torch.tensor(UVn, dtype=torch.float32, device=dev)
    faces_uvs = F

    # ---------- 2. observations ----------
    imgs, extrs, intrs = render_multiview(app_rep, 1024, 1000)  # (B, H, W, 3)
    H = W = imgs[0].shape[0]

    obs = [torch.tensor(i/255.0, dtype=torch.float32, device=dev)
           for i in imgs]
    msk = [torch.tensor((i > 0).any(-1), dtype=torch.bool, device=dev)
           for i in imgs]

    # ---------- 3. empty-texture PyTorch3D mesh ----------
    tex0 = torch.zeros((1, texture_size, texture_size, 3),
                       dtype=torch.float32, device=dev)

    mesh_p3d = Meshes(
        verts=[V], faces=[F],
        textures=TexturesUV(maps=tex0.permute(0, 3, 1, 2),
                            faces_uvs=[faces_uvs],
                            verts_uvs=[UV])
    )

    rast = MeshRasterizer(
        raster_settings=RasterizationSettings(image_size=(H, W),
                                              faces_per_pixel=1)
    )

    tex_acc = torch.zeros_like(tex0[0])                  # RGB accumulator
    w_acc   = torch.zeros(texture_size, texture_size, 1, device=dev)  # weights

    # ---------- 4. main loop ----------
    views_ok = 0
    for k, (rgb, mask, Ex, K3) in enumerate(tqdm(zip(obs, msk, extrs, intrs),
                 total=len(obs), disable=not verbose)):
        
        R_cv = Ex[:3, :3].to(dev).float()   
        t_cv = Ex[:3,  3].to(dev).float()   

        R_cam = R_cv.T                      
        T_cam = t_cv                        
        
        # Keep original principal point (no flipping)
        fx, fy = K3[0, 0] * W, K3[1, 1] * H
        cx, cy = K3[0, 2] * W, K3[1, 2] * H
        
        cam = PerspectiveCameras(
            device=dev,
            in_ndc=False,
            R=R_cam[None],
            T=T_cam[None],
            focal_length=torch.tensor([[fx, fy]], device=dev),
            principal_point=torch.tensor([[cx, cy]], device=dev),
            image_size=torch.tensor([[H, W]], device=dev),
        )
        
        # Transform vertices (camera looks along positive Z)
        verts_cam = (R_cam @ V.T + T_cam[:, None]).T  
        z_vals = verts_cam[:, 2]
        print(f"View {k}: Z-range [{z_vals.min().item():.2f}, {z_vals.max().item():.2f}] "
            f"Front: {(z_vals > 0).sum()}/{len(z_vals)} vertices")

        # --- rotate mask & colour image by 180° (two 90° turns) ---
        mask = torch.rot90(mask, 2, (0, 1))   # dims=(Y,X)
        rgb  = torch.rot90(rgb,  2, (0, 1))

        # ---------- rasterise ----------
        frags = rast(mesh_p3d, cameras=cam)
        p2f   = frags.pix_to_face[0, ..., 0]   # (H, W)
        rast_vis = p2f >= 0
        olap     = rast_vis & mask

        # ---------- debug CSV / overlay ----------
        if debug:
            Z = verts_cam[:, 2]

            overlay = torch.zeros(H, W, 3, dtype=torch.uint8, device=dev)
            overlay[..., 0] = rast_vis.to(torch.uint8) * 255   # blue = raster
            overlay[..., 1] = mask.to(torch.uint8)      * 255   # green = mask

            
            # log for alignment debugger
            align_dbg.log_view(k, rast_vis, mask, overlay.cpu().numpy())

            if k in (0, len(obs) - 1):  # first & last for brevity
                trimesh.points.PointCloud(
                    verts_cam.cpu().numpy()
                ).export(dbg_root / f"pointcloud_{k:03d}.ply")


        if olap.sum() == 0:
            continue          # skip empty view

        views_ok += 1

        # ---------- texture splat ----------
        fi   = p2f[olap]                                   # (N,)
        bary = frags.bary_coords[0, ..., 0, :][olap]       # (N,3)
        uv   = (bary[..., None] * UV[faces_uvs[fi]]).sum(-2)
        uv[..., 1] = 1 - uv[..., 1]                        # flip V

        xy = (uv * (texture_size - 1)).long()
        u, v = xy[:, 0], xy[:, 1]

        tex_acc.index_put_((v, u), rgb[olap], accumulate=True)
        w_acc.index_put_((v, u),
                         torch.ones_like(v, dtype=torch.float32)[:, None],
                         accumulate=True)

    if debug:
        align_dbg.close()

    # ---------- sanity guards ----------
    if views_ok == 0:
        raise RuntimeError("All views had zero overlap.")

    if (w_acc > 0).sum() == 0:
        raise RuntimeError("Rasteriser saw faces but all UVs mapped outside [0,1].")

    # ---------- resolve & save ----------
    tex = tex_acc / w_acc.clamp(min=1e-6)
    tex_np = (tex.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    if debug:
        Image.fromarray(tex_np).save(dbg_root / "baked_texture.png")
        hits = (w_acc[..., 0] > 0).float().cpu().numpy()
        heat = cv2.applyColorMap((hits * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(str(dbg_root / "uv_hits.png"), heat)

    visual = trimesh.visual.TextureVisuals(
        uv=UV.cpu().numpy(),
        image=Image.fromarray(tex_np),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(tex_np))
    )

    up = np.array([[1, 0, 0],
                   [0, 0, 1],
                   [0,-1, 0]], np.float32)

    return trimesh.Trimesh((Vn @ up.T), Fn, visual=visual, process=False)

