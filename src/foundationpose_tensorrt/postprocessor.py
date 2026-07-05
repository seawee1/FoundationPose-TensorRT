# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions to be used for FoundationPose."""

import torch
import trimesh
import kornia
import cv2
import warp as wp
import open3d as o3d
import numpy as np
import nvdiffrast.torch as dr
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional


def depth2xyzmap(depth, K, uvs=None):
    """Convert the depth map to point cloud"""
    invalid_mask = depth < 0.1
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(
            np.arange(0, H), np.arange(0, W), sparse=False, indexing="ij"
        )
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map


wp.init()


@wp.kernel(enable_backward=False)
def bilateral_filter_depth_kernel(
    depth: wp.array(dtype=float, ndim=2),
    out: wp.array(dtype=float, ndim=2),
    radius: int,
    zfar: float,
    sigmaD: float,
    sigmaR: float,
):
    """Filter the depth map"""
    h, w = wp.tid()
    H = depth.shape[0]
    W = depth.shape[1]
    if w >= W or h >= H:
        return
    out[h, w] = 0.0
    mean_depth = float(0.0)
    num_valid = int(0)
    for u in range(w - radius, w + radius + 1):
        if u < 0 or u >= W:
            continue
        for v in range(h - radius, h + radius + 1):
            if v < 0 or v >= H:
                continue
            cur_depth = depth[v, u]
            if cur_depth >= 0.1 and cur_depth < zfar:
                num_valid += 1
                mean_depth += cur_depth
    if num_valid == 0:
        return
    mean_depth /= float(num_valid)

    depthCenter = depth[h, w]
    sum_weight = float(0.0)
    sum = float(0.0)
    for u in range(w - radius, w + radius + 1):
        if u < 0 or u >= W:
            continue
        for v in range(h - radius, h + radius + 1):
            if v < 0 or v >= H:
                continue
            cur_depth = depth[v, u]
            if (
                cur_depth >= 0.1
                and cur_depth < zfar
                and abs(cur_depth - mean_depth) < 0.01
            ):
                weight = wp.exp(
                    -float((u - w) * (u - w) + (h - v) * (h - v))
                    / (2.0 * sigmaD * sigmaD)
                    - (depthCenter - cur_depth)
                    * (depthCenter - cur_depth)
                    / (2.0 * sigmaR * sigmaR)
                )
                sum_weight += weight
                sum += weight * cur_depth
    if sum_weight > 0 and num_valid > 0:
        out[h, w] = sum / sum_weight


def bilateral_filter_depth(
    depth, radius=2, zfar=100, sigmaD=2, sigmaR=100000, device="cuda"
):
    """Launch the filter kernel with warp"""
    if isinstance(depth, np.ndarray):
        depth_wp = wp.array(depth, dtype=float, device=device)
    else:
        depth_wp = wp.from_torch(depth)
    out_wp = wp.zeros(depth.shape, dtype=float, device=device)
    wp.launch(
        kernel=bilateral_filter_depth_kernel,
        device=device,
        dim=[depth.shape[0], depth.shape[1]],
        inputs=[depth_wp, out_wp, radius, zfar, sigmaD, sigmaR],
    )
    depth_out = wp.to_torch(out_wp)

    if isinstance(depth, np.ndarray):
        depth_out = depth_out.data.cpu().numpy()
    return depth_out


def sample_views_icosphere(n_views, subdivisions=None, radius=1):
    """Initialize the pose hypothesis"""
    if subdivisions is not None:
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    else:
        subdivision = 1
        while 1:
            mesh = trimesh.creation.icosphere(subdivisions=subdivision, radius=radius)
            if mesh.vertices.shape[0] >= n_views:
                break
            subdivision += 1
    cam_in_obs = np.tile(np.eye(4)[None], (len(mesh.vertices), 1, 1))
    cam_in_obs[:, :3, 3] = mesh.vertices
    up = np.array([0, 0, 1])
    z_axis = -cam_in_obs[:, :3, 3]
    z_axis /= np.linalg.norm(z_axis, axis=-1).reshape(-1, 1)
    x_axis = np.cross(up.reshape(1, 3), z_axis)
    invalid = (x_axis == 0).all(axis=-1)
    x_axis[invalid] = [1, 0, 0]
    x_axis /= np.linalg.norm(x_axis, axis=-1).reshape(-1, 1)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=-1).reshape(-1, 1)
    cam_in_obs[:, :3, 0] = x_axis
    cam_in_obs[:, :3, 1] = y_axis
    cam_in_obs[:, :3, 2] = z_axis
    return cam_in_obs


def compute_mesh_diameter(model_pts=None, n_sample=1000):
    """
    Compute the mesh diameter using the model points
    """
    if n_sample is None:
        pts = model_pts
    else:
        ids = np.random.choice(
            len(model_pts), size=min(n_sample, len(model_pts)), replace=False
        )
        pts = model_pts[ids]
    dists = np.linalg.norm(pts[None] - pts[:, None], axis=-1)
    diameter = dists.max()
    return diameter


def toOpen3dCloud(points, normals=None):
    """
    Convert the points to 3D point cloud
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def make_mesh_tensors(mesh, device="cuda"):
    """
    Build the mesh in torch.tensor
    """
    mesh_tensors = {}
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        img = np.array(mesh.visual.material.image.convert("RGB"))
        img = img[..., :3]
        mesh_tensors["tex"] = (
            torch.as_tensor(img, device=device, dtype=torch.float)[None] / 255.0
        )
        mesh_tensors["uv_idx"] = torch.as_tensor(
            mesh.faces, device=device, dtype=torch.int
        )
        uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
        uv[:, 1] = 1 - uv[:, 1]
        mesh_tensors["uv"] = uv
    else:
        if mesh.visual.vertex_colors is None:
            mesh.visual.vertex_colors = np.tile(
                np.array([128, 128, 128]).reshape(1, 3), (len(mesh.vertices), 1)
            )
        mesh_tensors["vertex_color"] = (
            torch.as_tensor(
                mesh.visual.vertex_colors[..., :3], device=device, dtype=torch.float
            )
            / 255.0
        )

    mesh_tensors.update(
        {
            "pos": torch.tensor(mesh.vertices, device=device, dtype=torch.float),
            "faces": torch.tensor(mesh.faces, device=device, dtype=torch.int),
            "vnormals": torch.tensor(
                mesh.vertex_normals, device=device, dtype=torch.float
            ),
        }
    )
    return mesh_tensors


def reset_object(model_pts, model_normals, mesh=None, device="cuda"):
    """
    Reset the mesh hyper-parameters
    """
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    model_center = (min_xyz + max_xyz) / 2

    if mesh is not None:
        mesh = mesh.copy()
        mesh.vertices = mesh.vertices - model_center.reshape(1, 3)

    model_pts = mesh.vertices
    diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
    vox_size = max(diameter / 20.0, 0.003)
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(vox_size)

    max_xyz = np.asarray(pcd.points).max(axis=0)
    min_xyz = np.asarray(pcd.points).min(axis=0)

    mesh_tensors = make_mesh_tensors(mesh, device)

    return mesh, mesh_tensors, diameter, model_center


def compute_tf_batch(left, right, top, bottom, out_size):
    """Calculate the pose location"""
    B = len(left)
    left = left.round()
    right = right.round()
    top = top.round()
    bottom = bottom.round()

    tf = torch.eye(3)[None].expand(B, -1, -1).contiguous()
    tf[:, 0, 2] = -left
    tf[:, 1, 2] = -top
    new_tf = torch.eye(3)[None].expand(B, -1, -1).contiguous()
    new_tf[:, 0, 0] = out_size[0] / (right - left)
    new_tf[:, 1, 1] = out_size[1] / (bottom - top)
    tf = new_tf @ tf
    return tf


def compute_crop_window_tf_batch(
    pts=None,
    poses=None,
    K=None,
    crop_ratio=1.2,
    out_size=None,
    uvs=None,
    method="min_box",
    mesh_diameter=None,
):
    """Project the points and find the cropping transform
    @pts: (N,3)
    @poses: (B,4,4) tensor
    @min_box: min_box/min_circle
    @scale: scale to apply to the tightly enclosing roi
    """
    B = len(poses)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    if method == "box_3d":
        radius = mesh_diameter * crop_ratio / 2
        offsets = torch.tensor(
            [0, 0, 0, radius, 0, 0, -radius, 0, 0, 0, radius, 0, 0, -radius, 0]
        ).reshape(-1, 3)
        pts = poses[:, :3, 3].reshape(-1, 1, 3) + offsets.reshape(1, -1, 3)
        K = torch.as_tensor(K)
        projected = (K.to(dtype=pts.dtype) @ pts.reshape(-1, 3).T).T
        uvs = projected[:, :2] / projected[:, 2:3]
        uvs = uvs.reshape(B, -1, 2)
        center = uvs[:, 0]
        radius = (
            torch.abs(uvs - center.reshape(-1, 1, 2))
            .reshape(B, -1)
            .max(axis=-1)[0]
            .reshape(-1)
        )
        left = center[:, 0] - radius
        right = center[:, 0] + radius
        top = center[:, 1] - radius
        bottom = center[:, 1] + radius
        tfs = compute_tf_batch(left, right, top, bottom, out_size)
        return tfs
    else:
        raise RuntimeError(f"No processing {method} method.")


def transform_pts(pts, tf):
    """Transform 2d or 3d points
    @pts: (...,N_pts,3)
    @tf: (...,4,4)
    """
    if len(tf.shape) >= 3 and tf.shape[-3] != pts.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]


def projection_matrix_from_intrinsics(
    K, height, width, znear, zfar, window_coords="y_down"
):
    """Conversion of Hartley-Zisserman intrinsic matrix to OpenGL proj. matrix.

    Ref:
    1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
    2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py

    :param K: 3x3 ndarray with the intrinsic camera matrix.
    :param x0 The X coordinate of the camera image origin (typically 0).
    :param y0: The Y coordinate of the camera image origin (typically 0).
    :param w: Image width.
    :param h: Image height.
    :param nc: Near clipping plane.
    :param fc: Far clipping plane.
    :param window_coords: 'y_up' or 'y_down'.
    :return: 4x4 ndarray with the OpenGL projection matrix.
    """
    x0 = 0
    y0 = 0
    w = width
    h = height
    nc = znear
    fc = zfar

    depth = float(fc - nc)
    q = -(fc + nc) / depth
    qn = -2 * (fc * nc) / depth

    # Draw our images upside down, so that all the pixel-based coordinate
    # systems are the same.
    if window_coords == "y_up":
        proj = np.array(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
                [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                [0, 0, -1, 0],
            ]
        )

    # Draw the images upright and modify the projection matrix so that OpenGL
    # will generate window coords that compensate for the flipped image coords.
    elif window_coords == "y_down":
        proj = np.array(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
                [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                [0, 0, -1, 0],
            ]
        )
    else:
        raise NotImplementedError

    return proj


def transform_dirs(dirs, tf):
    """
    @dirs: (...,3)
    @tf: (...,4,4)
    """
    if len(tf.shape) >= 3 and tf.shape[-3] != dirs.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :3, :3] @ dirs[..., None])[..., 0]


def to_homo_torch(pts):
    """
    @pts: shape can be (...,N,3 or 2) or (N,3) will homogeneliaze the last dimension
    """
    ones = torch.ones((*pts.shape[:-1], 1), dtype=torch.float, device=pts.device)
    homo = torch.cat((pts, ones), dim=-1)
    return homo


def nvdiffrast_render(
    K=None,
    H=None,
    W=None,
    ob_in_cams=None,
    glctx=None,
    context="cuda",
    get_normal=False,
    mesh_tensors=None,
    projection_mat=None,
    bbox2d=None,
    output_size=None,
    use_light=False,
    light_color=None,
    light_dir=np.array([0, 0, 1]),
    light_pos=np.array([0, 0, 0]),
    w_ambient=0.8,
    w_diffuse=0.5,
    extra={},
):
    """Just plain rendering, not support any gradient
    @K: (3,3) np array
    @ob_in_cams: (N,4,4) torch tensor, openCV camera
    @projection_mat: np array (4,4)
    @output_size: (height, width)
    @bbox2d: (N,4) (umin,vmin,umax,vmax) if only roi need to render.
    @light_dir: in cam space
    @light_pos: in cam space
    """
    if glctx is None:
        if context == "gl":
            glctx = dr.RasterizeGLContext()
        elif context == "cuda":
            glctx = dr.RasterizeCudaContext()
        else:
            raise NotImplementedError

    pos = mesh_tensors["pos"]
    vnormals = mesh_tensors["vnormals"]
    pos_idx = mesh_tensors["faces"]
    has_tex = "tex" in mesh_tensors

    glcam_in_cvcam = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    ).astype(float)
    ob_in_glcams = (
        torch.tensor(glcam_in_cvcam, device="cuda", dtype=torch.float)[None]
        @ ob_in_cams
    )

    if projection_mat is None:
        projection_mat = projection_matrix_from_intrinsics(
            K, height=H, width=W, znear=0.1, zfar=100
        )
    projection_mat = torch.as_tensor(
        projection_mat.reshape(-1, 4, 4), device="cuda", dtype=torch.float
    )
    mtx = projection_mat @ ob_in_glcams

    if output_size is None:
        output_size = np.asarray([H, W])

    pts_cam = transform_pts(pos, ob_in_cams)
    pos_homo = to_homo_torch(pos)
    pos_clip = (mtx[:, None] @ pos_homo[None, ..., None])[..., 0]
    if bbox2d is not None:
        l = bbox2d[:, 0]
        t = H - bbox2d[:, 1]
        r = bbox2d[:, 2]
        b = H - bbox2d[:, 3]
        tf = (
            torch.eye(4, dtype=torch.float, device="cuda")
            .reshape(1, 4, 4)
            .expand(len(ob_in_cams), 4, 4)
            .contiguous()
        )
        tf[:, 0, 0] = W / (r - l)
        tf[:, 1, 1] = H / (t - b)
        tf[:, 3, 0] = (W - r - l) / (r - l)
        tf[:, 3, 1] = (H - t - b) / (t - b)
        pos_clip = pos_clip @ tf
    rast_out, _ = dr.rasterize(
        glctx, pos_clip, pos_idx, resolution=np.asarray(output_size)
    )
    xyz_map, _ = dr.interpolate(pts_cam, rast_out, pos_idx)
    depth = xyz_map[..., 2]
    if has_tex:
        texc, _ = dr.interpolate(mesh_tensors["uv"], rast_out, mesh_tensors["uv_idx"])
        color = dr.texture(mesh_tensors["tex"], texc, filter_mode="linear")
    else:
        color, _ = dr.interpolate(mesh_tensors["vertex_color"], rast_out, pos_idx)

    if use_light:
        get_normal = True
    if get_normal:
        vnormals_cam = transform_dirs(vnormals, ob_in_cams)
        normal_map, _ = dr.interpolate(vnormals_cam, rast_out, pos_idx)
        normal_map = F.normalize(normal_map, dim=-1)
        normal_map = torch.flip(normal_map, dims=[1])
    else:
        normal_map = None

    if use_light:
        if light_dir is not None:
            light_dir_neg = -torch.as_tensor(
                light_dir, dtype=torch.float, device="cuda"
            )
        else:
            light_dir_neg = (
                torch.as_tensor(light_pos, dtype=torch.float, device="cuda").reshape(
                    1, 1, 3
                )
                - pts_cam
            )
        diffuse_intensity = (
            (F.normalize(vnormals_cam, dim=-1) * F.normalize(light_dir_neg, dim=-1))
            .sum(dim=-1)
            .clip(0, 1)[..., None]
        )
        diffuse_intensity_map, _ = dr.interpolate(
            diffuse_intensity, rast_out, pos_idx
        )  # (N_pose, H, W, 1)
        if light_color is None:
            light_color = color
        else:
            light_color = torch.as_tensor(light_color, device="cuda", dtype=torch.float)
        color = color * w_ambient + diffuse_intensity_map * light_color * w_diffuse

    color = color.clip(0, 1)
    color = color * torch.clamp(
        rast_out[..., -1:], 0, 1
    )  # Mask out background using alpha
    color = torch.flip(color, dims=[1])  # Flip Y coordinates
    depth = torch.flip(depth, dims=[1])
    extra["xyz_map"] = torch.flip(xyz_map, dims=[1])

    return color, depth, normal_map


def make_crop_data_batch(
    render_size,
    ob_in_cams,
    mesh,
    rgb,
    depth,
    K,
    crop_ratio,
    xyz_map,
    normal_map=None,
    mesh_diameter=None,
    mesh_tensors=None,
    device="cuda",
):
    """Cropping data from the image by batch"""
    H, W = depth.shape[:2]
    method = "box_3d"
    tf_to_crops = compute_crop_window_tf_batch(
        pts=mesh.vertices,
        poses=ob_in_cams,
        K=K,
        crop_ratio=crop_ratio,
        out_size=(render_size[1], render_size[0]),
        method=method,
        mesh_diameter=mesh_diameter,
    )

    B = len(ob_in_cams)
    poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device=device)

    bs = 512
    rgb_rs = []
    depth_rs = []
    normal_rs = []
    xyz_map_rs = []

    bbox2d_crop = torch.as_tensor(
        np.array([0, 0, render_size[0] - 1, render_size[1] - 1]).reshape(2, 2),
        device=device,
        dtype=torch.float,
    )
    bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()).reshape(-1, 4)

    for b in range(0, len(poseA), bs):
        extra = {}
        rgb_r, depth_r, normal_r = nvdiffrast_render(
            K=K,
            H=H,
            W=W,
            ob_in_cams=poseA[b : b + bs],
            context="cuda",
            get_normal=False,
            mesh_tensors=mesh_tensors,
            output_size=render_size,
            bbox2d=bbox2d_ori[b : b + bs],
            use_light=True,
            extra=extra,
        )
        rgb_rs.append(rgb_r)
        depth_rs.append(depth_r[..., None])
        normal_rs.append(normal_r)
        xyz_map_rs.append(extra["xyz_map"])

    rgb_rs = torch.cat(rgb_rs, dim=0).permute(0, 3, 1, 2) * 255
    depth_rs = torch.cat(depth_rs, dim=0).permute(0, 3, 1, 2)
    xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0, 3, 1, 2)
    Ks = torch.as_tensor(K, device=device, dtype=torch.float).reshape(1, 3, 3)

    rgbBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(rgb, dtype=torch.float, device=device)
        .permute(2, 0, 1)[None]
        .expand(B, -1, -1, -1),
        tf_to_crops,
        dsize=render_size,
        mode="bilinear",
        align_corners=False,
    )

    if rgb_rs.shape[-2:] != render_size:
        rgbAs = kornia.geometry.transform.warp_perspective(
            rgb_rs, tf_to_crops, dsize=render_size, mode="bilinear", align_corners=False
        )
    else:
        rgbAs = rgb_rs

    if xyz_map_rs.shape[-2:] != render_size:
        xyz_mapAs = kornia.geometry.transform.warp_perspective(
            xyz_map_rs,
            tf_to_crops,
            dsize=render_size,
            mode="nearest",
            align_corners=False,
        )
    else:
        xyz_mapAs = xyz_map_rs
    xyz_mapBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(xyz_map, device=device, dtype=torch.float)
        .permute(2, 0, 1)[None]
        .expand(B, -1, -1, -1),
        tf_to_crops,
        dsize=render_size,
        mode="nearest",
        align_corners=False,
    )  # (B,3,H,W)

    normalAs = None
    normalBs = None

    mesh_diameters = (
        torch.ones((len(rgbAs)), dtype=torch.float, device=device) * mesh_diameter
    )
    pose_data = BatchPoseData(
        rgbAs=rgbAs,
        rgbBs=rgbBs,
        depthAs=None,
        depthBs=None,
        normalAs=normalAs,
        normalBs=normalBs,
        poseA=poseA,
        poseB=None,
        xyz_mapAs=xyz_mapAs,
        xyz_mapBs=xyz_mapBs,
        tf_to_crops=tf_to_crops,
        Ks=Ks,
        mesh_diameters=mesh_diameters,
    )
    pose_data = transform_batch(batch=pose_data, H_ori=H, W_ori=W)
    return pose_data


@dataclass
class PoseData:
    """
    rgb: (h, w, 3) uint8
    depth: (bsz, h, w) float32
    bbox: (4, ) int
    K: (3, 3) float32
    """

    rgb: np.ndarray = None
    bbox: np.ndarray = None
    K: np.ndarray = None
    depth: Optional[np.ndarray] = None
    object_data = None
    mesh_diameter: float = None
    rgbA: np.ndarray = None
    rgbB: np.ndarray = None
    depthA: np.ndarray = None
    depthB: np.ndarray = None
    maskA = None
    maskB = None
    poseA: np.ndarray = None  # (4,4)
    target: float = None

    def __init__(
        self,
        rgbA=None,
        rgbB=None,
        depthA=None,
        depthB=None,
        maskA=None,
        maskB=None,
        normalA=None,
        normalB=None,
        xyz_mapA=None,
        xyz_mapB=None,
        poseA=None,
        poseB=None,
        K=None,
        target=None,
        mesh_diameter=None,
        tf_to_crop=None,
        crop_mask=None,
        model_pts=None,
        label=None,
        model_scale=None,
    ):
        self.rgbA = rgbA
        self.rgbB = rgbB
        self.depthA = depthA
        self.depthB = depthB
        self.poseA = poseA
        self.poseB = poseB
        self.maskA = maskA
        self.maskB = maskB
        self.crop_mask = crop_mask
        self.normalA = normalA
        self.normalB = normalB
        self.xyz_mapA = xyz_mapA
        self.xyz_mapB = xyz_mapB
        self.target = target
        self.K = K
        self.mesh_diameter = mesh_diameter
        self.tf_to_crop = tf_to_crop
        self.model_pts = model_pts
        self.label = label
        self.model_scale = model_scale


@dataclass
class BatchPoseData:
    """
    rgbs: (bsz, 3, h, w) torch tensor uint8
    depths: (bsz, h, w) float32
    bboxes: (bsz, 4) int
    K: (bsz, 3, 3) float32
    """

    rgbs: torch.Tensor = None
    object_datas = None
    bboxes: torch.Tensor = None
    K: torch.Tensor = None
    depths: Optional[torch.Tensor] = None
    rgbAs = None
    rgbBs = None
    depthAs = None
    depthBs = None
    normalAs = None
    normalBs = None
    poseA = None
    poseB = None
    targets = None

    def __init__(
        self,
        rgbAs=None,
        rgbBs=None,
        depthAs=None,
        depthBs=None,
        normalAs=None,
        normalBs=None,
        maskAs=None,
        maskBs=None,
        poseA=None,
        poseB=None,
        xyz_mapAs=None,
        xyz_mapBs=None,
        tf_to_crops=None,
        Ks=None,
        crop_masks=None,
        model_pts=None,
        mesh_diameters=None,
        labels=None,
    ):
        self.rgbAs = rgbAs
        self.rgbBs = rgbBs
        self.depthAs = depthAs
        self.depthBs = depthBs
        self.normalAs = normalAs
        self.normalBs = normalBs
        self.poseA = poseA
        self.poseB = poseB
        self.maskAs = maskAs
        self.maskBs = maskBs
        self.xyz_mapAs = xyz_mapAs
        self.xyz_mapBs = xyz_mapBs
        self.tf_to_crops = tf_to_crops
        self.crop_masks = crop_masks
        self.Ks = Ks
        self.model_pts = model_pts
        self.mesh_diameters = mesh_diameters
        self.labels = labels

    def pin_memory(self) -> "BatchPoseData":
        for k in self.__dict__:
            if self.__dict__[k] is not None:
                try:
                    self.__dict__[k] = self.__dict__[k].pin_memory()
                except Exception as e:
                    pass
        return self

    def cuda(self):
        for k in self.__dict__:
            if self.__dict__[k] is not None:
                try:
                    self.__dict__[k] = self.__dict__[k].cuda()
                except:
                    pass
        return self

    def select_by_indices(self, ids):
        out = BatchPoseData()
        for k in self.__dict__:
            if self.__dict__[k] is not None:
                out.__dict__[k] = self.__dict__[k][ids.to(self.__dict__[k].device)]
        return out


def depth2xyzmap_batch(depths, Ks, zfar):
    """
    @depths: torch tensor (B,H,W)
    @Ks: torch tensor (B,3,3)
    """
    bs = depths.shape[0]
    invalid_mask = (depths < 0.1) | (depths > zfar)
    H, W = depths.shape[-2:]
    vs, us = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing="ij")
    vs = vs.reshape(-1).float().cuda()[None].expand(bs, -1)
    us = us.reshape(-1).float().cuda()[None].expand(bs, -1)
    zs = depths.reshape(bs, -1)
    Ks = Ks[:, None].expand(bs, zs.shape[-1], 3, 3)
    xs = (us - Ks[..., 0, 2]) * zs / Ks[..., 0, 0]
    ys = (vs - Ks[..., 1, 2]) * zs / Ks[..., 1, 1]
    pts = torch.stack([xs, ys, zs], dim=-1)
    xyz_maps = pts.reshape(bs, H, W, 3)
    xyz_maps[invalid_mask] = 0
    return xyz_maps


def transform_depth_to_xyzmap(batch: BatchPoseData, H_ori, W_ori, mini_batch=32):
    """Transform the depth map to point cloud"""
    bs = len(batch.rgbAs)
    H, W = batch.rgbAs.shape[-2:]
    mesh_radius = batch.mesh_diameters.cuda() / 2
    tf_to_crops = batch.tf_to_crops.cuda()
    crop_to_oris = batch.tf_to_crops.inverse().cuda()
    batch.poseA = batch.poseA.cuda()
    batch.Ks = batch.Ks.cuda()

    if batch.xyz_mapAs is None:
        batch_depthAs = batch.depthAs.cuda().expand(bs, -1, -1, -1)
        batch.xyz_mapAs = torch.zeros((bs, 3, H, W))

        for idx in range(0, bs, mini_batch):
            depthAs_ori = kornia.geometry.transform.warp_perspective(
                batch_depthAs[idx : idx + mini_batch, :, :, :],
                crop_to_oris[idx : idx + mini_batch, :, :],
                dsize=(H_ori, W_ori),
                mode="nearest",
                align_corners=False,
            )
            xyz_mapAs = depth2xyzmap_batch(
                depthAs_ori[:, 0], batch.Ks[idx : idx + mini_batch, :, :], zfar=np.inf
            ).permute(0, 3, 1, 2)
            xyz_mapAs = kornia.geometry.transform.warp_perspective(
                xyz_mapAs,
                tf_to_crops[idx : idx + mini_batch, :, :],
                dsize=(H, W),
                mode="nearest",
                align_corners=False,
            )
            batch.xyz_mapAs[idx : idx + mini_batch, :, :, :] = xyz_mapAs
    batch.xyz_mapAs = batch.xyz_mapAs.cuda()
    invalid = batch.xyz_mapAs[:, 2:3] < 0.1
    batch.xyz_mapAs = batch.xyz_mapAs - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1)

    batch.xyz_mapAs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
    invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapAs) >= 2)
    batch.xyz_mapAs[invalid.expand(bs, 3, -1, -1)] = 0

    if batch.xyz_mapBs is None:
        batch_depthBs = batch.depthBs.cuda().expand(bs, -1, -1, -1)
        batch.xyz_mapBs = torch.zeros((bs, 3, H, W))
        for idx in range(0, bs, mini_batch):
            depthBs_ori = kornia.geometry.transform.warp_perspective(
                batch_depthBs[idx : idx + mini_batch, :, :, :],
                crop_to_oris[idx : idx + mini_batch, :, :],
                dsize=(H_ori, W_ori),
                mode="nearest",
                align_corners=False,
            )
            xyz_mapBs = depth2xyzmap_batch(
                depthBs_ori[:, 0], batch.Ks[idx : idx + mini_batch, :, :], zfar=np.inf
            ).permute(0, 3, 1, 2)
            xyz_mapBs = kornia.geometry.transform.warp_perspective(
                xyz_mapBs,
                tf_to_crops[idx : idx + mini_batch, :, :],
                dsize=(H, W),
                mode="nearest",
                align_corners=False,
            )
            batch.xyz_mapBs[idx : idx + mini_batch, :, :, :] = xyz_mapBs

    batch.xyz_mapBs = batch.xyz_mapBs.cuda()
    invalid = batch.xyz_mapBs[:, 2:3] < 0.1
    batch.xyz_mapBs = batch.xyz_mapBs - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1)

    batch.xyz_mapBs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
    invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapBs) >= 2)
    batch.xyz_mapBs[invalid.expand(bs, 3, -1, -1)] = 0

    return batch


def transform_batch(batch: BatchPoseData, H_ori, W_ori):
    """Transform the batch before feeding to the network
    !NOTE the H_ori, W_ori could be different at test time from the training data, and needs to be set
    """
    batch.rgbAs = batch.rgbAs.cuda().float() / 255.0
    batch.rgbBs = batch.rgbBs.cuda().float() / 255.0
    batch = transform_depth_to_xyzmap(batch, H_ori, W_ori)
    return batch


def get_tf_to_centered_mesh(model_center):
    """Calculate the center of mesh"""
    tf_to_center = torch.eye(4, dtype=torch.float, device="cuda")
    tf_to_center[:3, 3] = -torch.as_tensor(
        model_center, device="cuda", dtype=torch.float
    )
    return tf_to_center


def to_homo(pts):
    """
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    """
    assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
    homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
    return homo


def draw_posed_3d_box(K, img, ob_in_cam, bbox, line_color=(0, 255, 0), linewidth=2):
    """Revised from 6pack dataset/inference_dataset_nocs.py::projection
    @bbox: (2,3) min/max
    @line_color: RGB
    """
    min_xyz = bbox.min(axis=0)
    xmin, ymin, zmin = min_xyz
    max_xyz = bbox.max(axis=0)
    xmax, ymax, zmax = max_xyz

    def draw_line3d(start, end, img):
        """Draw the 3D bounding box"""
        pts = np.stack((start, end), axis=0).reshape(-1, 3)
        pts = (ob_in_cam @ to_homo(pts).T).T[:, :3]  # (2,3)
        projected = (np.array(K) @ pts.T).T
        uv = np.round(projected[:, :2] / projected[:, 2].reshape(-1, 1)).astype(
            int
        )  # (2,2)
        img = cv2.line(
            img,
            uv[0].tolist(),
            uv[1].tolist(),
            color=line_color,
            thickness=linewidth,
            lineType=cv2.LINE_AA,
        )
        return img

    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            start = np.array([xmin, y, z])
            end = start + np.array([xmax - xmin, 0, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for z in [zmin, zmax]:
            start = np.array([x, ymin, z])
            end = start + np.array([0, ymax - ymin, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            start = np.array([x, y, zmin])
            end = start + np.array([0, 0, zmax - zmin])
            img = draw_line3d(start, end, img)

    return img


def project_3d_to_2d(pt, K, ob_in_cam):
    """Project 3D keypoint to 2D image plane"""
    pt = pt.reshape(4, 1)
    projected = np.array(K) @ ((ob_in_cam @ pt)[:3, :])
    projected = projected.reshape(-1)
    projected = projected / projected[2]
    return projected.reshape(-1)[:2].round().astype(int)


def draw_xyz_axis(
    color,
    ob_in_cam,
    scale=0.1,
    K=np.eye(3),
    thickness=3,
    transparency=0,
    is_input_rgb=False,
):
    """
    Draw the object pose.
    @color: BGR
    """
    if is_input_rgb:
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    xx = np.array([1, 0, 0, 1]).astype(float)
    yy = np.array([0, 1, 0, 1]).astype(float)
    zz = np.array([0, 0, 1, 1]).astype(float)
    xx[:3] = xx[:3] * scale
    yy[:3] = yy[:3] * scale
    zz[:3] = zz[:3] * scale

    origin = tuple(project_3d_to_2d(np.array([0, 0, 0, 1]), K, ob_in_cam).tolist())
    xx = tuple(project_3d_to_2d(xx, K, ob_in_cam).tolist())
    yy = tuple(project_3d_to_2d(yy, K, ob_in_cam).tolist())
    zz = tuple(project_3d_to_2d(zz, K, ob_in_cam).tolist())
    line_type = cv2.LINE_AA
    arrow_len = 0

    tmp1 = color.copy()
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        xx,
        color=(0, 0, 255),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        yy,
        color=(0, 255, 0),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )
    tmp1 = cv2.arrowedLine(
        tmp1,
        origin,
        zz,
        color=(255, 0, 0),
        thickness=thickness,
        line_type=line_type,
        tipLength=arrow_len,
    )

    if is_input_rgb:
        tmp1 = cv2.cvtColor(tmp1, cv2.COLOR_BGR2RGB)

    return tmp1


def egocentric_delta_pose_to_pose(A_in_cam, trans_delta, rot_mat_delta):
    """Used for Pose Refinement. Given the object's two poses in camera, convert them to relative poses in camera's egocentric view
    @A_in_cam: (B,4,4) torch tensor
    """
    B_in_cam = (
        torch.eye(4, dtype=torch.float, device=A_in_cam.device)[None]
        .expand(len(A_in_cam), -1, -1)
        .contiguous()
    )
    B_in_cam[:, :3, 3] = A_in_cam[:, :3, 3] + trans_delta
    B_in_cam[:, :3, :3] = rot_mat_delta @ A_in_cam[:, :3, :3]
    return B_in_cam


@torch.no_grad()
def make_crop_data_batch_score(
    render_size,
    ob_in_cams,
    mesh,
    rgb,
    depth,
    K,
    crop_ratio,
    normal_map=None,
    mesh_diameter=None,
    mesh_tensors=None,
):
    """Cropping the data batch for the score network"""
    H, W = depth.shape[:2]

    method = "box_3d"
    tf_to_crops = compute_crop_window_tf_batch(
        pts=mesh.vertices,
        poses=ob_in_cams,
        K=K,
        crop_ratio=crop_ratio,
        out_size=(render_size[1], render_size[0]),
        method=method,
        mesh_diameter=mesh_diameter,
    )

    B = len(ob_in_cams)
    poseAs = torch.as_tensor(ob_in_cams, dtype=torch.float, device="cuda")

    rgb_rs = []
    depth_rs = []

    bbox2d_crop = torch.as_tensor(
        np.array([0, 0, 159, 159]).reshape(2, 2), device="cuda", dtype=torch.float
    )
    bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()[:, None]).reshape(
        -1, 4
    )

    extra = {}
    rgb_r, depth_r, normal_r = nvdiffrast_render(
        K=K,
        H=H,
        W=W,
        ob_in_cams=poseAs,
        context="cuda",
        get_normal=False,
        mesh_tensors=mesh_tensors,
        output_size=(160, 160),
        bbox2d=bbox2d_ori,
        use_light=True,
        extra=extra,
    )
    rgb_rs.append(rgb_r)
    depth_rs.append(depth_r[..., None])

    rgb_rs = torch.cat(rgb_rs, dim=0).permute(0, 3, 1, 2) * 255
    depth_rs = torch.cat(depth_rs, dim=0).permute(0, 3, 1, 2)

    rgbBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(rgb, dtype=torch.float, device="cuda")
        .permute(2, 0, 1)[None]
        .expand(B, -1, -1, -1),
        tf_to_crops,
        dsize=render_size,
        mode="bilinear",
        align_corners=False,
    )
    depthBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(depth, dtype=torch.float, device="cuda")[None, None].expand(
            B, -1, -1, -1
        ),
        tf_to_crops,
        dsize=render_size,
        mode="nearest",
        align_corners=False,
    )

    if rgb_rs.shape[-2:] != render_size:
        rgbAs = kornia.geometry.transform.warp_perspective(
            rgb_rs, tf_to_crops, dsize=render_size, mode="bilinear", align_corners=False
        )
        depthAs = kornia.geometry.transform.warp_perspective(
            depth_rs,
            tf_to_crops,
            dsize=render_size,
            mode="nearest",
            align_corners=False,
        )
    else:
        rgbAs = rgb_rs
        depthAs = depth_rs

    normalAs = None
    normalBs = None

    Ks = torch.as_tensor(K, dtype=torch.float).reshape(1, 3, 3).expand(len(rgbAs), 3, 3)
    mesh_diameters = (
        torch.ones((len(rgbAs)), dtype=torch.float, device="cuda") * mesh_diameter
    )

    pose_data = BatchPoseData(
        rgbAs=rgbAs,
        rgbBs=rgbBs,
        depthAs=depthAs,
        depthBs=depthBs,
        normalAs=normalAs,
        normalBs=normalBs,
        poseA=poseAs,
        tf_to_crops=tf_to_crops,
        Ks=Ks,
        mesh_diameters=mesh_diameters,
    )
    pose_data = transform_batch(pose_data, H_ori=H, W_ori=W)

    return pose_data
