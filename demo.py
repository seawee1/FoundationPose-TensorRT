import glob
import logging
import os
from foundationpose_tensorrt import FoundationPoseWrapper, FoundationPoseWrapperConfig

import time
import numpy as np
import cv2
import imageio
import trimesh


# Taken from https://nvlabs.github.io/FoundationPose/
def depth2xyzmap(depth, K, uvs=None):
    invalid_mask = depth < 0.001
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
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N, 3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map


# Taken from https://nvlabs.github.io/FoundationPose/
class YcbineoatReader:
    def __init__(self, video_dir, downscale=1, shorter_side=None, zfar=np.inf):
        self.video_dir = video_dir
        self.downscale = downscale
        self.zfar = zfar
        self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))
        self.K = np.loadtxt(f"{video_dir}/cam_K.txt").reshape(3, 3)
        self.id_strs = []
        for color_file in self.color_files:
            id_str = os.path.basename(color_file).replace(".png", "")
            self.id_strs.append(id_str)
        self.H, self.W = cv2.imread(self.color_files[0]).shape[:2]

        if shorter_side is not None:
            self.downscale = shorter_side / min(self.H, self.W)

        self.H = int(self.H * self.downscale)
        self.W = int(self.W * self.downscale)
        self.K[:2] *= self.downscale

        self.gt_pose_files = sorted(glob.glob(f"{self.video_dir}/annotated_poses/*"))

        self.videoname_to_object = {
            "bleach0": "021_bleach_cleanser",
            "bleach_hard_00_03_chaitanya": "021_bleach_cleanser",
            "cracker_box_reorient": "003_cracker_box",
            "cracker_box_yalehand0": "003_cracker_box",
            "mustard0": "006_mustard_bottle",
            "mustard_easy_00_02": "006_mustard_bottle",
            "sugar_box1": "004_sugar_box",
            "sugar_box_yalehand0": "004_sugar_box",
            "tomato_soup_can_yalehand0": "005_tomato_soup_can",
        }

    def get_video_name(self):
        return self.video_dir.split("/")[-1]

    def __len__(self):
        return len(self.color_files)

    def get_gt_pose(self, i):
        try:
            pose = np.loadtxt(self.gt_pose_files[i]).reshape(4, 4)
            return pose
        except:
            logging.info("GT pose not found, return None")
            return None

    def get_color(self, i):
        color = imageio.imread(self.color_files[i])[..., :3]
        color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return color

    def get_mask(self, i):
        mask = cv2.imread(self.color_files[i].replace("rgb", "masks"), -1)
        if len(mask.shape) == 3:
            for c in range(3):
                if mask[..., c].sum() > 0:
                    mask = mask[..., c]
                    break
        mask = (
            cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            .astype(bool)
            .astype(np.uint8)
        )
        return mask

    def get_depth(self, i):
        depth = cv2.imread(self.color_files[i].replace("rgb", "depth"), -1) / 1e3
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.001) | (depth >= self.zfar)] = 0
        return depth

    def get_xyz_map(self, i):
        depth = self.get_depth(i)
        xyz_map = depth2xyzmap(depth, self.K)
        return xyz_map

    def get_occ_mask(self, i):
        hand_mask_file = self.color_files[i].replace("rgb", "masks_hand")
        occ_mask = np.zeros((self.H, self.W), dtype=bool)
        if os.path.exists(hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(hand_mask_file, -1) > 0)

        right_hand_mask_file = self.color_files[i].replace("rgb", "masks_hand_right")
        if os.path.exists(right_hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(right_hand_mask_file, -1) > 0)

        occ_mask = cv2.resize(
            occ_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST
        )

        return occ_mask.astype(np.uint8)

    def get_gt_mesh(self):
        ob_name = self.videoname_to_object[self.get_video_name()]
        YCB_VIDEO_DIR = os.getenv("YCB_VIDEO_DIR")
        mesh = trimesh.load(f"{YCB_VIDEO_DIR}/models/{ob_name}/textured_simple.obj")
        return mesh


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_scene_dir = os.path.join(script_dir, "demo_data", "mustard0")

    # Download test data
    if not os.path.exists(test_scene_dir):
        import zipfile
        import gdown

        file_id = "1AwV9sESDKMgXGUu2n1o0Pc4x2JGYdVB3"
        zip_path = os.path.join(script_dir, "demo_data.zip")
        print("demo_data/mustard0 not found, downloading demo data...")
        try:
            gdown.download(id=file_id, output=zip_path, quiet=False)
        except Exception as e:
            print(f"Error downloading demo data: {e}")
            print(f"Please download the demo data manually from https://drive.google.com/file/d/{file_id}/view?usp=sharing and extract it to demo/demo_data/")
            exit()
        print("Extracting demo data...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.join(script_dir, "demo_data"))
        os.remove(zip_path)
        print("Done.")

    test_scene_dir = test_scene_dir + "/"
    mesh_file = test_scene_dir + "mesh/textured_simple.obj"
    obj_name = "mustard0"

    # Initialize wrapper
    cfg = FoundationPoseWrapperConfig(
        downsample_width=None,  # Probably leave None for best accuracy, or set to, e.g., 256 for faster prediction
        est_refine_iter=5,  # Increase if the initial pose is not good enough
        track_refine_iter=2,  # Increase if the tracking is not good enough
        chunk_size=252,  # Specify which chunk size to use for the TensorRT engines (must match the value used in convert_onnx.sh when generating the engines
    )
    fp_wrapper = FoundationPoseWrapper(cfg=cfg)

    reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)  # type: ignore

    # Set camera intrinsics
    camera_intrinsics = reader.K
    fp_wrapper.set_camera_intrinsics(camera_intrinsics)

    # Load mesh
    mesh = FoundationPoseWrapper.load_mesh(mesh_file)

    step_scene_times = []
    estimation_times = []

    for i in range(len(reader.color_files)):
        # Get images from dataset
        color = reader.get_color(i)
        depth = reader.get_depth(i)

        if i == 0:
            mask = reader.get_mask(0).astype(bool)

            # Reset to new scene
            fp_wrapper.reset_scene(color, depth)

            # Add objects to be detected and tracked (triggers initial pose estimation)
            poses = {}
            poses[obj_name] = fp_wrapper.add_object(obj_name, mesh, mask)

            # Do a bunch of pose estimation steps just for benchmarking (add_object might contain additional time-intensive program logic)
            for i in range(10):
                print("Re-running initial estimation for benchmarking...")
                start_time = time.perf_counter()
                poses[obj_name] = fp_wrapper.register_object(obj_name)
                end_time = time.perf_counter()
                print(f"Estimation time: {(end_time - start_time) * 1000:.2f} ms")
                estimation_times.append(end_time - start_time)

        else:
            # Step to next frame (triggers pose tracking)
            start_time = time.perf_counter()
            poses = fp_wrapper.step_scene(color, depth)
            end_time = time.perf_counter()
            step_scene_times.append(end_time - start_time)

        print(f"Frame {i}, estimated poses: {poses}")

        res = fp_wrapper.render_results()
        cv2.imshow("rendered", res)
        cv2.waitKey(1)

    if estimation_times:
        mean_estimation_time = sum(estimation_times) / len(estimation_times)
        print(f"Mean time for initial estimation: {mean_estimation_time * 1000:.2f} ms")

    if step_scene_times:
        mean_step_scene_time = sum(step_scene_times) / len(step_scene_times)
        print(f"Mean time for step_scene: {mean_step_scene_time * 1000:.2f} ms")
