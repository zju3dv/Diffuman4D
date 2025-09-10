import os
import os.path as osp
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

from src.data.utils.camera_parser import parse_cameras
from src.data.utils.ray_utils import calc_plucker_embeds, calc_relative_poses
from src.data.utils.crop_utils import mask_crop_aspect_ratio, skeleton_to_mask
from src.data.utils.image_utils import apply_fmask, norm_vae_tensor
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SpaTemDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        camera_path_pat: str = "{data_dir}/{scene_label}/transforms.json",
        image_path_pat: str = "{data_dir}/{scene_label}/images/{spa_label}/{tem_label}.webp",
        fmask_path_pat: str = "{data_dir}/{scene_label}/fmasks/{spa_label}/{tem_label}.png",
        skeleton_path_pat: str = "{data_dir}/{scene_label}/skeletons/{spa_label}/{tem_label}.webp",
        scene_label: list[str] | None = None,
        height: int = 1024,
        width: int = 1024,
        has_gt_target: bool = True,  # if False, use skeleton as image and fmask for target samples
    ):
        self.data_dir = data_dir
        self.camera_path_pat = camera_path_pat
        self.image_path_pat = image_path_pat
        self.fmask_path_pat = fmask_path_pat
        self.skeleton_path_pat = skeleton_path_pat
        self.scene_label = scene_label

        if "$" in self.data_dir:
            self.data_dir = osp.expandvars(self.data_dir)

        if self.scene_label is None:
            self.scene_label = ""

        # load camera parameters
        self.cameras = {}
        camera_path = self.camera_path_pat.format(data_dir=self.data_dir, scene_label=self.scene_label)
        self.cameras[self.scene_label] = parse_cameras(camera_path, coord_system="opencv", normalize_scene=True)

        self.height = height
        self.width = width
        self.has_gt_target = has_gt_target

    def get_file_path(self, pat: str, scene_label: str, spa_label: str, tem_label: str) -> str:
        return pat.format(data_dir=self.data_dir, scene_label=scene_label, spa_label=spa_label, tem_label=tem_label)

    def transform_image(self, image: Image.Image, crop: tuple[int, int, int, int]) -> torch.Tensor:
        # crop and resize the image
        top, left, height, width, _, _ = crop
        image = TF.crop(image, top, left, height, width)

        image = TF.resize(image, (self.height, self.width), interpolation=InterpolationMode.BICUBIC, antialias=True)

        # normalize the image to [-1, 1]
        image = TF.to_tensor(image)
        image = norm_vae_tensor(image)
        return image

    def transform_intrinsic(self, K: torch.Tensor, crop: tuple[int, int, int, int]) -> torch.Tensor:
        top, left, height, _, _, _ = crop
        K = K.clone()
        K[0, 2] = K[0, 2] - left
        K[1, 2] = K[1, 2] - top
        K = K * (self.height / height)
        K[2, 2] = 1.0
        return K

    def get_item(
        self,
        scene_label: str,
        spa_labels: list[str],
        tem_labels: list[str],
        input_spa_labels: list[str],
    ) -> dict:
        if len(spa_labels) > 1 and len(tem_labels) == 1:
            domain = "spatial"
        elif len(spa_labels) == 1 and len(tem_labels) > 1:
            domain = "temporal"
        else:
            raise ValueError(f"Error: invalid spa_labels and tem_labels: {spa_labels} and {tem_labels}")

        cameras = self.cameras[scene_label]
        if domain == "spatial":
            labels = []
            for spa_label in spa_labels:
                label = (scene_label, spa_label, tem_labels[0])
                labels.append(label)
        elif domain == "temporal":
            # find nearest-distance ref camera
            cams = [cameras[spa_labels[0]]] + [cameras[label] for label in input_spa_labels]
            poses = [cam["pose"] for cam in cams]
            positions = torch.stack(poses)[:, :3, 3]
            distances = torch.norm(positions[1:] - positions[:1], dim=1)
            nearest_index = torch.argmin(distances).item()
            cond_spa_label = input_spa_labels[nearest_index]

            labels = []
            spa_labels = [cond_spa_label] + spa_labels
            for spa_label in spa_labels:
                for tem_label in tem_labels:
                    label = (scene_label, spa_label, tem_label)
                    labels.append(label)

        # > load frame samples
        images, fmasks, skeletons = [], [], []
        Ks, poses, hws, crops = [], [], [], []
        for i, label in enumerate(labels):
            scene_label, spa_label, tem_label = label
            image_path = self.get_file_path(self.image_path_pat, scene_label, spa_label, tem_label)
            fmask_path = self.get_file_path(self.fmask_path_pat, scene_label, spa_label, tem_label)
            skeleton_path = self.get_file_path(self.skeleton_path_pat, scene_label, spa_label, tem_label)

            # read data
            skeleton = Image.open(skeleton_path)
            if not self.has_gt_target and spa_label not in input_spa_labels:
                # use skeleton as image and fmask
                image = skeleton
                fmask = skeleton_to_mask(skeleton)
            else:
                image = Image.open(image_path)
                fmask = Image.open(fmask_path)

            camera = cameras[spa_label]
            K, pose, h, w = camera["K"], camera["pose"], camera["height"], camera["width"]
            # (top, left, height, width, ori_height, original_width)
            crop = mask_crop_aspect_ratio(fmask)

            # sanity check
            if not (image.size == fmask.size == skeleton.size):
                raise AssertionError(
                    f"Error: image size: {image.size} != fmask size: {fmask.size} != skeleton size: {skeleton.size}"
                )
            if self.has_gt_target and spa_label in input_spa_labels and TF.to_tensor(fmask).mean() <= 0.02:
                raise AssertionError(f"Error: foreground mask < 2%. Please check the data.")

            # transform the data
            image = self.transform_image(image, crop)
            fmask = self.transform_image(fmask, crop)
            skeleton = self.transform_image(skeleton, crop)
            K = self.transform_intrinsic(K, crop)

            images.append(image)
            fmasks.append(fmask)
            skeletons.append(skeleton)
            Ks.append(K)
            poses.append(pose)
            hws.append((h, w))
            crops.append(crop)

        Ks = torch.stack(Ks)
        poses = torch.stack(poses)
        images = torch.stack(images)
        fmasks = torch.stack(fmasks)
        skeletons = torch.stack(skeletons)

        # background removal
        pixel_values = apply_fmask(images, fmasks, background_color="white", vae_normalized=True)

        # relative poses
        poses = calc_relative_poses(poses)

        # plucker embeddings
        plucker_embeds = calc_plucker_embeds(self.height, self.width, Ks, poses)

        # conditional masks
        cond_masks = torch.ones_like(pixel_values)[:, :1, ...]
        cond_masks[len(pixel_values) // 2 :, ...] = 0.0  # ? hard code

        sample = {
            "domain": domain,
            "labels": labels,
            "pixel_values": pixel_values,
            "plucker_embeds": plucker_embeds,
            "skeletons": skeletons,
            "cond_masks": cond_masks,
            "Ks": Ks,
            "hws": hws,
            "crops": crops,
            "poses": poses,
        }

        def check_output(sample):
            if sample["domain"] == "temporal":
                half_len = len(sample["labels"]) // 2
                spa_labels = [label[1] for label in sample["labels"]]
                if any(spa_label != spa_labels[0] for spa_label in spa_labels[:half_len]):
                    raise ValueError(
                        f"Error: temporal labels are not consistent for the first {half_len} temporal frames."
                    )
                if any(spa_label != spa_labels[-1] for spa_label in spa_labels[half_len:]):
                    raise ValueError(
                        f"Error: temporal labels are not consistent for the last {half_len} temporal frames."
                    )
            elif sample["domain"] == "spatial":
                tem_labels = [label[2] for label in sample["labels"]]
                if any(tem_label != tem_labels[0] for tem_label in tem_labels):
                    raise ValueError(
                        f"Error: spatial labels are not consistent for the {len(sample['labels'])} spatial frames."
                    )

            min_val, max_val = -1.0 - 1e-6, 1.0 + 1e-6
            if min_val > sample["pixel_values"].min() or max_val < sample["pixel_values"].max():
                raise ValueError(
                    f"Error: pixel values are out of range: {sample['pixel_values'].min()} < {min_val} or {sample['pixel_values'].max()} > {max_val}"
                )
            if min_val > sample["skeletons"].min() or max_val < sample["skeletons"].max():
                raise ValueError(
                    f"Error: skeletons are out of range: {sample['skeletons'].min()} < {min_val} or {sample['skeletons'].max()} > {max_val}"
                )
            if min_val > sample["plucker_embeds"].min() or max_val < sample["plucker_embeds"].max():
                raise ValueError(
                    f"Error: plucker embeds are out of range: {sample['plucker_embeds'].min()} < {min_val} or {sample['plucker_embeds'].max()} > {max_val}"
                )
            if min_val > sample["cond_masks"].min() or max_val < sample["cond_masks"].max():
                raise ValueError(
                    f"Error: cond masks are out of range: {sample['cond_masks'].min()} < {min_val} or {sample['cond_masks'].max()} > {max_val}"
                )

        check_output(sample)
        return sample
