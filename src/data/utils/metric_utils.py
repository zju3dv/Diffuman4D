import os
import torch
from tqdm import tqdm
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import to_tensor, to_pil_image, resize
from torchvision.transforms import InterpolationMode

from src.data.utils.data_utils import save_json


class ImageEvaluator:
    def __init__(self, device):
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device)

    def mask_to_bbox(self, fmask, padding=8):
        if fmask.dim() == 3:
            fmask = fmask.squeeze()
        rows = torch.any(fmask, dim=1).nonzero(as_tuple=True)[0]
        cols = torch.any(fmask, dim=0).nonzero(as_tuple=True)[0]
        if rows.numel() == 0 or cols.numel() == 0:
            return None
        # xmin, ymin, xmax, ymax
        # left, top, right, bottom
        return (
            max(cols[0].item() - padding, 0),
            max(rows[0].item() - padding, 0),
            min(cols[-1].item() + 1 + padding, fmask.shape[1]),
            min(rows[-1].item() + 1 + padding, fmask.shape[0]),
        )

    def obbs_union(self, obbs: list[tuple[int, int, int, int]]):
        l1 = [obb[0] for obb in obbs]
        t1 = [obb[1] for obb in obbs]
        r1 = [obb[2] for obb in obbs]
        b1 = [obb[3] for obb in obbs]
        union_obb = (min(l1), min(t1), max(r1), max(b1))
        return union_obb

    def crop_with_obb(self, image, bbox):
        left, top, right, bottom = bbox
        return image[..., top:bottom, left:right]

    def apply_fmask(self, image, fmask, background_color):
        if background_color == "black":
            return image * fmask
        elif background_color == "white":
            return image * fmask + (1.0 - fmask)
        elif background_color == "grey":
            return image * fmask + (1.0 - fmask) * 0.5
        else:
            raise ValueError(f"Invalid background color: {background_color}")

    def __call__(
        self,
        pred: torch.Tensor | str,
        gt: torch.Tensor | str,
        pred_fmask: torch.Tensor | str | None = None,
        gt_fmask: torch.Tensor | str | None = None,
        canvas_size: int = 1024,
        crop_with_fmask: bool = True,
        background_color: str = "black",
    ):
        # load images if paths are provided
        if isinstance(pred, str):
            pred = to_tensor(Image.open(pred))
        if isinstance(gt, str):
            gt = to_tensor(Image.open(gt))
        if isinstance(pred_fmask, str):
            pred_fmask = to_tensor(Image.open(pred_fmask))
        if isinstance(gt_fmask, str):
            gt_fmask = to_tensor(Image.open(gt_fmask))

        pred = pred.to(self.device)
        gt = gt.to(self.device)
        if pred_fmask is not None:
            pred_fmask = pred_fmask.to(self.device)
        if gt_fmask is not None:
            gt_fmask = gt_fmask.to(self.device)

        # sanity check
        if gt.shape != pred.shape:
            raise ValueError("The GT and predicted images should have the same shape.")
        if pred_fmask is not None and pred.shape[-2:] != pred_fmask.shape[-2:]:
            raise ValueError(f"shape mismatch: {pred.shape} != {pred_fmask.shape}")
        if gt_fmask is not None and gt.shape[-2:] != gt_fmask.shape[-2:]:
            raise ValueError(f"shape mismatch: {gt.shape} != {gt_fmask.shape}")
        if background_color not in ["black", "white", "grey"]:
            raise ValueError(f"Invalid background color: {background_color}")
        if crop_with_fmask and (pred_fmask is None and gt_fmask is None):
            raise ValueError("Either pred_fmask or gt_fmask should be provided to crop with fmask.")

        # apply fmasks if provided
        if gt_fmask is not None:
            gt = self.apply_fmask(gt, gt_fmask, background_color)
        if pred_fmask is not None:
            pred = self.apply_fmask(pred, pred_fmask, background_color)

        # resize images and fmasks
        if canvas_size != gt.shape[-1]:
            # the smaller edge of the image is resized to canvas_size
            gt = resize(gt, size=canvas_size, interpolation=InterpolationMode.NEAREST)
            pred = resize(pred, size=canvas_size, interpolation=InterpolationMode.NEAREST)
            if gt_fmask is not None:
                gt_fmask = resize(gt_fmask, size=canvas_size, interpolation=InterpolationMode.NEAREST)
            if pred_fmask is not None:
                pred_fmask = resize(pred_fmask, size=canvas_size, interpolation=InterpolationMode.NEAREST)

        # crop with fmasks union
        if crop_with_fmask:
            obbs = [self.mask_to_bbox(fmask) for fmask in [gt_fmask, pred_fmask] if fmask is not None]
            if obbs:
                obb = self.obbs_union(obbs)
                # sanity check
                if (obb[2] - obb[0]) * (obb[3] - obb[1]) < gt.numel() * 0.02:
                    raise ValueError("The cropped region is too small. Please check your data.")
                gt = self.crop_with_obb(gt, obb)
                pred = self.crop_with_obb(pred, obb)

        # sanity check
        if 0.0 - 1e-6 > gt.min() or gt.max() > 1.0 + 1e-6:
            raise ValueError("The GT image should be normalized.")
        if 0.0 - 1e-6 > pred.min() or pred.max() > 1.0 + 1e-6:
            raise ValueError("The predicted image should be normalized.")

        # we can't use batch processing because the croppings are different
        gt = gt[None, ...]
        pred = pred[None, ...]
        psnr = self.psnr(gt, pred).item()
        ssim = self.ssim(gt, pred).item()
        lpips = self.lpips(gt, pred).item()
        return psnr, ssim, lpips


def evaluate_results(
    pred_images_dir: str,
    gt_images_dir: str,
    fmasks_dir: str | None = None,
    pred_image_ext: str = ".jpg",
    gt_image_ext: str = ".jpg",
    fmask_ext: str = ".png",
    spa_labels: list[str] | None = None,
    tem_labels: list[str] | None = None,
    out_metrics_path: str | None = None,
    crop_with_fmask: bool = True,
    background_color: str = "black",
    gpu_ids: list[int] | None = None,
) -> dict:
    def split_batches(data, num_batches):
        return [data[i::num_batches] for i in range(num_batches)]

    def evaluate_on_single_gpu(
        pred_image_paths: list[str],
        gt_image_paths: list[str],
        fmask_paths: list[str],
        keys: list[str],
        gpu_id: int,
    ):
        res = []
        device = f"cuda:{gpu_id}"
        image_evaluator = ImageEvaluator(device=device)

        for pred_path, gt_path, fmask_path, key in tqdm(
            zip(pred_image_paths, gt_image_paths, fmask_paths, keys),
            total=len(pred_image_paths),
            desc=f"Evaluating results on {device}",
        ):
            psnr, ssim, lpips = image_evaluator(
                pred=pred_path,
                gt=gt_path,
                pred_fmask=fmask_path,
                gt_fmask=fmask_path,
                crop_with_fmask=crop_with_fmask,
                background_color=background_color,
            )
            res.append({"key": key, "psnr": psnr, "ssim": ssim, "lpips": lpips})
        return res

    if spa_labels is None:
        spa_labels = sorted(os.listdir(pred_images_dir))
    if tem_labels is None:
        tem_labels = sorted(os.listdir(f"{pred_images_dir}/{spa_labels[0]}"))
        tem_labels = [tem_label.split(".")[0] for tem_label in tem_labels]

    keys = [f"{spa_label}/{tem_label}" for spa_label in spa_labels for tem_label in tem_labels]
    pred_image_paths = [f"{pred_images_dir}/{key}{pred_image_ext}" for key in keys]
    gt_image_paths = [f"{gt_images_dir}/{key}{gt_image_ext}" for key in keys]
    fmask_paths = [f"{fmasks_dir}/{key}{fmask_ext}" if fmasks_dir is not None else None for key in keys]

    # multi-GPU evaluation
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    num_batches = len(gpu_ids)
    pred_image_path_batches = split_batches(pred_image_paths, num_batches)
    gt_image_path_batches = split_batches(gt_image_paths, num_batches)
    fmask_path_batches = split_batches(fmask_paths, num_batches)
    key_batches = split_batches(keys, num_batches)

    try:
        from easyvolcap.utils.parallel_utils import parallel_execution
    except ImportError:
        raise ImportError(
            "Please install EasyVolcap using `pip install git+https://github.com/zju3dv/EasyVolcap.git --no-deps`"
        )

    res_batches = parallel_execution(
        pred_image_paths=pred_image_path_batches,
        gt_image_paths=gt_image_path_batches,
        fmask_paths=fmask_path_batches,
        keys=key_batches,
        gpu_id=gpu_ids,
        action=evaluate_on_single_gpu,
        num_workers=len(gpu_ids),
        sequential=False,
    )

    # aggregate results
    metrics = {"mean": {}, "values": []}
    for res_batch in res_batches:
        metrics["values"].extend(res_batch)
    metrics["values"].sort(key=lambda x: x["key"])
    metrics["mean"] = {
        "psnr": round(torch.tensor([x["psnr"] for x in metrics["values"]]).mean().item(), 3),
        "ssim": round(torch.tensor([x["ssim"] for x in metrics["values"]]).mean().item(), 3),
        "lpips": round(torch.tensor([x["lpips"] for x in metrics["values"]]).mean().item(), 3),
    }

    if out_metrics_path is not None:
        save_json(out_metrics_path, metrics)
    return metrics
