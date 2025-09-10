import os
import os.path as osp
import torch
from glob import glob
from huggingface_hub import snapshot_download
from torchvision.utils import save_image as save_grid
from torchvision.transforms.functional import to_pil_image, to_tensor, resize

from src.data.utils.data_utils import save_image, save_json
from src.data.utils.image_utils import restore_cropped_image, denorm_vae_tensor
from src.diffusers.pipelines.diffuman4d.pipeline_diffuman4d import Diffuman4DPipeline
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def load_pipelines(
    repo_id: str = "krahets/Diffuman4D",
    model_dir: str = "./models/krahets-Diffuman4D",
    torch_dtype: "str" = "bf16",
    gpu_ids: list[int] = None,
):
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
        log.info(f"Found {len(gpu_ids)} CUDA devices.")

    # download models
    if torch_dtype == "fp16":
        allow_patterns = ["*.json", "*model.fp16.safetensors"]
        torch_dtype = torch.float16
    elif torch_dtype == "bf16":
        allow_patterns = ["*.json", "*model.safetensors"]
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}. Supported types are 'bf16' and 'fp16'.")

    try:
        snapshot_download(repo_id, local_dir=model_dir, allow_patterns=allow_patterns)
        log.info(f"Downloaded model from {repo_id} to {model_dir}")
    except Exception as e:
        log.error(f"Failed to download model from {repo_id} to {model_dir}: {e}. Skipping model loading.")

    # load models
    pipelines = []
    for gpu_id in gpu_ids:
        pipeline = Diffuman4DPipeline.from_pretrained(model_dir, torch_dtype=torch_dtype)
        pipeline.to(f"cuda:{gpu_id}")
        pipeline.set_progress_bar_config(disable=True)
        pipelines.append(pipeline)
        log.info(f"Loaded pipeline from {model_dir} of {torch_dtype} to cuda:{gpu_id}")
    return pipelines


def save_sampling_results(
    sample: dict[str, any],
    output_dir: str = "./results",
    save_image_grid: bool = True,
    save_output_image: bool = True,
    save_crop_param: bool = False,
    image_ext: str = ".jpg",
    image_quality: int = 90,
    max_image_size: int = 8192,
):
    output_images = sample["images"]
    input_indices = sample["input_indices"]
    target_indices = sample["target_indices"]
    input_images = denorm_vae_tensor(sample["pixel_values"])

    # save snapshots
    if save_image_grid:
        # calculate L1 error
        image_errors = (output_images - input_images).abs().clamp(0, 1)
        # transparent the input images
        output_images[input_indices, ...] *= 0.2

        if sample["skeletons"] is not None:
            # blend skeletons with images
            skeletons = denorm_vae_tensor(sample["skeletons"])
            skeletons = skeletons * 0.8 + input_images * 0.2
            image_grid = torch.cat([skeletons, input_images, output_images, image_errors])
        else:
            image_grid = torch.cat([input_images, output_images, image_errors])

        # downscale the image grid and use .webp to save disk space
        max_size = min(
            max_image_size // len(output_images),
            max(image_grid.shape[-2:]),
        )
        image_grid = resize(image_grid, max_size)

        image_grid_path = f'{output_dir}/grids/alt{sample["alt"]}_{"spa" if sample["domain"] == "temporal" else "tem"}{sample["domain_label"]}.webp'
        os.makedirs(osp.dirname(image_grid_path), exist_ok=True)
        save_grid(image_grid, image_grid_path, nrow=len(output_images), padding=2, pad_value=0)

    # save the images and crops
    output_images[input_indices] = input_images[input_indices]
    for i, (output_image, crop, (_, spa_label, tem_label)) in enumerate(
        zip(output_images, sample["crops"], sample["labels"])
    ):
        if save_output_image:
            image_path = f"{output_dir}/images/{spa_label}/{tem_label}{image_ext}"
            # skip the noised target images
            if not sample["fully_denoised"][i] and i in target_indices:
                continue
            # skip the saved images (input images)
            if osp.isfile(image_path):
                continue

            image = to_pil_image(output_image)
            image = restore_cropped_image(image, crop)
            save_image(path=image_path, image=image, quality=image_quality)

        if save_crop_param:
            save_json(path=f"{output_dir}/crops/{spa_label}/{tem_label}.json", data=crop)


def check_sampling_results(spa_labels, tem_labels, output_dir: str):
    # check saved images
    num_saved_images = len(glob(f"{output_dir}/images/**/*.*"))
    num_expected_images = len(spa_labels) * len(tem_labels)
    if num_saved_images != num_expected_images:
        log.warning(
            "Found incomplete sampling results: "
            f"Num of saved images: {num_saved_images} != Num of expected images: {num_expected_images} in {output_dir}."
        )
        return False

    log.info(f"Found complete results in {output_dir}.")
    return True
