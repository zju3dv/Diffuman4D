import os
import fire
import torch
import threading
from PIL import Image
from glob import glob
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor

from easyvolcap.utils.console_utils import tqdm
from easyvolcap.utils.parallel_utils import parallel_execution


# borrowed from https://huggingface.co/ZhengPeng7/BiRefNet
image_size = (1024, 1024)
transform_image = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_model(model_name, device="cuda"):
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)
    torch.set_float32_matmul_precision(["high", "highest"][0])
    model.to(device)
    model.eval()
    model.half()
    return model


def extract_object(images, model, sema):
    input_images = []
    for image in images:
        input_image = transform_image(image)
        input_images.append(input_image)
    input_images = torch.stack(input_images).to(model.device).half()

    # batch inference
    with sema:
        with torch.no_grad():
            preds = model(input_images)[-1].sigmoid().cpu()

    fmasks = []
    for image, pred in zip(images, preds):
        pred_pil = to_pil_image(pred)
        fmask = pred_pil.resize(image.size).convert("L")
        fmasks.append(fmask)
    return fmasks


def inference_batch(image_paths, fmask_paths, image_alpha_paths, model, sema, rotate_clockwise, skip_exists):
    if skip_exists:
        is_completed = True
        for fmask_path in fmask_paths:
            if not os.path.exists(fmask_path):
                is_completed = False
                break
            try:
                Image.open(fmask_path).verify()
            except Exception as e:
                print(f"Error loading {fmask_path}: {e}")
                is_completed = False
                break
        if is_completed:
            return

    # load images
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        if rotate_clockwise != 0:
            image = image.rotate(-rotate_clockwise, expand=True)
        images.append(image)

    # inference
    fmasks = extract_object(images, model, sema)

    for image, fmask, fmask_path, image_alpha_path in zip(images, fmasks, fmask_paths, image_alpha_paths):
        if rotate_clockwise != 0:
            fmask = fmask.rotate(rotate_clockwise, expand=True)
        os.makedirs(os.path.dirname(fmask_path), exist_ok=True)
        fmask.save(fmask_path)

        if image_alpha_path is not None:
            os.makedirs(os.path.dirname(image_alpha_path), exist_ok=True)
            image = image.rotate(rotate_clockwise, expand=True)
            image.putalpha(fmask)
            image.save(image_alpha_path)


def remove_background(
    images_dir: str,
    out_fmasks_dir: str,
    out_images_alpha_dir: str | None = None,
    model_name: str = "ZhengPeng7/BiRefNet",
    image_ext: str = ".webp",
    mask_ext: str = ".png",
    rotate_clockwise: int = 0,
    batch_size: int = 8,
    num_workers: int = 4,
    skip_exists: bool = False,
    gpu_ids: tuple[int, ...] | None = None,
):
    """
    Predict foreground masks for all images in the given directory.

    Args:
        model_name: str
            "briaai/RMBG-2.0": for general background removal.
            "ZhengPeng7/BiRefNet": for general background removal.
            "ZhengPeng7/BiRefNet": for human segmentation.
        batch_size: int
            The number of images to process in a single batch. Recommended to be GPU memory // 2.
    """
    if gpu_ids is None:
        gpu_ids = tuple(range(torch.cuda.device_count()))

    # prepare paths
    image_paths = sorted(glob(f"{images_dir}/**/*{image_ext}", recursive=True))
    fmask_paths = [p.replace(image_ext, mask_ext).replace(images_dir, out_fmasks_dir) for p in image_paths]
    if out_images_alpha_dir is not None:
        image_alpha_paths = [
            p.replace(image_ext, ".png").replace(images_dir, out_images_alpha_dir) for p in image_paths
        ]
    else:
        image_alpha_paths = [None] * len(image_paths)

    # load models
    models = []
    semas = []
    for gpu_id in tqdm(gpu_ids, desc=f"Loading '{model_name}' to cuda:{gpu_ids}"):
        model = load_model(model_name, f"cuda:{gpu_id}")
        models.append(model)
        # prevent CUDA OOM
        sema = threading.Semaphore(1)
        semas.append(sema)

    # split batches
    def split_batch(paths, batch_size):
        return [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]

    image_path_batches = split_batch(image_paths, batch_size)
    fmask_path_batches = split_batch(fmask_paths, batch_size)
    image_alpha_path_batches = split_batch(image_alpha_paths, batch_size)
    num_batches = len(image_path_batches)
    models_batch = (models * num_batches)[:num_batches]
    semas_batch = (semas * num_batches)[:num_batches]

    parallel_execution(
        image_path_batches,
        fmask_path_batches,
        image_alpha_path_batches,
        models_batch,
        semas_batch,
        rotate_clockwise,
        skip_exists,
        action=inference_batch,
        sequential=False,
        num_workers=num_workers * len(gpu_ids),
        print_progress=True,
        desc="Predicting foreground masks",
    )


if __name__ == "__main__":
    # usage:
    # python scripts/preprocess/remove_background.py --images_dir $DATADIR/images --out_fmasks_dir $DATADIR/fmasks --model_name ZhengPeng7/BiRefNet
    fire.Fire(remove_background)
