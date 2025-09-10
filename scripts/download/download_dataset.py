import fire
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path
from typing import Iterator, Dict, Any
from huggingface_hub import snapshot_download

from easyvolcap.utils.console_utils import log


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def write_bytes(p: Path, data: bytes | None, overwrite=False):
    if data is None:
        raise ValueError(f"data is None for {p}")
    if p.exists() and not overwrite:
        return
    ensure_dir(p)
    p.write_bytes(data)


def write_text(p: Path, s: str | None, overwrite=False):
    if s is None:
        raise ValueError(f"string is None for {p}")
    if p.exists() and not overwrite:
        return
    ensure_dir(p)
    p.write_text(s, encoding="utf-8")


def iter_rows(parquet_path: Path, batch_size: int = 1024) -> Iterator[Dict[str, Any]]:
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size):
        bd = batch.to_pydict()
        if not bd:
            continue
        n = len(next(iter(bd.values())))
        for i in range(n):
            yield {k: v[i] for k, v in bd.items()}


def download_and_extract_dataset(
    repo_id: str = "krahets/dna_rendering_processed",
    out_dir: str = None,
    scenes: list[str] = None,
    types: list[str] = ["images", "fmasks", "skeletons", "poses_2d", "poses_3d", "cameras", "cameras_ccm_colmap"],
    overwrite: bool = False,
    batch_size: int = 1024,
):
    """
    Download per-scene Parquet from HF repo, and write back to the original structure:
    - scene/fmasks/<cam>/<frame>.png
    - scene/skeletons/<cam>/<frame>.webp
    - scene/poses_2d/<cam>/<frame>.json
    - scene/poses_3d/<frame>.json
    - scene/transforms.json, scene/sparse_pcd.ply
      scene/cameras/(intri|extri).yml, ccm/..., colmap/...

    Args:
        repo_id: The Hugging Face repository ID (https://huggingface.co/datasets/krahets/dna_rendering_processed).
        out_dir: The output dataset directory. Requires ~1TB disk space.
        scenes: The scenes (0007_01, ...) to download.
        types: The data types (fmasks, skeletons, poses_2d, poses_3d, cameras) to download.
        overwrite: Whether to overwrite the existing files.
        batch_size: The batch size.
    """

    if out_dir is None:
        user, repo = repo_id.split("/")
        out_dir = f"./data/datasets--{user}--{repo}"
    if scenes is None:
        scenes = "*"

    allow_patterns = []
    for scene in scenes:
        for typ in types:
            allow_patterns.append(f"{typ}/{scene}.parquet")

    out_data_dir = Path(out_dir)
    out_cache_dir = Path(out_dir + ".hf")

    log(f"Downloading dataset from '{repo_id}' to '{out_dir}'...")
    local_root = Path(
        snapshot_download(
            repo_id=repo_id, repo_type="dataset", allow_patterns=allow_patterns, local_dir=str(out_cache_dir)
        )
    )

    # directories for each split
    dirs = {typ: local_root / typ for typ in types}

    # cameras (scene-level)
    if "cameras" in types:
        for f in sorted(dirs["cameras"].glob("*.parquet")):
            for row in iter_rows(f, batch_size=batch_size):
                scene = str(row["scene"])
                scene_root = out_data_dir / scene
                # transforms.json / sparse_pcd.ply
                write_text(scene_root / "transforms.json", row["cam_ns"], overwrite)
                write_bytes(scene_root / "sparse_pcd.ply", row["sparse_pcd"], overwrite)
                # cameras/
                write_text(scene_root / "cameras/intri.yml", row["cam_evc_intri"], overwrite)
                write_text(scene_root / "cameras/extri.yml", row["cam_evc_extri"], overwrite)

                if "cameras_ccm_colmap" in types:
                    # cameras/ccm/
                    write_text(scene_root / "cameras/ccm/intri.yml", row["cam_ccm_intri"], overwrite)
                    write_text(scene_root / "cameras/ccm/extri.yml", row["cam_ccm_extri"], overwrite)
                    # cameras/colmap/sparse/0/
                    write_text(
                        scene_root / "cameras/colmap/sparse/0/intri.yml", row["cam_colmap_sparse_intri"], overwrite
                    )
                    write_text(
                        scene_root / "cameras/colmap/sparse/0/extri.yml", row["cam_colmap_sparse_extri"], overwrite
                    )
                    # cameras/colmap/dense/sparse/0/
                    write_text(
                        scene_root / "cameras/colmap/dense/sparse/0/intri.yml", row["cam_colmap_dense_intri"], overwrite
                    )
                    write_text(
                        scene_root / "cameras/colmap/dense/sparse/0/extri.yml", row["cam_colmap_dense_extri"], overwrite
                    )

    # poses_3d (scene + frame)
    if "poses_3d" in types:
        for f in sorted(dirs["poses_3d"].glob("*.parquet")):
            for row in iter_rows(f, batch_size=batch_size):
                scene = str(row["scene"])
                frame = str(row["frame"])
                write_text(out_data_dir / scene / "poses_3d" / f"{frame}.json", row["pose_3d"], overwrite)

    # images / fmasks / skeletons / poses_2d (scene + camera + frame)
    def _extract_data(key: str, reldir: str, ext: str, writer):
        d = dirs[reldir]
        for f in tqdm(sorted(d.glob("*.parquet")), desc=f"Extracting {key}"):
            for row in iter_rows(f, batch_size=batch_size):
                scene = str(row["scene"])
                camera = str(row["camera"])
                frame = str(row["frame"])
                writer(out_data_dir / scene / reldir / camera / f"{frame}.{ext}", row[key], overwrite)

    if "images" in types:
        _extract_data("image", "images", "webp", write_bytes)
    if "fmasks" in types:
        _extract_data("fmask", "fmasks", "png", write_bytes)
    if "skeletons" in types:
        _extract_data("skeleton", "skeletons", "webp", write_bytes)
    if "poses_2d" in types:
        _extract_data("pose_2d", "poses_2d", "json", write_text)

    log(f"✅ Dataset downloaded and extracted to: '{out_dir}'")


if __name__ == "__main__":
    # This script is for downloading and extracting the dataset from https://huggingface.co/datasets/krahets/dna_rendering_processed
    # requirements: pip install -U huggingface_hub datasets pyarrow pandas && pip install git+https://github.com/zju3dv/EasyVolcap.git --no-deps
    # usage: python scripts/download/download_dataset.py -h
    fire.Fire(download_and_extract_dataset)
