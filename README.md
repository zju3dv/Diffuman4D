<h1 align="center">
  <a href="(https://diffuman4d.github.io/">
    <img src="assets/images/logo_title.png" width="280" alt="Diffuman4D"></a>
</h1>

<p align="center">
  <a href="https://diffuman4d.github.io/"><strong>Project Page</strong></a>
  &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2507.13344"><strong>Paper</strong></a>
</p>

> The official repo for "Diffuman4D: 4D Consistent Human View Synthesis from Sparse-View Videos with Spatio-Temporal Diffusion Models".

<img src="assets/images/teaser_dna.gif" width="100%" alt="teaser">

<p align="center">Diffuman4D enables high-fidelity free-viewpoint rendering of human performances from sparse-view videos.</p>

## Interactive Demo

[Click here](https://www.4dv.ai/viewer/diffuman4d_fdvai_dance_colored?showdemo=diffuman4d) to experience immersive 4DGS rendering.

<a href="https://www.4dv.ai/viewer/diffuman4d_fdvai_dance_colored?showdemo=diffuman4d"><img src="assets/images/interactive_demo_preview.gif" width="100%" alt="interactive_demo_preview"></a>

## Quick Start

**1. Install.** For inference and data preprocessing, please install the environment via:

```sh
conda create -n diffuman4d python=3.12
conda activate diffuman4d
# for inference
pip install -r requirements.txt
# for 3D/4D reconstruction and data processing
pip install git+https://github.com/zju3dv/EasyVolcap.git --no-deps
```

**2. Download Example Data.** Please download the [example test data](https://huggingface.co/datasets/krahets/diffuman4d_example) using:

```sh
python scripts/download/download_dataset.py --repo_id "krahets/diffuman4d_example" --types='["images", "fmasks", "skeletons", "cameras"]'
```

The extracted data is structured as `{scene_label}/{data_type}/{camera_label}/{frame_label}{file_ext}`:

```sh
└── 0023_06 # scene label
    ├── fmasks # foreground masks
    │   ├── 00 # camera label
    │   │   ├── 000000.png # frame label
    │   │   ├── 000001.png
    │   │   └── ... (148 more items)
    │   └── 01
    │       ├── 000000.png
    │       ├── 000001.png
    │       └── ... (148 more items)
    │   └── ... (46 more items)
    ├── images # rgb images
    │   ├── 00
    │   └── ... (47 more items)
    ├── skeletons # skeleton maps
    │   ├── 00
    │   └── ... (47 more items)
    ├── sparse_pcd.ply # sparse point cloud
    └── transforms.json # cameras in nerfstudio format
```

> [!tip]
> If you want to test the model on more DNA-Rendering scenes, please see the [Dataset section](#processed-dna-rendering-dataset).

**3. (Optional) Download Pretrained Model.** The inference code will attempt to download the model from Hugging Face. If you encounter network issues, please manually download [the model](https://huggingface.co/krahets/Diffuman4D) to `./models/` via:

```sh
hf download krahets/Diffuman4D --local-dir ./models/models--krahets--Diffuman4D
```

**4. Inference.** Run inference with the following code, and the sampling results will be saved in `./output/results/dna_rendering/0023_06`. It is recommended to run `exp=demo_3d` or `exp=demo_4d_tiny` if using a single-GPU server for quicker testing.

```sh
# generate a tiny 4D image grid (4 input cameras * 15 frames -> 44 cameras * 15 frames)
python inference.py exp=demo_4d_tiny data.scene_label=0023_06 data.data_dir=./data/datasets--krahets--diffuman4d_example

# generate a 3D image grid (4 input cameras -> 44 cameras)
python inference.py exp=demo_3d data.scene_label=0023_06 data.data_dir=./data/datasets--krahets--diffuman4d_example

# generate entire 4D image grid (4 input cameras * 150 frames -> 44 cameras * 150 frames)
python inference.py exp=demo_4d data.scene_label=0023_06 data.data_dir=./data/datasets--krahets--diffuman4d_example
```

**5. Reconstruct 3DGS Model.** Please first install [nerfstudio](https://github.com/nerfstudio-project/nerfstudio/), then train the human 3DGS model via:

```sh
ns-train splatfacto --data "./output/results/dna_rendering_tiny/0023_06/transforms.json"
```

**6. Reconstruct 4DGS Model.** Since [LongVolcap](https://zju3dv.github.io/longvolcap/) has not been open-sourced, we will attempt to provide the alternative [4D-Gaussian-Splatting](https://github.com/fudan-zvg/4d-gaussian-splatting) reconstruction scripts.

## Processed DNA-Rendering Dataset

To enable model training, we meticulously process the [DNA-Rendering dataset](https://dna-rendering.github.io/index.html) by recalibrating camera parameters, optimizing image color correction matrices (CCMs), predicting foreground masks, and estimating human skeletons.

To promote future research in the field of human-centric 3D/4D generation, we have open-sourced our re-annotated labels for the DNA-Rendering dataset in [dna_rendering_processed](https://huggingface.co/datasets/krahets/dna_rendering_processed), which includes 1000+ human multi-view video sequences. Each sequence contains 48 cameras, 225 (or 150) frames, totaling 10 million images.

<img src="assets/images/dna_rendering_processed.gif" width="100%"></img>

> [!note]
> If you find our method or dataset helpful, please give us a Star ⭐ and [cite our work](#cite). Thank you!

### Dataset Download

Before starting, please install the requirements via:

```sh
pip install -U huggingface_hub datasets pyarrow pandas
pip install git+https://github.com/zju3dv/EasyVolcap.git --no-deps
```

Download re-annotated labels (foreground masks, 2D skeletons, 3D skeletons, camera parameters) for the DNA-Rendering dataset:

1. Please concurrently **(1)** fill out [this form](https://docs.google.com/forms/d/1v-X0bnl5GUO9ewYW5eY2wk-yrwQx4_u2lEWbLIao4VU) and **(2)** request access to the dataset on [this page](https://huggingface.co/datasets/krahets/dna_rendering_processed).
2. Download the dataset using [the script](scripts/download/download_dataset.py) below.

```sh
# Download and extract the entire dataset
python scripts/download/download_dataset.py --out_dir "./data/dna_rendering_processed"

# Download specific scenes and data types
python scripts/download/download_dataset.py --out_dir "./data/dna_rendering_processed" --scenes '["0007_01"]' --types '["fmasks"]'
```

Download the corresponding RGB images:

1. Download the raw data from the [official DNA-Rendering website](https://dna-rendering.github.io/inner-download.html).
2. Extract the RGB images from the archived dataset files using [the script](scripts/download/extract_dnar_images.py) below.

```sh
# Extract images from all scenes in `processed_root`. You may replace `raw_root` with your own path.
# You can also specify scenes by passing `--scenes '["0007_01"]'`
python scripts/download/extract_dnar_images.py --raw_root "./data/dna_rendering_release_data" --processed_root "./data/dna_rendering_processed"
```

The dataset file structure looks like:

```sh
└── 0007_01 # scene label
    ├── cameras # intermediate camera files
    │   ├── ccm # easyvolcap cameras used to correct image color
    │   ├── colmap # easyvolcap cameras used to undistort images
    │   ├── intri.yml # easyvolcap camera intrinsics
    │   └── extri.yml # easyvolcap camera extrinsics
    ├── fmasks # foreground masks
    │   ├── 00 # camera label
    │   │   ├── 000000.png # frame label
    │   │   └── 000001.png
    │   │   └── ... (148 more items)
    │   └── 01
    │       ├── 000000.png
    │       └── 000001.png
    │       └── ... (148 more items)
    │   └── ... (46 more items)
    ├── images # rgb images
    │   ├── 00
    │   │   ├── 000000.webp
    │   │   └── ... (149 more items)
    │   └── ... (47 more items)
    ├── poses_2d # 2D projections of poses_3d
    │   ├── 00
    │   │   ├── 000000.json
    │   │   └── ... (149 more items)
    │   └── ... (47 more items)
    ├── poses_3d # 3D poses triangulated from Sapiens 2D poses
    │   ├── 000000.json
    │   └── ... (149 more items)
    ├── skeletons # rgb maps drawn from poses_2d
    │   ├── 00
    │   │   ├── 000000.webp
    │   │   └── ... (149 more items)
    │   └── ... (47 more items)
    ├── sparse_pcd.ply # sparse point cloud of the first frame
    └── transforms.json # cameras in nerfstudio format
└── ... (1037 more items)
```

> [!tip]
> nerfstudio use the OpenGL/Blender coordinate convention for cameras. If you need the Colmap/OpenCV coordinate convention, please flip the Y and Z axes of the `transform_matrix`. For more details, see the [nerfstudio documentation](https://docs.nerf.studio/quickstart/data_conventions.html).

### Custom Data Processing

1. **Install**. You can use the inference environment to run all data processing scripts except `predict_keypoints.py`. If you want to predict keypoints with your data, please install `sapiens-lite` following the guidance in [lite/README.md](https://github.com/facebookresearch/sapiens/blob/main/lite/README.md) and [POSE_README.md](https://github.com/facebookresearch/sapiens/blob/main/lite/docs/POSE_README.md). Note that sapiens-lite requires pytorch<=2.4.1 (See this [issue](https://github.com/open-mmlab/mmdetection/issues/12008)). It is recommended to create a new conda environment to run it.

2. **Prepare the data**. Organize your data in the following directory structure. Note that:

   - The recorded multi-view video data must be time-synchronized (i.e., the set of images under the same frame label is captured at the same moment).
   - It is required to add a new element `camera_label` for each frame in `transforms.json` to indicate the corresponding camera.

```sh
{YOUR_DATA_DIR}
  ├── images # foreground masks
  │   ├── 00 # camera label
  │   │   ├── 000000.jpg # frame label
  │   │   ├── 000001.jpg
  │   │   └── ... (m more items)
  │   └── 01
  │       ├── 000000.jpg
  │       ├── 000001.jpg
  │       └── ... (m more items)
  │   └── ... (n more items)
  └── transforms.json # cameras in nerfstudio format
```

3. **Process the data**. Run the following script to preprocess the data, including predicting foreground masks, predicting 2D keypoints using Sapiens, triangulating and projecting keypoints, and drawing skeletons.

```sh
# Run all preprocessing scripts
bash scripts/preprocess/preprocess.sh --data_dir YOUR_DATA_DIR

# Run specific actions
bash scripts/preprocess/preprocess.sh --data_dir YOUR_DATA_DIR --actions triangulate_skeleton,draw_skeleton
```

## Todos

- [x] Release project page and paper.
- [x] Release inference code and models.
- [x] Release processed DNA-Rendering dataset.
- [x] Release custom data preprocessing scripts.

## Cite

```
@inproceedings{jin2025diffuman4d,
  title={Diffuman4D: 4D Consistent Human View Synthesis from Sparse-View Videos with Spatio-Temporal Diffusion Models},
  author={Jin, Yudong and Peng, Sida and Wang, Xuan and Xie, Tao and Xu, Zhen and Yang, Yifan and Shen, Yujun and Bao, Hujun and Zhou, Xiaowei},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
