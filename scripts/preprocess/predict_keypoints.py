import os
import fire
import torch
import subprocess

ckpt_root = os.environ["SAPIENS_CHECKPOINT_ROOT"]


def predict_keypoints(
    images_dir: str,
    out_kp2d_dir: str,
    fmasks_dir: str | None = None,
    sapiens_ckpt_path: str = f"{ckpt_root}/torchscript/pose/checkpoints/sapiens_2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2",
    detector_ckpt_path: str = f"{ckpt_root}/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
    gpu_ids: tuple[int, ...] | None = None,
    num_workers: int = 4,
):
    if gpu_ids is None:
        gpu_ids = tuple(range(torch.cuda.device_count()))

    print(os.path.dirname(__file__))
    subprocess.run(
        f"python sapiens/lite/demo/vis_pose.py \
            {sapiens_ckpt_path} --det-checkpoint {detector_ckpt_path} \
            --det-config sapiens/lite/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py \
            --images_dir {images_dir} --fmasks_dir {fmasks_dir} --output_dir {out_kp2d_dir} \
            --skip_exists --gpu_ids {','.join(map(str, gpu_ids))} --num_workers {num_workers}",
        cwd=os.path.dirname(__file__),
        check=False,
        shell=True,
    )


if __name__ == "__main__":
    # usage:
    # python scripts/preprocess/predict_keypoints.py --images_dir $DATADIR/images --fmasks_dir $DATADIR/fmasks --out_kp2d_dir $DATADIR/poses_2d
    fire.Fire(predict_keypoints)
