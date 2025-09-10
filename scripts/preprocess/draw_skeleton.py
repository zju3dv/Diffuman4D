import os
import cv2
import fire
import json
import math
import numpy as np
from glob import glob
from PIL import Image
from easyvolcap.utils.parallel_utils import parallel_execution

from sapiens.lite.demo.classes_and_palettes import (
    COCO_WHOLEBODY_KPTS_COLORS,
    COCO_WHOLEBODY_SKELETON_INFO,
    BLUE,
)


def score_to_color(rgb, score, low=0.5, high=0.9):
    score = np.clip(score, low, high)
    norm_score = (score - low) / (high - low)
    rgb = np.array(rgb, dtype=np.float32) * norm_score
    rgb = np.round(rgb, decimals=0).astype(np.uint8).tolist()
    return rgb


def draw_one_skeleton(
    kp2d_path,
    out_kpmap_path,
    kp2d_score_path=None,
    kp2d_canvas_shape=(1024, 1024),
    out_kpmap_shape=(1024, 1024),
    low_thr=0.5,
    high_thr=0.9,
    colors_info=COCO_WHOLEBODY_KPTS_COLORS,
    skeleton_info=COCO_WHOLEBODY_SKELETON_INFO,
    radius=2,
    thickness=2,
    image_quality=85,
    draw_face_keypoints=False,
    skip_exists=False,
):
    if skip_exists and os.path.exists(out_kpmap_path):
        try:
            Image.open(out_kpmap_path).verify()
            return
        except Exception as e:
            print(f"Error reading {out_kpmap_path}: {e}")

    # currently, we only support one instance per image
    kpts_dict = json.load(open(kp2d_path))["instance_info"][0]

    kpts = np.array(kpts_dict["keypoints"], dtype=np.float32)

    if kp2d_score_path is not None:
        # override the scores from the kp2d_score_path
        kpts_score_dict = json.load(open(kp2d_score_path))["instance_info"][0]
        scores = np.array(kpts_score_dict["keypoint_scores"], dtype=np.float32)
    elif "keypoint_scores" in kpts_dict:
        scores = np.array(kpts_dict["keypoint_scores"], dtype=np.float32)
    else:
        scores = np.ones(kpts.shape[0], dtype=np.float32)

    if "keypoint_depths" in kpts_dict:
        depths = np.array(kpts_dict["keypoint_depths"], dtype=np.float32)
    else:
        depths = np.zeros_like(scores)

    # update scores for invalid keypoints
    scores[kpts.min(axis=1) < 0] = 0.0

    # scale and shift the keypoints to match the output image shape
    # draw skeleton map at 2048p for anti-aliasing
    drawing_scale = 2048 / max(out_kpmap_shape)
    out_kpmap_shape = (np.array(out_kpmap_shape) * drawing_scale).astype(np.int32)
    kp2d_canvas_shape = np.array(kp2d_canvas_shape)
    scale_ratio = out_kpmap_shape.min() / kp2d_canvas_shape.min()
    kpts = kpts * scale_ratio
    kp2d_canvas_shape = kp2d_canvas_shape * scale_ratio
    kp2d_padding = (out_kpmap_shape.min() - kp2d_canvas_shape.min()) / 2
    kpts += kp2d_padding

    # kp2d_canvas_shape represents the shape of the canvas to draw the skeleton,
    # please ensure the canvas shape matches the input image shape of sapiens poses
    canvas = np.zeros(np.concatenate([out_kpmap_shape, [3]]), dtype=np.uint8)

    if colors_info is None or len(colors_info) != len(kpts):
        raise ValueError(
            f"the length of kpt_color ({len(colors_info)}) \
            does not matches that of keypoints ({len(kpts)})"
        )

    # add x links for the body
    skeleton_info.update(
        {
            65: dict(link=(5, 12), id=65, color=BLUE),  # left shoulder to right hip
            66: dict(link=(6, 11), id=66, color=BLUE),  # right shoulder to left hip
        }
    )

    # reweight the radius and thickness of the skeleton
    radius = int(round(radius * scale_ratio))
    thickness = int(round(thickness * scale_ratio))
    radius = np.ones(len(skeleton_info)) * radius
    thickness = np.ones(len(skeleton_info)) * thickness
    # highlight the major body parts
    radius[:25] *= 2
    thickness[:25] *= 2
    radius = radius.astype(np.int32)
    thickness = thickness.astype(np.int32)

    # draw skeleton
    lines = []
    for skid, link_info in skeleton_info.items():
        i1, i2 = link_info["link"]
        p1_score = scores[i1]
        p2_score = scores[i2]
        line_score = np.min((p1_score, p2_score))

        if line_score < low_thr:
            continue

        # draw skeleton
        p1_color = score_to_color(colors_info[i1], p1_score, low=low_thr, high=high_thr)
        p2_color = score_to_color(colors_info[i2], p2_score, low=low_thr, high=high_thr)
        line_color = score_to_color(link_info["color"], line_score, low=low_thr, high=high_thr)

        p1, p2 = kpts[i1], kpts[i2]
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        d1, d2 = float(depths[i1]), float(depths[i2])
        d = (d1 + d2) / 2

        lines.append(
            {
                "type": "line",
                "p1": (x1, y1),
                "p2": (x2, y2),
                "depth": d,
                "score": line_score,
                "p1_color": p1_color[::-1],
                "p2_color": p2_color[::-1],
                "line_color": line_color[::-1],
                "radius": radius[skid],
                "thickness": thickness[skid],
            }
        )

    if (depths != 0.0).any():
        # sort lines by depth
        # ideally, we should use z-buffer to pixel-wise sort the lines
        # here we naively sort the lines by average depth of the two endpoints
        lines = sorted(lines, key=lambda x: x["depth"], reverse=True)
    elif (scores != 1.0).any():
        # sort lines by score
        # lines with higher score are more likely to be at the front
        lines = sorted(lines, key=lambda x: x["score"])

    for line in lines:
        cv2.line(canvas, line["p1"], line["p2"], line["line_color"], line["thickness"])
        cv2.circle(canvas, line["p1"], line["radius"], line["p1_color"], -1)
        cv2.circle(canvas, line["p2"], line["radius"], line["p2_color"], -1)

    # draw face keypoints
    if draw_face_keypoints:
        for kid, kpt in enumerate(kpts):
            if not (23 < kid < 91):
                continue
            if scores[kid] < low_thr or colors_info[kid] is None:
                continue
            color = score_to_color(colors_info[kid], scores[kid], low=low_thr, high=high_thr)
            x, y = int(round(kpt[0])), int(round(kpt[1]))
            cv2.circle(canvas, (x, y), radius, color[::-1], -1)

    os.makedirs(os.path.dirname(out_kpmap_path), exist_ok=True)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(canvas)
    w, h = canvas.size
    canvas = canvas.resize((int(w / drawing_scale), int(h / drawing_scale)))
    canvas.save(out_kpmap_path, quality=image_quality)


def draw_skeleton(
    kp2d_dir: str,
    out_kpmap_dir: str,
    kp2d_score_dir: str | None = None,
    kp2d_canvas_shape: tuple[int, int] = (1024, 1024),
    out_kpmap_shape: tuple[int, int] = (1024, 1024),
    spa_labels: list[int] | None = None,
    tem_labels: list[int] | None = None,
    image_ext: str = ".webp",
    image_quality: int = 85,
    num_workers: int = 16,
    skip_exists: bool = False,
):
    if spa_labels is None:
        spa_labels = sorted(os.listdir(kp2d_dir))
    else:
        spa_labels = [f"{spa_label:02d}" for spa_label in spa_labels]
    if tem_labels is None:
        tem_labels = sorted(os.listdir(f"{kp2d_dir}/{spa_labels[0]}"))
        tem_labels = [tem_label.split(".")[0] for tem_label in tem_labels]
    else:
        tem_labels = [f"{tem_label:06d}" for tem_label in tem_labels]

    kp2d_paths = [f"{kp2d_dir}/{spa_label}/{tem_label}.json" for spa_label in spa_labels for tem_label in tem_labels]
    out_kpmap_paths = [p.replace(kp2d_dir, out_kpmap_dir).replace(".json", image_ext) for p in kp2d_paths]

    if kp2d_score_dir is not None:
        kp2d_score_paths = [p.replace(kp2d_dir, kp2d_score_dir) for p in kp2d_paths]
    else:
        kp2d_score_paths = [None] * len(kp2d_paths)

    parallel_execution(
        kp2d_paths,
        out_kpmap_paths,
        kp2d_score_paths,
        kp2d_canvas_shape=kp2d_canvas_shape,
        out_kpmap_shape=out_kpmap_shape,
        image_quality=image_quality,
        skip_exists=skip_exists,
        action=draw_one_skeleton,
        num_workers=num_workers,
        print_progress=True,
        desc="Drawing skeleton map",
        sequential=False,
    )


if __name__ == "__main__":
    # usage:
    # python scripts/preprocess/draw_skeleton.py --kp2d_dir $DATADIR/poses_2d --out_kpmap_dir $DATADIR/skeletons
    fire.Fire(draw_skeleton)
