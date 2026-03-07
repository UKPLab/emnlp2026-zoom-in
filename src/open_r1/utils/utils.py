from qwen_vl_utils import smart_resize

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union
import random
from dataclasses import dataclass

import math
import numpy as np

from PIL import Image


QWEN2_5_MIN_PIXELS = 4*28*28
QWEN2_5_MAX_PIXELS = 16*1024*28*28

def get_resized_image_scales(height:int, width:int, min_pixels:int=None, max_pixels:int=None):

    if min_pixels is None:
        min_pixels = QWEN2_5_MIN_PIXELS
    if max_pixels is None:
        max_pixels = QWEN2_5_MAX_PIXELS


    resized_height, resized_width = smart_resize(height=height,
                                                 width=width,
                                                 min_pixels=min_pixels,
                                                 max_pixels=max_pixels)

    return resized_height, resized_width

def basic_iou_target_fn(step, start:float, end:float, increase: str, max_value: float = None):
    if max_value is None:
        max_value = 1.0
    if increase == "linear":
        div = 1 if end == start else end - start
        if step < start:
            return 0
        if step > end:
            return 1
        return max_value * float(step - start) / float(div)

def calculate_iou(box1,box2):
    return calculate_overlap_metrics(box1,box2)[0]

def calculate_overlap_metrics(box1, box2):
    """
    Calculate IoU of two bounding boxes [x1, y1, x2, y2].
    Uses NumPy for a fast, "out-of-the-box" vectorizable approach.
    box1 is prediction
    box2 is ground truth
    """
    box1 = np.array(box1)
    box2 = np.array(box2)

    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0, 0.0, 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - intersection_area

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = (intersection_area / union) if union > 0.0 else 0.0
    precision = (intersection_area / box1_area) if box1_area > 0.0 else 0.0
    recall = (intersection_area / box2_area) if box2_area > 0.0 else 0.0

    return iou,precision,recall



BBoxIn = Tuple[float, float, float, float]
BBoxOut = Union[Tuple[float, float, float, float], Tuple[int, int, int, int]]


@dataclass(frozen=True)
class BoxI:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def _clamp01(v: float) -> float:
    return min(max(0.0, v), 1.0)


def _iou_i(ref: BoxI, cand: BoxI, metric="iou") -> float:
    ix1 = max(ref.x1, cand.x1)
    iy1 = max(ref.y1, cand.y1)
    ix2 = min(ref.x2, cand.x2)
    iy2 = min(ref.y2, cand.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = ref.area + cand.area - inter

    if metric == "iou":
        overlap = inter / union if union > 0 else 0.0
    elif metric == "recall":
        overlap = inter / ref.area if ref.area > 0 else 0.0
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return overlap




def _infer_mode(bbox_2d: BBoxIn, bbox_type: Optional[str]) -> str:
    """
    Match zoom_in():
    - if bbox_type=="relative": relative
    - if bbox_type=="absolute": absolute-resized
    - else: heuristic: if all < 1 => relative else absolute-resized
    """
    if bbox_type == "relative":
        return "relative"
    if bbox_type == "absolute":
        return "absolute_resized"
    x1, y1, x2, y2 = bbox_2d
    return "relative" if (x1 < 1 and y1 < 1 and x2 < 1 and y2 < 1) else "absolute_resized"


def _effective_padding(
    padding: Tuple[float, float],
    adaptive_padding_threshold: Optional[float],
    input_w: float,
    input_h: float,
) -> Tuple[float, float]:
    if adaptive_padding_threshold is None:
        return padding
    tr = (adaptive_padding_threshold / input_w, adaptive_padding_threshold / input_h)
    return (min(padding[0], tr[0]), min(padding[1], tr[1]))


def _normalized_bbox_like_zoom_in(
    bbox_2d: Union[BBoxIn, Tuple[int, int, int, int]],
    *,
    padding: Tuple[float, float],
    bbox_type: Optional[str],
    input_w: float,
    input_h: float,
) -> Tuple[float, float, float, float]:
    """
    Re-implement zoom_in() bbox interpretation + padding + clamp to [0,1],
    returning the final normalized box used for cropping.
    """
    x1, y1, x2, y2 = bbox_2d

    if bbox_type == "relative":
        if x1 < 1 and y1 < 1 and x2 < 1 and y2 < 1:
            nx1 = float(x1) - padding[0]
            ny1 = float(y1) - padding[1]
            nx2 = float(x2) + padding[0]
            ny2 = float(y2) + padding[1]
        else:
            raise ValueError("bbox_type='relative' but bbox values are not < 1.")
    elif bbox_type == "absolute":
        if isinstance(x1, int) and isinstance(y1, int) and isinstance(x2, int) and isinstance(y2, int):
            nx1 = float(x1) / input_w - padding[0]
            ny1 = float(y1) / input_h - padding[1]
            nx2 = float(x2) / input_w + padding[0]
            ny2 = float(y2) / input_h + padding[1]
        else:
            raise ValueError("bbox_type='absolute' but bbox values are not ints.")
    else:
        if x1 < 1 and y1 < 1 and x2 < 1 and y2 < 1:
            nx1 = float(x1) - padding[0]
            ny1 = float(y1) - padding[1]
            nx2 = float(x2) + padding[0]
            ny2 = float(y2) + padding[1]
        else:
            nx1 = float(x1) / input_w - padding[0]
            ny1 = float(y1) / input_h - padding[1]
            nx2 = float(x2) / input_w + padding[0]
            ny2 = float(y2) / input_h + padding[1]

    return (_clamp01(nx1), _clamp01(ny1), _clamp01(nx2), _clamp01(ny2))


def _pixel_box_from_norm(norm: Tuple[float, float, float, float], img_w: int, img_h: int) -> BoxI:
    nx1, ny1, nx2, ny2 = norm
    # match zoom_in(): int(norm * size)
    x1 = int(nx1 * img_w)
    y1 = int(ny1 * img_h)
    x2 = int(nx2 * img_w)
    y2 = int(ny2 * img_h)

    # clamp and fix ordering
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(1, min(x2, img_w))
    y2 = max(1, min(y2, img_h))
    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    return BoxI(x1, y1, x2, y2)


def _invert_padding_best_effort(
    final_norm: Tuple[float, float, float, float],
    padding: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    """
    Want input_norm so that zoom_in input_norm (+/- padding) ~= final_norm.
    """
    nx1, ny1, nx2, ny2 = final_norm
    px, py = padding
    in_x1 = _clamp01(nx1 + px)
    in_y1 = _clamp01(ny1 + py)
    in_x2 = _clamp01(nx2 - px)
    in_y2 = _clamp01(ny2 - py)
    if in_x2 <= in_x1:
        in_x2 = min(1.0, in_x1 + 1e-6)
    if in_y2 <= in_y1:
        in_y2 = min(1.0, in_y1 + 1e-6)
    return (in_x1, in_y1, in_x2, in_y2)


def _emit_bbox_like_input(
    *,
    mode: str,
    input_w: int,
    input_h: int,
    in_norm: Tuple[float, float, float, float],
) -> BBoxOut:
    x1, y1, x2, y2 = in_norm
    x1, y1, x2, y2 = _clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)

    if mode == "relative":
        return (float(x1), float(y1), float(x2), float(y2))

    ax1 = int(round(x1 * input_w))
    ay1 = int(round(y1 * input_h))
    ax2 = int(round(x2 * input_w))
    ay2 = int(round(y2 * input_h))

    if ax2 <= ax1:
        ax2 = ax1 + 1
    if ay2 <= ay1:
        ay2 = ay1 + 1

    ax1 = max(0, min(ax1, input_w - 1))
    ay1 = max(0, min(ay1, input_h - 1))
    ax2 = max(1, min(ax2, input_w))
    ay2 = max(1, min(ay2, input_h))

    return (ax1, ay1, ax2, ay2)


def _sample_candidate_size_target_aware(
    rng: random.Random,
    ref: BoxI,
    img_w: int,
    img_h: int,
    t: float,
) -> Tuple[int, int]:
    """
    Sample candidate (w2, h2) in pixels. Target-aware so the sampler remains fast
    across t in [0,1].
    """
    w1, h1 = ref.w, ref.h
    A1 = max(1, w1 * h1)

    # As t -> 1, sizes should be near ref to avoid impossible high-IoU constraints.
    # As t -> 0, sizes can vary a lot without harming feasibility.
    sigma_area = 0.9 * (1.0 - t) + 0.08  # heuristic
    sigma_ar = 0.55  # aspect ratio spread

    for _ in range(200):
        log_r = rng.gauss(0.0, sigma_area)
        r = math.exp(log_r)
        A2 = int(max(1, min(img_w * img_h, round(A1 * r))))

        log_ar = rng.gauss(0.0, sigma_ar)
        ar = math.exp(log_ar)  # w/h

        w2 = int(round(math.sqrt(A2 * ar)))
        h2 = int(round(math.sqrt(A2 / ar)))

        # clamp to image bounds and avoid degenerate tiny boxes
        w2 = max(2, min(img_w, w2))
        h2 = max(2, min(img_h, h2))

        # slight correction to keep area near A2 after clamping
        if w2 * h2 < 1:
            continue
        return w2, h2

    # fallback: ref size clamped
    return max(2, min(img_w, ref.w)), max(2, min(img_h, ref.h))


def _propose_box_with_target_iou(
    rng: random.Random,
    ref: BoxI,
    img_w: int,
    img_h: int,
    t: float,
) -> Optional[BoxI]:
    """
    Propose a final pixel box aiming at IoU≈t using size sampling + overlap solving.
    Returns None if it can't place a valid box this try.
    """
    w1, h1 = ref.w, ref.h
    A1 = w1 * h1
    if A1 <= 0:
        return None

    w2, h2 = _sample_candidate_size_target_aware(rng, ref, img_w, img_h, t)
    A2 = w2 * h2

    # Required intersection area for target IoU:
    # I = t*(A1 + A2) / (1 + t)
    if t >= 1.0:
        I_req = float(min(A1, A2))
    else:
        I_req = (t * (A1 + A2)) / (1.0 + t)
    I_req = max(0.0, min(float(min(A1, A2)), I_req))

    max_ow = min(w1, w2)
    max_oh = min(h1, h2)
    if max_ow <= 0 or max_oh <= 0:
        return None

    # Pick overlap width/height that (approximately) satisfy ow*oh ~= I_req
    # Bias away from extreme skinny overlaps for numerical stability.
    for _ in range(300):
        if I_req == 0.0:
            # aim for (near) disjoint by making at least one overlap dimension 0
            ow = 0
            oh = rng.randint(0, max_oh)
        else:
            # choose overlap width as a fraction of max_ow
            u_min = 0.15 if t > 0.0 else 0.05
            u_max = 1.0
            ow = int(round(rng.uniform(u_min, u_max) * max_ow))
            ow = max(1, min(max_ow, ow))
            oh = int(round(I_req / ow))
            if oh < 1 or oh > max_oh:
                continue

        # For 1D intervals: overlap = (w1+w2)/2 - |dx|
        dx_abs = int(round((w1 + w2) / 2.0 - ow))
        dy_abs = int(round((h1 + h2) / 2.0 - oh))
        dx_abs = max(0, dx_abs)
        dy_abs = max(0, dy_abs)

        sx = -1 if rng.random() < 0.5 else 1
        sy = -1 if rng.random() < 0.5 else 1
        dx = sx * dx_abs
        dy = sy * dy_abs

        cx2 = ref.cx + dx
        cy2 = ref.cy + dy

        x1 = int(round(cx2 - w2 / 2.0))
        y1 = int(round(cy2 - h2 / 2.0))
        x2 = x1 + w2
        y2 = y1 + h2
        cand = BoxI(x1, y1, x2, y2)

        # Reject if out of bounds; clamping would change IoU in hard-to-control ways.
        if cand.x1 < 0 or cand.y1 < 0 or cand.x2 > img_w or cand.y2 > img_h:
            continue

        return cand

    return None

def _propose_box_free_floating_iou0(
    rng: random.Random,
    ref: BoxI,
    img_w: int,
    img_h: int,
    min_side: int,
    overlap_metric: str = "iou"
) -> Optional[BoxI]:
    """
    For target_iou == 0: sample a box uniformly in the image (free-floating),
    preferring disjoint (IoU==0) candidates. This avoids "sharing an edge" artifacts
    caused by anchor-based constructions.

    Returns a pixel box, or None if sampling is impossible (very small images).
    """
    if img_w < 2 or img_h < 2:
        return None

    # Choose size with a broad distribution; keep within image bounds.
    # You can tune these ranges to control variety.
    min_w = max(2, min(min_side, img_w))
    min_h = max(2, min(min_side, img_h))

    # Cap max size so it's easier to be disjoint (huge boxes make IoU=0 impossible).
    max_w = max(min_w, int(0.6 * img_w))
    max_h = max(min_h, int(0.6 * img_h))

    for _ in range(400):
        w2 = rng.randint(min_w, max_w)
        h2 = rng.randint(min_h, max_h)
        x1 = rng.randrange(0, max(1, img_w - w2 + 1))
        y1 = rng.randrange(0, max(1, img_h - h2 + 1))
        cand = BoxI(x1, y1, x1 + w2, y1 + h2)
        if _iou_i(ref, cand, metric=overlap_metric) == 0.0:
            return cand

    # If we couldn't find a perfectly disjoint one quickly, return a random box anyway;
    # outer rejection sampling will filter by tolerance.
    w2 = rng.randint(min_w, max_w)
    h2 = rng.randint(min_h, max_h)
    x1 = rng.randrange(0, max(1, img_w - w2 + 1))
    y1 = rng.randrange(0, max(1, img_h - h2 + 1))
    return BoxI(x1, y1, x1 + w2, y1 + h2)


def generate_bbox_2d_new_close_iou_targeted(
    bbox_2d: BBoxIn,
    target_iou: float,
    *,
    image_size: Optional[Tuple[int, int]] = None,   # (W,H) test mode
    image_path: Optional[str] = None,               # optional convenience
    padding: Tuple[float, float] = (0.1, 0.1),
    min_pixels: int = 400_000,
    max_pixels: int = 4_000_000,
    bbox_type: Optional[str] = None,
    adaptive_padding_threshold: Optional[float] = 600.0,
    tol: float = 0.1,
    other_bboxes: Optional[list[BBoxIn]] = None,
    other_bbox_threshold: Optional[float] = 0.05,
    max_tries: int = 5000,
    min_crop_size: int = 28,
    seed: Optional[int] = None,
    return_debug: bool = False,
    same_digit_number: bool = False, # if true, enforces that the number of digits for all 4 pixels in bbox_2d and bbox_2d_new are the same
    overlap_metric: str = "iou"
) -> Union[BBoxOut, Tuple[BBoxOut, dict]]:
    """
    Target-aware proposal + rejection sampling to get realized IoU close to target_iou.

    Returns bbox_2d_new in the same representation that zoom_in() will interpret for bbox_2d:
      - relative floats in [0,1] OR
      - absolute ints in resized-space (input_width/input_height)

    Requires either image_size=(W,H) or image_path.

    Speed: Typically tens/hundreds of proposals; robust across target_iou in [0,1] with tol≈0.05–0.1.
    """


    if not ((0.0 <= target_iou <= 1.0) or target_iou == -1.0):
        raise ValueError("target_iou must be in [0,1] or -1.")
    if tol < 0.0:
        raise ValueError("tol must be >= 0.")
    if max_tries < 1:
        raise ValueError("max_tries must be >= 1.")

    if image_size is None:
        if image_path is None:
            raise ValueError("Provide image_size=(W,H) for test mode or image_path.")
        if Image is None:
            raise RuntimeError("PIL is not available but image_path was provided.")
        with Image.open(image_path) as im:
            img_w, img_h = im.size
    else:
        img_w, img_h = image_size

    # Mirror zoom_in resizing-scale logic
    input_h, input_w = get_resized_image_scales(img_h, img_w, min_pixels, max_pixels)

    eff_pad = _effective_padding(padding, adaptive_padding_threshold, float(input_w), float(input_h))
    mode = _infer_mode(bbox_2d, bbox_type)

    # Reference final crop box (simulate zoom_in)
    ref_final_norm = _normalized_bbox_like_zoom_in(
        bbox_2d,
        padding=eff_pad,
        bbox_type=bbox_type,
        input_w=float(input_w),
        input_h=float(input_h),
    )
    ref_pix = _pixel_box_from_norm(ref_final_norm, img_w, img_h)

    other_bboxes_boxI = []
    if other_bboxes is not None:
        for other_bbox in other_bboxes:
            # Reference final crop box (simulate zoom_in)
            ref_final_norm_other = _normalized_bbox_like_zoom_in(
                other_bbox,
                padding=eff_pad,
                bbox_type=bbox_type,
                input_w=float(input_w),
                input_h=float(input_h),
            )
            ref_pix_other = _pixel_box_from_norm(ref_final_norm_other, img_w, img_h)
            other_bboxes_boxI.append(ref_pix_other)


    if target_iou == -1.0:
        return ref_pix.x1, ref_pix.y1, ref_pix.x2, ref_pix.y2

    rng = random.Random(seed)

    # Edge case requested earlier
    if target_iou == 0.0 and ref_final_norm == (0.0, 0.0, 1.0, 1.0):
        w = min(min_crop_size, img_w)
        h = min(min_crop_size, img_h)
        x1 = rng.randrange(0, max(1, img_w - w + 1))
        y1 = rng.randrange(0, max(1, img_h - h + 1))
        cand_pix = BoxI(x1, y1, x1 + w, y1 + h)
        cand_final_norm = (cand_pix.x1 / img_w, cand_pix.y1 / img_h, cand_pix.x2 / img_w, cand_pix.y2 / img_h)
        cand_input_norm = _invert_padding_best_effort(cand_final_norm, eff_pad)
        bbox_new = _emit_bbox_like_input(mode=mode, input_w=int(input_w), input_h=int(input_h), in_norm=cand_input_norm)
        if return_debug:
            dbg = {
                "mode": mode,
                "img_size": (img_w, img_h),
                "input_size": (int(input_w), int(input_h)),
                "eff_padding": eff_pad,
                "ref_pixel_box": ref_pix,
                "new_pixel_box_simulated": cand_pix,
                "target_iou": target_iou,
                "simulated_iou": _iou_i(ref_pix, cand_pix, metric=overlap_metric),
                "abs_error": abs(_iou_i(ref_pix, cand_pix, metric=overlap_metric) - target_iou),
                "note": "edge-case path: target_iou=0 and ref is full image",
            }
            return bbox_new, dbg
        return bbox_new

    best_bbox: Optional[BBoxOut] = None
    best_dbg: dict = {}
    best_err = float("inf")

    for attempt in range(max_tries):
        if target_iou == 0.0:
            cand_pix = _propose_box_free_floating_iou0(rng, ref_pix, img_w, img_h, min_crop_size)
        else:
            # Propose a FINAL pixel crop aimed at target_iou
            cand_pix = _propose_box_with_target_iou(rng, ref_pix, img_w, img_h, target_iou)
        if cand_pix is None:
            continue

        # Convert proposed final pixel crop -> final normalized
        cand_final_norm = (
            cand_pix.x1 / img_w,
            cand_pix.y1 / img_h,
            cand_pix.x2 / img_w,
            cand_pix.y2 / img_h,
        )

        # Invert padding to get an INPUT bbox (what we will output)
        cand_input_norm = _invert_padding_best_effort(cand_final_norm, eff_pad)
        bbox_new = _emit_bbox_like_input(mode=mode, input_w=int(input_w), input_h=int(input_h), in_norm=cand_input_norm)

        # Verify by simulating zoom_in on bbox_new
        cand_sim_final = _normalized_bbox_like_zoom_in(
            bbox_new,  # type: ignore[arg-type]
            padding=eff_pad,
            bbox_type=bbox_type,
            input_w=float(input_w),
            input_h=float(input_h),
        )
        cand_sim_pix = _pixel_box_from_norm(cand_sim_final, img_w, img_h)
        iou_sim = _iou_i(ref_pix, cand_sim_pix, metric=overlap_metric)
        err = abs(iou_sim - target_iou)

        is_digit_number_valid = True
        print(f"same digit number: {same_digit_number}")
        print(f"bbox_2d: {bbox_2d}, bbox_new: {bbox_new}")
        if same_digit_number:
            for idx in range(4):
                if str(bbox_new[idx]) != str(bbox_2d[idx]):
                    print(f"bbox is invalid")
                    is_digit_number_valid = False

        if err < best_err:

            if is_digit_number_valid:
                best_err = err
                best_bbox = bbox_new
                best_dbg = {
                    "attempt": attempt,
                    "mode": mode,
                    "img_size": (img_w, img_h),
                    "input_size": (int(input_w), int(input_h)),
                    "eff_padding": eff_pad,
                    "ref_pixel_box": ref_pix,
                    "proposal_pixel_box": cand_pix,
                    "new_pixel_box_simulated": cand_sim_pix,
                    "target_iou": target_iou,
                    "simulated_iou": iou_sim,
                    "abs_error": err,
                    "ref_final_norm": ref_final_norm,
                    "cand_final_norm_targeted": cand_final_norm,
                    "cand_final_norm_simulated": cand_sim_final,
                }


        #print(other_bboxes_boxI)
        too_close_to_previous = False
        if err <= tol:
            if other_bboxes is not None and len(other_bboxes) > 0:
                for other_bbox in other_bboxes_boxI:
                    print(f"iou to previous: {_iou_i(other_bbox, cand_sim_pix, metric=overlap_metric)}")
                    if _iou_i(other_bbox, cand_sim_pix, metric=overlap_metric) > other_bbox_threshold:
                        too_close_to_previous = True
            print(f"too close to previous: {too_close_to_previous}")
            if not too_close_to_previous and is_digit_number_valid:
                print(f"returning bbox")
                if return_debug:
                    return bbox_new, best_dbg
                return bbox_new
        print(f"next try")
    if best_bbox is None:
        raise RuntimeError("Failed to generate any candidate bbox (unexpected).")

    if return_debug:
        best_dbg["note"] = f"Did not reach tol={tol} in {max_tries} tries; returning best."
        return best_bbox, best_dbg
    return best_bbox