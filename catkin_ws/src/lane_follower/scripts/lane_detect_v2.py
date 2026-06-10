#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 Monolithic Lane Detect Node
Subscribes to sensor_msgs/Image (default: /dev/video0 or ~image_topic)
Publishes to lane_follower/LaneData
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, Final, Protocol
import math
import time

import cv2
import numpy as np

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
try:
    from lane_follower.msg import LaneData
except ImportError:
    rospy.logwarn("lane_follower.msg not found, using a dummy class for testing.")
    class LaneData:
        def __init__(self):
            self.offset = 0.0
            self.angle = 0.0


# ============================================================
# types.py
# ============================================================
"""Pipeline-wide dataclasses and enumerated status constants.

This is the only module every other pipeline module imports for inter-stage
data. Frozen dataclasses enforce that downstream stages cannot mutate upstream
output. Status / reject-reason strings live here as ``Final`` so no module
sprinkles raw string literals across the codebase.

See ``doc/design.md`` §2 (dataclass schemas) and §4 (status / reject enums).
"""




# --- MeasureResult.status ----------------------------------------------------
STATUS_OK: Final[str] = "ok"
STATUS_LEFT_ONLY_ESTIMATED: Final[str] = "left_only_estimated"
STATUS_RIGHT_ONLY_ESTIMATED: Final[str] = "right_only_estimated"
STATUS_NONE: Final[str] = "none"

MEASURE_STATUSES: Final[frozenset[str]] = frozenset(
    {STATUS_OK, STATUS_LEFT_ONLY_ESTIMATED, STATUS_RIGHT_ONLY_ESTIMATED, STATUS_NONE}
)

# --- SmoothResult.status (superset of MeasureResult.status) ------------------
STATUS_PREDICTED: Final[str] = "predicted"
SMOOTH_STATUSES: Final[frozenset[str]] = MEASURE_STATUSES | {STATUS_PREDICTED}

# --- LaneObservation.*_source ------------------------------------------------
SOURCE_DETECTED: Final[str] = "detected"
SOURCE_MEMORY: Final[str] = "memory"
SOURCE_MISSING: Final[str] = "missing"

OBSERVATION_SOURCES: Final[frozenset[str]] = frozenset(
    {SOURCE_DETECTED, SOURCE_MEMORY, SOURCE_MISSING}
)

@dataclass(frozen=True)
class FrameContext:
    """A single captured frame with monotonic identity."""

    bgr: np.ndarray
    timestamp: float
    frame_idx: int
    width: int
    height: int


@dataclass(frozen=True)
class PreprocessResult:
    """Per-frame intermediate masks. Values in ``binary`` / mask fields are in {0, 255}."""

    gray: np.ndarray
    binary: np.ndarray
    roi_mask: np.ndarray
    masked_binary: np.ndarray


@dataclass(frozen=True)
class RawDetection:
    """Area-filtered contours; not yet split into left / right."""

    contours: list[np.ndarray]
    areas: list[float]


@dataclass(frozen=True)
class LaneObservation:
    """Per-side classified points with tracker memory state."""

    left_points: np.ndarray | None
    right_points: np.ndarray | None
    left_source: str
    right_source: str
    frames_left_missing: int
    frames_right_missing: int


@dataclass(frozen=True)
class MeasureResult:
    """Per-frame offset/yaw measurement at the car anchor point."""

    offset_px: float | None
    yaw_deg: float | None
    status: str
    lane_width_used: float | None
    lane_width_raw: float | None
    anchor_y: int
    car_x: int


@dataclass(frozen=True)
class SmoothResult:
    """Time-filtered offset/yaw plus predict-mode bookkeeping."""

    offset_px: float | None
    yaw_deg: float | None
    status: str
    predicted: bool
    predict_streak: int


@dataclass(frozen=True)
class RenderInputs:
    """Single argument bundle handed to visualize.render()."""

    frame: FrameContext
    preprocess: PreprocessResult
    observation: LaneObservation
    measure: MeasureResult
    smooth: SmoothResult
    fps: float


# ============================================================
# config.py
# ============================================================
"""Single source of truth for tunable parameters.

The pipeline modules import ``Config`` and treat it as read-only. ``main.py``
constructs one instance after CLI parsing and passes it to every stage.
Test code should construct its own Config to avoid leaking state across tests.

Note: this module deliberately depends only on the standard library. It must
never import cv2, numpy, or any pipeline module.
"""




@dataclass(frozen=True)
class Config:
    """Frozen container of every tunable knob in the pipeline.

    All fields are immutable after creation; they are intended to be passed
    around as a single source of truth.  The inline comments below describe the
    practical effect of each knob so that users can tune the system without
    reading the source code.
    """

    # ------- Capture --------------------------------------------------------
    # Index of the USB camera (or video source) used by `capture.py`.
    CAMERA_INDEX: int = 0
    # Desired width/height of each captured frame.  Frames are resized to
    # this resolution before any processing; larger values give more detail
    # but increase CPU load.
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    TARGET_FPS: int = 30

    # ------- Preprocess -----------------------------------------------------
    
    # Kernel size for Gaussian blur; must be odd. Larger kernels smooth more
    # noise but can erase thin lane markings.
    BLUR_KERNEL: int = 5

    # Block size for adaptive thresholding (must be odd). Used only when
    # `USE_OTSU_FALLBACK` is False.
    ADAPTIVE_BLOCK_SIZE: int = 25
    
    # Constant subtracted from the mean in adaptive thresholding.
    ADAPTIVE_C: int = 10
    
    # Default to Otsu: the lanes on the TurtleBot floor are thick dark stripes on a
    # near-white background, which Otsu's single global threshold captures as filled
    # regions. Adaptive thresholding would treat them as edges only (block_size << lane
    # width). Flip to False to test adaptive on Pi if lighting becomes uneven.
    USE_OTSU_FALLBACK: bool = True
    # Kernel size for morphological opening (remove small speckles).
    MORPH_OPEN_KERNEL: int = 3
    # Kernel size for morphological closing (fill small gaps). Larger values
    # merge nearby fragments but can also wipe thin lane lines.
    MORPH_CLOSE_KERNEL: int = 9
    
    # If True, strictly crop the image using the ROI mask during preprocessing.
    # If False, the full image is processed to preserve lane continuity, and ROI is 
    # only used as a reference boundary for tracking logic.
    STRICT_ROI: bool = True
    # ROI polygon as (x_ratio, y_ratio) tuples in CCW order, bottom half full width
    # by default (matches legacy ref/file/lane_detection_HGL_v2.py).
    # Normalised ROI polygon (x_ratio, y_ratio) in counter‑clockwise order.
    # The default covers the lower half of the image, matching the original
    # reference implementation.  Adjusting these ratios lets you focus on a
    # different vertical region (e.g. a higher ROI for a forward‑looking camera).
    ROI_POLYGON_RATIOS: tuple[tuple[float, float], ...] = (
        (0.0, 1.0),
        (0.0, 0.45),
        (1.0, 0.45),
        (1.0, 1.0),
    )

    # ------- Detect ---------------------------------------------------------
    # Minimum contour area (in pixels) to be considered a lane line.
    # Smaller contours are usually noise; raising this value makes the system
    # more robust to speckles but may discard tiny lane fragments.
    MIN_CONTOUR_AREA: float = 500.0

    # ------- Tracker --------------------------------------------------------
    # Maximum number of consecutive frames a side may be missing before
    # the tracker forgets its previous position.
    MAX_MEMORY_FRAMES: int = 15
    # IoU threshold for matching a new contour with the previous frame's bounding
    # box.  Lower values make matching stricter; higher values tolerate more drift.
    OVERLAP_IOU_THRESHOLD: float = 0.15

    # ------- Fit ------------------------------------------------------------
    # Degree of polynomial used by the (optional) fitting stage. 2 = quadratic.
    POLY_DEGREE: int = 2
    # Minimum vertical span a contour must cover to be considered for fitting.
    MIN_Y_SPAN: int = 30  # px; reject if (y_max - y_min) < this
    # Maximum absolute quadratic coefficient; filters out overly curved fits.
    MAX_ABS_A: float = 0.01  # px / y**2; reject when |a| larger (over-curved)
    # Number of bins used when fitting the upper/lower lane boundaries.
    EDGE_Y_BINS: int = 20

    # ------- Measure --------------------------------------------------------
    # Ratio (0‑1) of image height where the anchor point is placed. 0.85 places
    # the measurement line near the bottom of the image.
    ANCHOR_RATIO: float = 0.8
    # 前瞻 anchor 的高度比例（較小 = 影像更上方 = 車前更遠處）。只用來算 offset_far /
    # angle_far，提供給轉彎後對正（T3_ALIGN）參考比較前方的車道方向。
    ALIGN_ANCHOR_RATIO: float = 0.5
    # Number of recent lane‑width samples kept for width‑estimation when only one
    # side is visible.
    LANE_WIDTH_HISTORY_LEN: int = 30

    # ------- Smooth ---------------------------------------------------------
    # Smoothing backend – either a Kalman filter or an exponential moving
    # average (EMA).  Kalman gives more realistic physical dynamics; EMA is cheap.
    SMOOTH_BACKEND: str = "kalman"  # "kalman" | "ema"
    # Kalman process noise (Q) and measurement noise (R) for the offset.
    KALMAN_Q_OFFSET: float = 0.5
    KALMAN_R_OFFSET: float = 4.0
    # Same for yaw (when yaw calculation is enabled).
    KALMAN_Q_YAW: float = 0.1
    KALMAN_R_YAW: float = 1.0
    # EMA blending factor (alpha). Larger values follow recent measurements more
    # closely.
    EMA_ALPHA: float = 0.4
    # How many future frames can be predicted when no lane is detected.
    MAX_PREDICT_FRAMES: int = 15

    # ------- Visualize ------------------------------------------------------
    # BGR (OpenCV order) colour palette used by the visualiser.
    COLOR_LEFT_BGR: tuple[int, int, int] = (0, 0, 255)   # Red for left lane
    COLOR_RIGHT_BGR: tuple[int, int, int] = (255, 0, 0)  # Blue for right lane
    COLOR_CENTER_BGR: tuple[int, int, int] = (0, 255, 255)  # Cyan for lane centre
    COLOR_ROI_BGR: tuple[int, int, int] = (0, 200, 0)   # Green ROI outline
    COLOR_ANCHOR_BGR: tuple[int, int, int] = (255, 255, 255)  # White anchor point
    COLOR_TEXT_OK_BGR: tuple[int, int, int] = (255, 255, 255)  # White status text
    COLOR_TEXT_PREDICTED_BGR: tuple[int, int, int] = (0, 255, 255)  # Cyan when predicted
    COLOR_TEXT_NONE_BGR: tuple[int, int, int] = (160, 160, 160)  # Gray when no lane
    FONT_SCALE: float = 0.6
    LANE_FILL_ALPHA: float = 0.45  # Blend factor for filled lane regions

    # Internal — used only by tests / future hooks; populated nowhere yet.
    # Reserved for future extensions; left empty on purpose.
    _reserved: tuple[str, ...] = field(default_factory=tuple)

    # -------------------------------------------------------
    # NEW FEATURE: yaw calculation mode selection
    # -------------------------------------------------------
    # When False (default) the system uses the lightweight
    # contour‑averaging method (scheme A) and reports yaw = 0.0.
    # When True the pipeline falls back to scheme B which fits a
    # line (or quadratic curve) to each side and computes yaw from the
    # slope.  This is more computationally expensive but provides a true
    # heading angle.  Users can enable it via `--enable-yaw-b` CLI flag or
    # by overriding the config in code: `Config(ENABLE_YAW_B=True)`.
    ENABLE_YAW_B: bool = True


# ============================================================
# preprocess.py
# ============================================================
"""Per-frame image cleaning: gray, blur, threshold, morph, ROI mask.

Pure function ``process(frame, cfg) -> PreprocessResult``. No mutable state,
no cv2.imshow / VideoCapture. Threshold method is adaptive by default; flip
``cfg.USE_OTSU_FALLBACK`` to switch to Otsu (per spec §10 / design §3.3).

Lane markings are dark on the light floor (see ``ref/test/test1.png``), so we
use ``THRESH_BINARY_INV`` — lane pixels end up as 255 in the binary mask.
"""





def _build_roi_mask(height: int, width: int, cfg: Config) -> np.ndarray:
    """Rasterize the ROI polygon from ratio-coordinates into a uint8 mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    poly = np.array(
        [(int(rx * width), int(ry * height)) for rx, ry in cfg.ROI_POLYGON_RATIOS],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [poly], 255)
    return mask


def process(frame: FrameContext, cfg: Config) -> PreprocessResult:
    """Run gray → blur → threshold → morph → ROI on a single frame."""
    gray = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)

    k = cfg.BLUR_KERNEL | 1  # ensure odd
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    if cfg.USE_OTSU_FALLBACK:
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        block = cfg.ADAPTIVE_BLOCK_SIZE | 1
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            cfg.ADAPTIVE_C,
        )

    open_k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (cfg.MORPH_OPEN_KERNEL, cfg.MORPH_OPEN_KERNEL)
    )
    close_k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (cfg.MORPH_CLOSE_KERNEL, cfg.MORPH_CLOSE_KERNEL)
    )
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_k)

    roi_mask = _build_roi_mask(frame.height, frame.width, cfg)
    
    if getattr(cfg, "STRICT_ROI", False):
        masked_binary = cv2.bitwise_and(cleaned, roi_mask)
    else:
        masked_binary = cleaned

    return PreprocessResult(
        gray=blurred,
        binary=cleaned,
        roi_mask=roi_mask,
        masked_binary=masked_binary,
    )


# ============================================================
# detect.py
# ============================================================
"""Contour extraction from the ROI-masked binary image.

Pure function ``find_lane_contours(pre, cfg) -> RawDetection``. Filters by
``cfg.MIN_CONTOUR_AREA``; does NOT split left vs right (that's the tracker's
job per design.md §3.4). Returns parallel lists ``contours`` and ``areas``.
Empty input ⇒ empty lists, never None.
"""





def min_contour_distance(c1, c2):
    step1 = max(1, len(c1) // 50)
    step2 = max(1, len(c2) // 50)
    pts1 = c1[::step1].reshape(-1, 2).astype(np.float32)
    pts2 = c2[::step2].reshape(-1, 2).astype(np.float32)
    min_dist = float('inf')
    for pt in pts1:
        dists = np.sqrt(np.sum((pts2 - pt) ** 2, axis=1))
        d = float(np.min(dists))
        if d < min_dist:
            min_dist = d
        if min_dist < 1:
            return min_dist
    return min_dist


def find_intersection(line1, line2):
    vx1, vy1, x1, y1 = line1.flatten()
    vx2, vy2, x2, y2 = line2.flatten()
    det = vx1 * vy2 - vy1 * vx2
    if abs(det) < 1e-6:
        return None
    t1 = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det
    return np.array([x1 + t1 * vx1, y1 + t1 * vy1])


def merge_contours_via_mask(contours, h, w, close_size=7):
    if not contours:
        return None
    if len(contours) == 1:
        return contours[0]
    mask = np.zeros((h, w), dtype=np.uint8)
    for c in contours:
        cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found:
        return contours[0]
    return max(found, key=cv2.contourArea)


def resolve_floating_v4(main_contours, floating_contours, h, w, merge_dist=10, ix_thresh=30, ix_to_segment_max=200):
    buckets = [[c] for c in main_contours]
    for float_c in floating_contours:
        bucket_idx = -1
        best_dist = float('inf')
        best_a_idx = -1
        for i, main_c in enumerate(main_contours):
            d = min_contour_distance(float_c, main_c)
            if d < best_dist:
                best_dist = d
                best_a_idx = i
        if best_dist < merge_dist and best_a_idx != -1:
            bucket_idx = best_a_idx

        if bucket_idx == -1 and len(float_c) >= 5:
            float_line = cv2.fitLine(float_c, cv2.DIST_L2, 0, 0.01, 0.01)
            best_total = float('inf')
            best_b_idx = -1
            for i, main_c in enumerate(main_contours):
                if len(main_c) < 5:
                    continue
                main_line = cv2.fitLine(main_c, cv2.DIST_L2, 0, 0.01, 0.01)
                ix_pt = find_intersection(float_line, main_line)
                if ix_pt is None:
                    continue
                pts_float = float_c.reshape(-1, 2).astype(np.float64)
                pts_main = main_c.reshape(-1, 2).astype(np.float64)
                d_float = float(np.min(np.sqrt(np.sum((pts_float - ix_pt) ** 2, axis=1))))
                d_main = float(np.min(np.sqrt(np.sum((pts_main - ix_pt) ** 2, axis=1))))
                if d_float < ix_thresh and d_main < ix_thresh:
                    if d_float < ix_to_segment_max and d_main < ix_to_segment_max:
                        total = d_float + d_main
                        if total < best_total:
                            best_total = total
                            best_b_idx = i
            if best_b_idx != -1:
                bucket_idx = best_b_idx

        if bucket_idx != -1:
            buckets[bucket_idx].append(float_c)

    merged_main_contours = []
    for bucket in buckets:
        if len(bucket) == 1:
            merged_main_contours.append(bucket[0])
        else:
            merged = merge_contours_via_mask(bucket, h, w, close_size=11)
            if merged is not None:
                merged_main_contours.append(merged)
    return merged_main_contours


def clip_contour_to_roi(contour, roi_mask, h, w, min_area=500):
    cmask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(cmask, [contour], -1, 255, cv2.FILLED)
    cmask_in_roi = cv2.bitwise_and(cmask, roi_mask)
    sub_contours, _ = cv2.findContours(cmask_in_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [sc for sc in sub_contours if cv2.contourArea(sc) >= min_area]


def find_lane_contours(pre: PreprocessResult, cfg: Config) -> RawDetection:
    h, w = pre.masked_binary.shape[:2]
    found, _ = cv2.findContours(
        pre.masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    valid_full = [c for c in found if cv2.contourArea(c) > cfg.MIN_CONTOUR_AREA]

    roi_y_top = int(h * 0.45)
    main_contours = []
    floating_contours = []
    for c in valid_full:
        if np.any(c[:, 0, 1] >= roi_y_top):
            main_contours.append(c)
        else:
            floating_contours.append(c)

    resolved_main = resolve_floating_v4(main_contours, floating_contours, h, w)

    max_area_thresh = 40000
    valid_contours = []
    for c in resolved_main:
        if cv2.contourArea(c) > max_area_thresh:
            sub = clip_contour_to_roi(c, pre.roi_mask, h, w, min_area=cfg.MIN_CONTOUR_AREA)
            valid_contours.extend(sub)
        else:
            valid_contours.append(c)

    areas = [float(cv2.contourArea(c)) for c in valid_contours]
    return RawDetection(contours=valid_contours, areas=areas)


# ============================================================
# tracker.py
# ============================================================
"""Per-side lane tracker with short-term memory.

State machine (each side independent):
    Missing → Detected → Memory → Detected | Missing

Matching strategy:
  1. If a previous bbox exists (and memory hasn't expired), pick the current
     contour with highest bbox-IoU vs that previous bbox, if above
     ``cfg.OVERLAP_IOU_THRESHOLD``.
  2. Otherwise bisect by centroid x vs frame midline (left = cx < mid).
  3. If a side is unmatched this frame, return the *previous frame's* points
     verbatim with ``source = "memory"``, up to ``cfg.MAX_MEMORY_FRAMES``.

The module deliberately does NOT import cv2 (design.md §3.5). Bounding-box
IoU is a numpy-only proxy for the legacy rasterized-mask overlap; it's a
slightly looser metric but adequate at the default 0.15 threshold.
"""




from typing import Tuple
# Internal helper type: (points_Nx2, bbox(x_min, y_min, x_max, y_max), centroid_x)
_Cand = Tuple[np.ndarray, Tuple[int, int, int, int], float]


def _contour_to_points(contour: np.ndarray) -> np.ndarray:
    """Convert OpenCV contour shape (N, 1, 2) -> (N, 2) int32."""
    return contour.reshape(-1, 2).astype(np.int32, copy=False)


def _bbox_of(points: np.ndarray) -> tuple[int, int, int, int]:
    x_min = int(points[:, 0].min())
    y_min = int(points[:, 1].min())
    x_max = int(points[:, 0].max())
    y_max = int(points[:, 1].max())
    return x_min, y_min, x_max, y_max


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class LaneTracker:
    """Holds previous-frame bboxes/points per side and emits ``LaneObservation``."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._prev_left_points: np.ndarray | None = None
        self._prev_right_points: np.ndarray | None = None
        self._prev_left_bbox: tuple[int, int, int, int] | None = None
        self._prev_right_bbox: tuple[int, int, int, int] | None = None
        self._frames_left_missing: int = 0
        self._frames_right_missing: int = 0

    def reset(self) -> None:
        """Drop all memory; next ``update`` starts cold."""
        self._prev_left_points = None
        self._prev_right_points = None
        self._prev_left_bbox = None
        self._prev_right_bbox = None
        self._frames_left_missing = 0
        self._frames_right_missing = 0

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------
    def update(self, detection: RawDetection, frame_width: int) -> LaneObservation:
        mid_x = frame_width / 2.0

        candidates: list[_Cand] = []
        for c in detection.contours:
            pts = _contour_to_points(c)
            if pts.size == 0:
                continue
            bbox = _bbox_of(pts)
            cx = float(pts[:, 0].mean())
            candidates.append((pts, bbox, cx))

        used: set[int] = set()
        left_idx = self._match_by_memory(
            candidates,
            used,
            self._prev_left_bbox,
            self._frames_left_missing,
        )
        right_idx = self._match_by_memory(
            candidates,
            used,
            self._prev_right_bbox,
            self._frames_right_missing,
        )

        if left_idx is None:
            left_idx = self._match_by_bisection(candidates, used, mid_x, side="left")
        if right_idx is None:
            right_idx = self._match_by_bisection(candidates, used, mid_x, side="right")

        # Ensure spatial sanity if both matched.
        left_idx, right_idx = self._enforce_left_right_order(candidates, left_idx, right_idx)

        left_points, left_source, frames_left_missing = self._resolve_side(
            candidates,
            left_idx,
            self._prev_left_points,
            self._frames_left_missing,
        )
        right_points, right_source, frames_right_missing = self._resolve_side(
            candidates,
            right_idx,
            self._prev_right_points,
            self._frames_right_missing,
        )

        # Commit state.
        if left_idx is not None:
            self._prev_left_points = candidates[left_idx][0]
            self._prev_left_bbox = candidates[left_idx][1]
        if right_idx is not None:
            self._prev_right_points = candidates[right_idx][0]
            self._prev_right_bbox = candidates[right_idx][1]

        if left_source == SOURCE_MISSING:
            self._prev_left_points = None
            self._prev_left_bbox = None
        if right_source == SOURCE_MISSING:
            self._prev_right_points = None
            self._prev_right_bbox = None

        self._frames_left_missing = frames_left_missing
        self._frames_right_missing = frames_right_missing

        return LaneObservation(
            left_points=left_points,
            right_points=right_points,
            left_source=left_source,
            right_source=right_source,
            frames_left_missing=frames_left_missing,
            frames_right_missing=frames_right_missing,
        )

    # ------------------------------------------------------------------
    # matching helpers
    # ------------------------------------------------------------------
    def _match_by_memory(
        self,
        candidates: list[_Cand],
        used: set[int],
        prev_bbox: tuple[int, int, int, int] | None,
        frames_missing: int,
    ) -> int | None:
        if prev_bbox is None or frames_missing >= self._cfg.MAX_MEMORY_FRAMES:
            return None
        best_iou = self._cfg.OVERLAP_IOU_THRESHOLD
        best_idx: int | None = None
        for i, (_, bbox, _) in enumerate(candidates):
            if i in used:
                continue
            iou = _bbox_iou(prev_bbox, bbox)
            if iou >= best_iou:
                best_iou = iou
                best_idx = i
        if best_idx is not None:
            used.add(best_idx)
        return best_idx

    def _match_by_bisection(
        self,
        candidates: list[_Cand],
        used: set[int],
        mid_x: float,
        side: str,
    ) -> int | None:
        # Prefer the outermost contour on each side (leftmost cx for "left", rightmost
        # cx for "right"). Picking nearest-to-midline tends to grab perspective-skewed
        # mid-frame fragments and miss the real lane that extends to the image border.
        best_idx: int | None = None
        for i, (_, _, cx) in enumerate(candidates):
            if i in used:
                continue
            if side == "left" and cx >= mid_x:
                continue
            if side == "right" and cx < mid_x:
                continue
            if best_idx is None:
                best_idx = i
                continue
            cur_cx = candidates[best_idx][2]
            if side == "left" and cx < cur_cx:
                best_idx = i
            elif side == "right" and cx > cur_cx:
                best_idx = i
        if best_idx is not None:
            used.add(best_idx)
        return best_idx

    def _enforce_left_right_order(
        self,
        candidates: list[_Cand],
        left_idx: int | None,
        right_idx: int | None,
    ) -> tuple[int | None, int | None]:
        """If left's centroid is right of right's centroid, drop the weaker side."""
        if left_idx is None or right_idx is None:
            return left_idx, right_idx
        left_cx = candidates[left_idx][2]
        right_cx = candidates[right_idx][2]
        if left_cx <= right_cx:
            return left_idx, right_idx

        # Order violated. Keep whichever has the stronger memory IoU.
        left_iou = (
            _bbox_iou(self._prev_left_bbox, candidates[left_idx][1])
            if self._prev_left_bbox is not None
            else 0.0
        )
        right_iou = (
            _bbox_iou(self._prev_right_bbox, candidates[right_idx][1])
            if self._prev_right_bbox is not None
            else 0.0
        )
        if left_iou >= right_iou:
            return left_idx, None
        return None, right_idx

    def _resolve_side(
        self,
        candidates: list[_Cand],
        matched_idx: int | None,
        prev_points: np.ndarray | None,
        prev_frames_missing: int,
    ) -> tuple[np.ndarray | None, str, int]:
        if matched_idx is not None:
            return candidates[matched_idx][0], SOURCE_DETECTED, 0

        # Not matched this frame — try memory.
        next_missing = prev_frames_missing + 1
        if prev_points is not None and next_missing <= self._cfg.MAX_MEMORY_FRAMES:
            return prev_points, SOURCE_MEMORY, next_missing
        return None, SOURCE_MISSING, next_missing


# ============================================================
# measure.py
# ============================================================
"""Offset + yaw computation at the car anchor row.

Pure stateless function. Cross-frame lane-width history is held by ``main``
(see design.md §3.7) and passed in as a list of floats (oldest first).

Note on signature: design.md §3.7 specified ``(fit, cfg, history)``, but
``MeasureResult.car_x`` / ``anchor_y`` need the frame dimensions, which aren't
in ``FitResult``. We extend the signature with ``frame_width`` / ``frame_height``
so measure works correctly even when the runtime frame size differs from
``cfg.FRAME_WIDTH`` (e.g. ``--image`` mode loads a PNG of arbitrary size).

Sign convention (spec §9):
  * ``offset_px = car_x - x_center(car_y)``  →  positive = car right of center.
  * ``yaw_deg  = degrees(arctan(slope))``    →  positive = car heading right.
"""





def _anchor_xy(
    cfg: Config,
    frame_width: int,
    frame_height: int,
    anchor_ratio: float | None = None,
) -> tuple[int, int]:
    car_x = frame_width // 2
    ratio = cfg.ANCHOR_RATIO if anchor_ratio is None else anchor_ratio
    anchor_y = int(frame_height * ratio)
    return car_x, anchor_y

def _mean_width(history: list[float]) -> float | None:
    if not history:
        return None
    return float(np.mean(history))

def _none_result(anchor_y: int, car_x: int) -> MeasureResult:
    """The 'no usable lane this frame' measurement (offset/yaw unknown)."""
    return MeasureResult(
        offset_px=None,
        yaw_deg=None,
        status=STATUS_NONE,
        lane_width_used=None,
        lane_width_raw=None,
        anchor_y=anchor_y,
        car_x=car_x,
    )

def measure_at_anchor(
    obs: LaneObservation,
    cfg: Config,
    lane_width_history: list[float],
    frame_width: int,
    frame_height: int,
    anchor_ratio: float | None = None,
) -> MeasureResult:
    """Compute offset & yaw at the anchor row directly from the raw point contours.

    ``anchor_ratio`` overrides ``cfg.ANCHOR_RATIO`` for this call only (used to
    measure a second, look-ahead anchor for post-turn alignment).
    """
    car_x, anchor_y = _anchor_xy(cfg, frame_width, frame_height, anchor_ratio)

    def avg_x_in_band(points: np.ndarray | None, band_half: int = 15) -> float | None:
        if points is None or len(points) == 0:
            return None
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cnt = points.reshape(-1, 1, 2)
        cv2.fillPoly(mask, [cnt], 255)
        y0 = max(0, anchor_y - band_half)
        y1 = min(frame_height, anchor_y + band_half + 1)
        band = mask[y0:y1, :]
        _, xs = np.where(band > 0)
        if len(xs) == 0:
            return None
        return float(np.mean(xs))

    left_x = avg_x_in_band(obs.left_points)
    right_x = avg_x_in_band(obs.right_points)
    
    def calc_yaw(points: np.ndarray | None) -> float | None:
        if points is None or len(points) < 5:
            return None
        pts = points.reshape(-1, 2)
        if getattr(cfg, "ENABLE_YAW_B", False):
            # 方案 B：二次曲線 (np.polyfit)，取微分算 yaw_deg
            xs = pts[:, 0]
            ys = pts[:, 1]
            if np.max(ys) - np.min(ys) < 10:
                return None
            poly = np.polyfit(ys, xs, 2)
            slope = 2 * poly[0] * anchor_y + poly[1]
            return math.degrees(math.atan(slope))
        else:
            # 方案 A：cv2.fitLine 擬合直線
            line = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(line[0][0]), float(line[1][0])
            # 使用 math.atan(vx/vy) 而不是 atan2(vy, vx) 來維持車頭朝正前方 = 0 度的系統定義
            if abs(vy) < 1e-6:
                return 0.0
            return math.degrees(math.atan(vx / vy))

    left_yaw = calc_yaw(obs.left_points)
    right_yaw = calc_yaw(obs.right_points)

    if left_yaw is not None and right_yaw is not None:
        yaw_deg = 0.5 * (left_yaw + right_yaw)
    elif left_yaw is not None:
        yaw_deg = left_yaw
    elif right_yaw is not None:
        yaw_deg = right_yaw
    else:
        yaw_deg = 0.0

    if left_x is not None and right_x is not None:
        lane_width_raw = right_x - left_x
        center_x = 0.5 * (left_x + right_x)
        offset = float(car_x) - center_x
        return MeasureResult(
            offset_px=offset,
            yaw_deg=yaw_deg,
            status=STATUS_OK,
            lane_width_used=None,
            lane_width_raw=lane_width_raw,
            anchor_y=anchor_y,
            car_x=car_x,
        )

    width = _mean_width(lane_width_history)

    if left_x is not None and right_x is None:
        if width is None:
            return _none_result(anchor_y, car_x)
        est_right = left_x + width
        center_x = 0.5 * (left_x + est_right)
        offset = float(car_x) - center_x
        return MeasureResult(
            offset_px=offset,
            yaw_deg=yaw_deg,
            status=STATUS_LEFT_ONLY_ESTIMATED,
            lane_width_used=width,
            lane_width_raw=None,
            anchor_y=anchor_y,
            car_x=car_x,
        )

    if right_x is not None and left_x is None:
        if width is None:
            return _none_result(anchor_y, car_x)
        est_left = right_x - width
        center_x = 0.5 * (est_left + right_x)
        offset = float(car_x) - center_x
        return MeasureResult(
            offset_px=offset,
            yaw_deg=yaw_deg,
            status=STATUS_RIGHT_ONLY_ESTIMATED,
            lane_width_used=width,
            lane_width_raw=None,
            anchor_y=anchor_y,
            car_x=car_x,
        )

    return _none_result(anchor_y, car_x)


# ============================================================
# smooth.py
# ============================================================
"""Time-domain smoother with selectable Kalman / EMA backend.

State machine (per design.md §4.4):

    Uninitialized → Tracking → Predicting → Uninitialized

  * In ``Uninitialized``: every frame emits ``status="none"`` regardless of the
    upstream measure status. On the *first* non-none measurement we initialize
    the filter and stay Uninitialized for one frame (the design's 1-frame
    startup latency is intentional — see §4.4 note).
  * In ``Tracking``: on a non-none upstream measurement we update the filter
    and emit ``status="ok"`` plus the smoothed values. The smoother's status
    deliberately collapses ``"left_only_estimated"`` / ``"right_only_estimated"``
    to ``"ok"`` because both are valid observations from the filter's POV.
  * In ``Predicting``: upstream is ``"none"``. We predict only (no update),
    emit ``status="predicted"`` with the predicted values, bump the streak. If
    ``predict_streak >= MAX_PREDICT_FRAMES``, we reset to Uninitialized and
    emit ``status="none"`` — the safety bound in spec §8.4 (avoid stale
    extrapolation "driving into a wall").

Both backends produce identical ``SmoothResult`` shape so they can be swapped
by ``cfg.SMOOTH_BACKEND`` without touching the rest of the pipeline.
"""






class _ScalarFilter(Protocol):
    """Minimal interface each backend exposes for one scalar channel."""

    def init(self, value: float) -> None: ...
    def predict(self) -> float: ...
    def update(self, measurement: float) -> float: ...
    def reset(self) -> None: ...


class _KalmanScalar:
    """1D constant-position Kalman filter, using filterpy."""

    def __init__(self, q: float, r: float) -> None:
        from filterpy.kalman import KalmanFilter

        self._cls = KalmanFilter
        self._q = q
        self._r = r
        self._kf: Any = self._build()

    def _build(self) -> Any:
        kf = self._cls(dim_x=1, dim_z=1)
        kf.F = np.array([[1.0]])
        kf.H = np.array([[1.0]])
        kf.Q = np.array([[self._q]])
        kf.R = np.array([[self._r]])
        kf.P = np.array([[10.0]])
        kf.x = np.array([[0.0]])
        return kf

    def init(self, value: float) -> None:
        self._kf.x = np.array([[value]])
        self._kf.P = np.array([[1.0]])

    def predict(self) -> float:
        self._kf.predict()
        return float(self._kf.x[0, 0])

    def update(self, measurement: float) -> float:
        self._kf.predict()
        self._kf.update(np.array([[measurement]]))
        return float(self._kf.x[0, 0])

    def reset(self) -> None:
        self._kf = self._build()


class _EMAScalar:
    """Exponential moving average backend; predict() echoes the last value."""

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._x: float | None = None

    def init(self, value: float) -> None:
        self._x = value

    def predict(self) -> float:
        # In EMA, "predict-only" just echoes the last smoothed value.
        return float(self._x) if self._x is not None else 0.0

    def update(self, measurement: float) -> float:
        if self._x is None:
            self._x = measurement
        else:
            self._x = self._alpha * measurement + (1.0 - self._alpha) * self._x
        return self._x

    def reset(self) -> None:
        self._x = None


def _make_offset_filter(cfg: Config) -> _ScalarFilter:
    if cfg.SMOOTH_BACKEND == "ema":
        return _EMAScalar(cfg.EMA_ALPHA)
    return _KalmanScalar(cfg.KALMAN_Q_OFFSET, cfg.KALMAN_R_OFFSET)


def _make_yaw_filter(cfg: Config) -> _ScalarFilter:
    if cfg.SMOOTH_BACKEND == "ema":
        return _EMAScalar(cfg.EMA_ALPHA)
    return _KalmanScalar(cfg.KALMAN_Q_YAW, cfg.KALMAN_R_YAW)


class LaneSmoother:
    """Manages the per-channel filters and the Uninit/Tracking/Predicting state."""

    _STATE_UNINIT = "uninit"
    _STATE_TRACKING = "tracking"

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._offset_filter = _make_offset_filter(cfg)
        self._yaw_filter = _make_yaw_filter(cfg)
        self._state: str = self._STATE_UNINIT
        self._predict_streak: int = 0

    def reset(self) -> None:
        self._offset_filter.reset()
        self._yaw_filter.reset()
        self._state = self._STATE_UNINIT
        self._predict_streak = 0

    def update(self, meas: MeasureResult) -> SmoothResult:
        has_obs = (
            meas.status != STATUS_NONE
            and meas.offset_px is not None
            and meas.yaw_deg is not None
        )

        if self._state == self._STATE_UNINIT:
            if has_obs:
                # Initialize filters but still emit "none" this frame (design §4.4).
                assert meas.offset_px is not None and meas.yaw_deg is not None
                self._offset_filter.init(meas.offset_px)
                self._yaw_filter.init(meas.yaw_deg)
                self._state = self._STATE_TRACKING
                self._predict_streak = 0
            return SmoothResult(
                offset_px=None,
                yaw_deg=None,
                status=STATUS_NONE,
                predicted=False,
                predict_streak=0,
            )

        # Tracking state.
        if has_obs:
            assert meas.offset_px is not None and meas.yaw_deg is not None
            off = self._offset_filter.update(meas.offset_px)
            yaw = self._yaw_filter.update(meas.yaw_deg)
            self._predict_streak = 0
            return SmoothResult(
                offset_px=off,
                yaw_deg=yaw,
                status=STATUS_OK,
                predicted=False,
                predict_streak=0,
            )

        # Upstream is "none" → predict.
        self._predict_streak += 1
        if self._predict_streak > self._cfg.MAX_PREDICT_FRAMES:
            # Safety bound exceeded — reset and bail.
            self.reset()
            return SmoothResult(
                offset_px=None,
                yaw_deg=None,
                status=STATUS_NONE,
                predicted=False,
                predict_streak=0,
            )

        off = self._offset_filter.predict()
        yaw = self._yaw_filter.predict()
        return SmoothResult(
            offset_px=off,
            yaw_deg=yaw,
            status=STATUS_PREDICTED,
            predicted=True,
            predict_streak=self._predict_streak,
        )


# ============================================================
# visualize.py
# ============================================================
"""Annotated-frame renderer.

Pure function ``render(inputs, cfg) -> np.ndarray``. Copies the input BGR
frame before drawing — never mutates upstream pixels (design.md §3.9).
Does NOT call ``cv2.imshow`` / ``cv2.waitKey``; that's main's responsibility.

Drawing order (back-to-front so later strokes win on overlap):
  1. ROI polygon outline (green)
  2. Filled lane regions (red left, blue right) from the tracker's per-side
     ``observation`` points
  3. Anchor point + offset connector line + lane-center marker
  4. Text overlay (offset / yaw / status / fps; "no lane detected" centered
     when status == "none")
"""





def _draw_roi(canvas: np.ndarray, cfg: Config) -> None:
    h, w = canvas.shape[:2]
    poly = np.array(
        [(int(rx * w), int(ry * h)) for rx, ry in cfg.ROI_POLYGON_RATIOS],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [poly], isClosed=True, color=cfg.COLOR_ROI_BGR, thickness=2)


def _draw_lane_side(
    canvas: np.ndarray,
    obs: LaneObservation,
    side: str,
    cfg: Config,
) -> None:
    points = obs.left_points if side == "left" else obs.right_points
    color = cfg.COLOR_LEFT_BGR if side == "left" else cfg.COLOR_RIGHT_BGR

    if points is not None and len(points) > 0:
        cnt = points.reshape(-1, 1, 2)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [cnt], color)
        cv2.addWeighted(overlay, cfg.LANE_FILL_ALPHA, canvas, 1.0 - cfg.LANE_FILL_ALPHA, 0, canvas)
        cv2.polylines(canvas, [cnt], isClosed=True, color=color, thickness=2)


def _draw_anchor_and_center(
    canvas: np.ndarray,
    inputs: RenderInputs,
    cfg: Config,
) -> None:
    car_x = inputs.measure.car_x
    car_y = inputs.measure.anchor_y
    cv2.circle(canvas, (car_x, car_y), 6, cfg.COLOR_ANCHOR_BGR, -1)
    cv2.circle(canvas, (car_x, car_y), 6, (0, 0, 0), 1)

    # Lane center marker from the smoothed offset (so the cursor doesn't jitter).
    off = inputs.smooth.offset_px
    if off is not None:
        center_x = int(round(car_x - off))
        cv2.line(canvas, (center_x, car_y - 10), (center_x, car_y + 10), cfg.COLOR_CENTER_BGR, 2)
        if abs(center_x - car_x) > 2:
            cv2.arrowedLine(
                canvas,
                (car_x, car_y),
                (center_x, car_y),
                cfg.COLOR_CENTER_BGR,
                2,
                tipLength=0.25,
            )


def _draw_text(canvas: np.ndarray, inputs: RenderInputs, cfg: Config) -> None:
    h, w = canvas.shape[:2]
    s = inputs.smooth

    if s.status == STATUS_PREDICTED:
        text_color = cfg.COLOR_TEXT_PREDICTED_BGR
    elif s.status == STATUS_NONE:
        text_color = cfg.COLOR_TEXT_NONE_BGR
    else:
        text_color = cfg.COLOR_TEXT_OK_BGR

    if s.offset_px is None or s.yaw_deg is None:
        info = f"offset=N/A  yaw=N/A  status={s.status}  fps={inputs.fps:5.1f}"
    else:
        info = (
            f"offset={s.offset_px:+6.1f}px  yaw={s.yaw_deg:+5.2f}deg  "
            f"status={s.status}  fps={inputs.fps:5.1f}"
        )
    cv2.rectangle(canvas, (5, h - 35), (5 + 460, h - 10), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        info,
        (10, h - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        cfg.FONT_SCALE,
        text_color,
        1,
        cv2.LINE_AA,
    )

    if s.status == STATUS_NONE:
        msg = "no lane detected"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(
            canvas,
            msg,
            ((w - tw) // 2, h // 2 + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def render(inputs: RenderInputs, cfg: Config) -> np.ndarray:
    """Build an annotated copy of ``inputs.frame.bgr``."""
    canvas = inputs.frame.bgr.copy()
    _draw_roi(canvas, cfg)
    _draw_lane_side(canvas, inputs.observation, "left", cfg)
    _draw_lane_side(canvas, inputs.observation, "right", cfg)
    _draw_anchor_and_center(canvas, inputs, cfg)
    _draw_text(canvas, inputs, cfg)
    return canvas


# ============================================================
# ROS Node & Main Loop
# ============================================================

class LaneDetectNode:
    def __init__(self):
        rospy.init_node('lane_detect_node', anonymous=True)
        
        # Load ROS Params
        self.image_topic = rospy.get_param('~image_topic', '/dev/video0')
        self.pub_topic = rospy.get_param('~pub_topic', 'lane_detect')
        self.debug_topic = rospy.get_param('~debug_topic', 'lane_detect/image_out')
        self.binary_topic = rospy.get_param('~binary_topic', 'lane_detect/image_binary')
        self.show_window = rospy.get_param('~show_window', False)

        # Initialize Config
        default_cfg = Config()
        cfg_kwargs = {}
        
        # Override algorithm configurations from ROS Params
        cfg_kwargs['STRICT_ROI'] = rospy.get_param('~strict_roi', default_cfg.STRICT_ROI)
        cfg_kwargs['MIN_CONTOUR_AREA'] = float(rospy.get_param('~min_contour_area', default_cfg.MIN_CONTOUR_AREA))
        cfg_kwargs['ANCHOR_RATIO'] = float(rospy.get_param('~anchor_ratio', default_cfg.ANCHOR_RATIO))
        cfg_kwargs['ALIGN_ANCHOR_RATIO'] = float(rospy.get_param('~align_anchor_ratio', default_cfg.ALIGN_ANCHOR_RATIO))
        cfg_kwargs['LANE_WIDTH_HISTORY_LEN'] = int(rospy.get_param('~lane_width_history_len', default_cfg.LANE_WIDTH_HISTORY_LEN))
        cfg_kwargs['ENABLE_YAW_B'] = rospy.get_param('~enable_yaw_b', default_cfg.ENABLE_YAW_B)
        cfg_kwargs['MAX_PREDICT_FRAMES'] = int(rospy.get_param('~max_predict_frames', default_cfg.MAX_PREDICT_FRAMES))

        # Binarization & Morphological Parameters
        cfg_kwargs['USE_OTSU_FALLBACK'] = rospy.get_param('~use_otsu_fallback', default_cfg.USE_OTSU_FALLBACK)
        cfg_kwargs['ADAPTIVE_BLOCK_SIZE'] = int(rospy.get_param('~adaptive_block_size', default_cfg.ADAPTIVE_BLOCK_SIZE))
        cfg_kwargs['ADAPTIVE_C'] = int(rospy.get_param('~adaptive_c', default_cfg.ADAPTIVE_C))
        cfg_kwargs['BLUR_KERNEL'] = int(rospy.get_param('~blur_kernel', default_cfg.BLUR_KERNEL))
        cfg_kwargs['MORPH_OPEN_KERNEL'] = int(rospy.get_param('~morph_open_kernel', default_cfg.MORPH_OPEN_KERNEL))
        cfg_kwargs['MORPH_CLOSE_KERNEL'] = int(rospy.get_param('~morph_close_kernel', default_cfg.MORPH_CLOSE_KERNEL))
        
        self.cfg = Config(**cfg_kwargs)
        
        self.bridge = CvBridge()
        
        # Setup Publishers
        self.lane_pub = rospy.Publisher(self.pub_topic, LaneData, queue_size=10)
        self.debug_pub = rospy.Publisher(self.debug_topic, Image, queue_size=1)
        self.binary_pub = rospy.Publisher(self.binary_topic, Image, queue_size=1)
        
        # Initialize Pipeline Modules
        self.tracker = LaneTracker(self.cfg)
        self.smoother = LaneSmoother(self.cfg)
        # 前瞻 anchor 的獨立平滑器，不可與近處 anchor 共用 filter 狀態。
        self.smoother_far = LaneSmoother(self.cfg)
        self.lane_width_history = deque(maxlen=self.cfg.LANE_WIDTH_HISTORY_LEN)
        
        self.frame_idx = 0
        self.prev_t = None
        self.fps_smoothed = 0.0
        self.fps_alpha = 0.1
        self.cap = None
        self.image_sub = None
        
        # Check if image_topic looks like a device port (/dev/videoX) or a number
        if str(self.image_topic).isdigit() or str(self.image_topic).startswith('/dev/video'):
            port = int(self.image_topic) if str(self.image_topic).isdigit() else self.image_topic
            self.cap = cv2.VideoCapture(port)
            if self.cap.isOpened():
                rospy.loginfo(f"LaneDetectNode initialized. Directly capturing from hardware: {self.image_topic}")
            else:
                rospy.logerr(f"Failed to open video port: {self.image_topic}")
        else:
            self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
            rospy.loginfo(f"LaneDetectNode initialized. Subscribing to: {self.image_topic}")

    def process_frame(self, cv_image):
        h, w = cv_image.shape[:2]
        now = rospy.Time.now().to_sec()
        
        # FPS Calculation
        sys_now = time.monotonic()
        if self.prev_t is None:
            self.prev_t = sys_now
            self.fps_smoothed = 0.0
        else:
            dt = max(sys_now - self.prev_t, 1e-6)
            self.prev_t = sys_now
            inst_fps = 1.0 / dt
            if self.fps_smoothed == 0.0:
                self.fps_smoothed = inst_fps
            else:
                self.fps_smoothed = self.fps_alpha * inst_fps + (1.0 - self.fps_alpha) * self.fps_smoothed

        self.frame_idx += 1
        
        frame = FrameContext(
            bgr=cv_image,
            timestamp=now,
            frame_idx=self.frame_idx,
            width=w,
            height=h
        )
        
        pre = process(frame, self.cfg)
        raw = find_lane_contours(pre, self.cfg)
        obs = self.tracker.update(raw, frame.width)
        meas = measure_at_anchor(obs, self.cfg, list(self.lane_width_history), frame.width, frame.height)
        smoothed = self.smoother.update(meas)

        # 前瞻 anchor：同一幀的觀測 obs，改在更前方的列上量測，獨立平滑後填到 *_far。
        meas_far = measure_at_anchor(
            obs, self.cfg, list(self.lane_width_history),
            frame.width, frame.height,
            anchor_ratio=self.cfg.ALIGN_ANCHOR_RATIO,
        )
        smoothed_far = self.smoother_far.update(meas_far)

        if meas.status == STATUS_OK and meas.lane_width_raw is not None:
            if meas.lane_width_raw > 0:
                self.lane_width_history.append(meas.lane_width_raw)

        msg = LaneData()
        msg.offset = smoothed.offset_px if smoothed.offset_px is not None else 0.0
        msg.angle = smoothed.yaw_deg if smoothed.yaw_deg is not None else 0.0
        msg.offset_far = smoothed_far.offset_px if smoothed_far.offset_px is not None else 0.0
        msg.angle_far = smoothed_far.yaw_deg if smoothed_far.yaw_deg is not None else 0.0
        self.lane_pub.publish(msg)
        
        if self.binary_pub.get_num_connections() > 0:
            try:
                binary_msg = self.bridge.cv2_to_imgmsg(pre.binary, "mono8")
                self.binary_pub.publish(binary_msg)
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error (binary image): {e}")

        if self.debug_pub.get_num_connections() > 0 or self.show_window:
            ri = RenderInputs(
                frame=frame,
                preprocess=pre,
                observation=obs,
                measure=meas,
                smooth=smoothed,
                fps=self.fps_smoothed
            )
            annotated = render(ri, self.cfg)
            
            if self.show_window:
                cv2.imshow("Lane Detection Debug", annotated)
                cv2.waitKey(1)
                
            if self.debug_pub.get_num_connections() > 0:
                try:
                    debug_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
                    self.debug_pub.publish(debug_msg)
                except CvBridgeError as e:
                    rospy.logerr(f"CvBridge Error (debug image): {e}")

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        self.process_frame(cv_image)

    def run(self):
        if self.cap is not None:
            rate = rospy.Rate(self.cfg.TARGET_FPS)
            while not rospy.is_shutdown():
                if not self.cap.isOpened():
                    rate.sleep()
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    rospy.logwarn_throttle(1.0, f"Cannot read frame from {self.image_topic}")
                    rate.sleep()
                    continue
                
                self.process_frame(frame)
                rate.sleep()
                
            self.cap.release()
            cv2.destroyAllWindows()
        else:
            rospy.spin()

if __name__ == "__main__":
    try:
        node = LaneDetectNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
