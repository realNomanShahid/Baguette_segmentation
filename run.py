

import argparse
import csv
import os
import sys

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Tunable parameters - adjust these first if results look off on new data
# ----------------------------------------------------------------------
class Params:
    # Step 2: thresholding
    gaussian_ksize = (5, 5)

   
    close_kernel_size = (3, 25)
    close_iterations = 2

    # Step 4: opening kernel to remove speckle noise
    open_kernel_size = (3, 3)
    open_iterations = 1

    # Step 5: shape-prior filters (tune to your camera resolution)
    min_area = 800
    min_height = 40
    max_width_to_height = 0.9
    min_width = 20                  # absolute min width (px) - kills thin
                                     # vertical streaks/reflections at seams
    min_width_to_height = 0.18      # lower bound on w/h - a real baguette
                                     # is chunky, not a hairline; kills streaks
    min_solidity = 0.80             # contour_area / convex_hull_area - a
                                     # baguette is smooth/capsule-shaped; kills
                                     # jagged/toothed machinery blobs
    max_angle_from_vertical = 30    # degrees; baguettes run ~vertically in
                                     # these frames - kills horizontal
                                     # belt/roller noise chains

    # Step 5b: ABSOLUTE physical-size sanity bounds (mm), independent of
    # any per-frame average.
    min_length_mm = 220
    max_length_mm = 420
    min_width_mm = 40
    max_width_mm = 110

    # Only trust the relative (avg-based) size filter once there are
    # enough real instances in the frame to make an average meaningful.
    min_instances_for_relative_filter = 5

    # ------------------------------------------------------------------
    # Step 6: ROI selection
    # ------------------------------------------------------------------
    # "fixed" -> use fixed_rois below, every frame, no detection.
    # "auto"  -> old automatic row-band detection (kept for debugging).
    # "full"  -> no ROI restriction, whole frame is valid.
    roi_mode = "fixed"

    # FIXED ROI rectangles: list of (x1, y1, x2, y2) in pixel
    # coordinates, in the *original, un-resized* frame. Fill these in
    # for your specific camera/rig - see "HOW TO GET YOUR COORDINATES"
    # in the module docstring. One rectangle per physically separate
    # row-block; most rigs only need one.
    #
    # Placeholder below covers the full frame minus the machinery strip
    # at the bottom - replace with your actual measured coordinates.
    fixed_rois = [
    (19, 143, 2048, 1084),
]

    # --- auto-ROI parameters, only used when roi_mode == "auto" ---
    roi_profile_smooth = 9
    roi_row_density_frac = 0.08
    roi_row_merge_gap = 20
    roi_min_band_height = 25
    roi_min_row_count_frac = 0.25
    roi_group_gap_factor = 2.2
    roi_padding = 15
    roi_min_group_total_frac = 0.30
    roi_min_group_width_frac = 0.45

    # Step 7: smoothing kernel (elliptical closing on each instance)
    smooth_kernel_size = (7, 7)

    # Step 9: watershed splitting trigger.
    split_width_ratio = 1.6
    watershed_min_distance = 15

    # Step 10: pixel-to-millimetre scale factor for physical measurements.
    #   mm = px / px_per_mm
    px_per_mm = 0.5

   
    length_squeeze_enabled = True
    length_squeeze_min_mm = 350
    length_squeeze_max_mm = 425
    length_squeeze_target_min_mm = 325
    length_squeeze_target_max_mm = 349

    # Step 11 (overlay labels): text scaling
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_min_scale = 0.28
    label_max_scale = 0.45
    label_id_thickness = 3
    label_dim_thickness = 3
    draw_axis_lines = True  # length (yellow) / width (red) guide lines
    label_color = (0, 255, 255)     # BGR yellow
    label_outline_color = (0, 0, 0)  # black outline for contrast
    roi_outline_color = (0, 220, 0)  # BGR green, drawn around each accepted ROI
    roi_outline_thickness = 3


# ----------------------------------------------------------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def raw_binary_mask(img, p: Params):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)
    L_blur = cv2.GaussianBlur(L, p.gaussian_ksize, 0)
    _, mask = cv2.threshold(
        L_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask


def bridge_and_denoise(mask, p: Params):
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, p.close_kernel_size)
    closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, close_k, iterations=p.close_iterations
    )
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, p.open_kernel_size)
    opened = cv2.morphologyEx(
        closed, cv2.MORPH_OPEN, open_k, iterations=p.open_iterations
    )
    return opened


def _component_shape_ok(labels, i, x, y, w, h, p: Params):
    """Contour-based shape checks: solidity (rejects jagged/toothed
    machinery blobs) and orientation (rejects horizontal belt/roller
    noise - baguettes run ~vertically in these frames)."""
    sub = np.uint8(labels[y:y + h, x:x + w] == i) * 255
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    cnt = max(contours, key=cv2.contourArea)
    cnt_area = cv2.contourArea(cnt)
    if cnt_area <= 0:
        return False

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = cnt_area / hull_area if hull_area > 0 else 0
    if solidity < p.min_solidity:
        return False

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    edge1 = box[1] - box[0]
    edge2 = box[2] - box[1]
    long_edge = edge1 if np.linalg.norm(edge1) >= np.linalg.norm(edge2) else edge2
    ang = abs(np.degrees(np.arctan2(long_edge[0], long_edge[1])))
    ang_from_vertical = min(ang, 180 - ang)
    if ang_from_vertical > p.max_angle_from_vertical:
        return False

    return True


def filter_components(binary, p: Params):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    keep = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]

        if area < p.min_area:
            continue
        if h < p.min_height:
            continue
        if w < p.min_width:
            continue
        if w > h * p.max_width_to_height:
            continue
        if w < h * p.min_width_to_height:
            continue

        if not _component_shape_ok(labels, i, x, y, w, h, p):
            continue

        keep.append(i)
    return labels, stats, centroids, keep


# ----------------------------------------------------------------------
# ROI SELECTION
# ----------------------------------------------------------------------
def get_fixed_roi_rects(img_shape, p: Params):
    """Validates and clamps the operator-supplied fixed ROI rectangles
    to the actual frame size, so a slightly-off manual coordinate
    doesn't crash anything - it just gets clipped to the frame edge."""
    img_h, img_w = img_shape[:2]
    if not p.fixed_rois:
        raise ValueError(
            "roi_mode is 'fixed' but Params.fixed_rois is empty. "
            "Set Params.fixed_rois (or pass --roi x1,y1,x2,y2) to your "
            "measured coordinates - see the module docstring."
        )
    rects = []
    for (x1, y1, x2, y2) in p.fixed_rois:
        rx1 = max(0, min(int(x1), img_w))
        ry1 = max(0, min(int(y1), img_h))
        rx2 = max(0, min(int(x2), img_w))
        ry2 = max(0, min(int(y2), img_h))
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        rects.append((rx1, ry1, rx2, ry2))
    return rects


def _row_density_profile(stats, keep_ids, img_h, p: Params):
    profile = np.zeros((img_h,), dtype=np.float32)
    for i in keep_ids:
        x, y, w, h, area = stats[i]
        yc = y + h / 2.0
        half = max(3.0, h * 0.12)
        y0 = max(0, int(yc - half))
        y1 = min(img_h, int(yc + half))
        profile[y0:y1] += 1.0

    k = max(1, p.roi_profile_smooth)
    if k > 1:
        kernel = np.ones(k, dtype=np.float32) / k
        profile = np.convolve(profile, kernel, mode="same")
    return profile


def _find_row_bands(profile, p: Params):
    peak = profile.max()
    if peak <= 0:
        return []
    thresh = peak * p.roi_row_density_frac
    is_content = profile > thresh

    bands = []
    y = 0
    H = len(is_content)
    while y < H:
        if is_content[y]:
            start = y
            while y < H and is_content[y]:
                y += 1
            bands.append([start, y])
        else:
            y += 1

    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= p.roi_row_merge_gap:
            merged[-1][1] = b[1]
        else:
            merged.append(list(b))

    merged = [b for b in merged if (b[1] - b[0]) >= p.roi_min_band_height]
    return merged


def _band_component_count(band, stats, keep_ids):
    y0, y1 = band
    count = 0
    for i in keep_ids:
        x, y, w, h, area = stats[i]
        yc = y + h / 2.0
        if y0 <= yc <= y1:
            count += 1
    return count


def detect_roi_regions_auto(stats, keep_ids, img_shape, p: Params):
    """Old automatic per-frame ROI detection. Kept only for debugging /
    recalibration (roi_mode == 'auto'); production runs should use the
    fixed ROI path instead - see get_fixed_roi_rects()."""
    img_h, img_w = img_shape[:2]

    if not keep_ids:
        return []

    profile = _row_density_profile(stats, keep_ids, img_h, p)
    bands = _find_row_bands(profile, p)
    if not bands:
        return []

    counts = [_band_component_count(b, stats, keep_ids) for b in bands]
    max_count = max(counts) if counts else 0
    min_count = max(1, int(round(max_count * p.roi_min_row_count_frac)))

    valid = [(b, c) for b, c in zip(bands, counts) if c >= min_count]
    if not valid:
        return []

    valid_bands = [b for b, c in valid]

    gaps = []
    for a, b in zip(valid_bands[:-1], valid_bands[1:]):
        gaps.append(b[0] - a[1])

    if gaps:
        median_gap = float(np.median(gaps))
        mad = float(np.median(np.abs(np.array(gaps) - median_gap)))
        if mad > 1e-6:
            gap_split_thresh = median_gap + p.roi_group_gap_factor * mad * 1.4826
        else:
            gap_split_thresh = max(median_gap * p.roi_group_gap_factor, p.roi_min_band_height * 2)
    else:
        gap_split_thresh = float("inf")

    groups = [[valid_bands[0]]]
    for prev, cur, gap in zip(valid_bands[:-1], valid_bands[1:], gaps):
        if gap > gap_split_thresh:
            groups.append([cur])
        else:
            groups[-1].append(cur)

    group_totals = [
        sum(_band_component_count(b, stats, keep_ids) for b in g) for g in groups
    ]
    max_group_total = max(group_totals) if group_totals else 0

    accepted_groups = []
    for group, total in zip(groups, group_totals):
        if max_group_total > 0 and total < max_group_total * p.roi_min_group_total_frac:
            continue

        xs0, xs1 = [], []
        for i in keep_ids:
            x, y, w, h, area = stats[i]
            yc = y + h / 2.0
            if any(gb[0] <= yc <= gb[1] for gb in group):
                xs0.append(x)
                xs1.append(x + w)
        if not xs0:
            continue
        span = max(xs1) - min(xs0)
        if span < img_w * p.roi_min_group_width_frac:
            continue

        accepted_groups.append(group)

    if not accepted_groups:
        return []

    roi_rects = []
    for group in accepted_groups:
        y0 = min(b[0] for b in group)
        y1 = max(b[1] for b in group)
        xs0, xs1 = [], []
        for i in keep_ids:
            x, y, w, h, area = stats[i]
            yc = y + h / 2.0
            if any(gb[0] <= yc <= gb[1] for gb in group):
                xs0.append(x)
                xs1.append(x + w)
        if not xs0:
            continue
        x0, x1 = min(xs0), max(xs1)
        pad = p.roi_padding
        rect = (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(img_w, x1 + pad),
            min(img_h, y1 + pad),
        )
        roi_rects.append(rect)

    return roi_rects


def get_roi_rects(stats, keep_ids, img_shape, p: Params):
    """Dispatches to fixed / auto / full ROI selection based on
    p.roi_mode. This is the single place that decides what ROI
    rectangles are used for a frame."""
    img_h, img_w = img_shape[:2]

    if p.roi_mode == "fixed":
        return get_fixed_roi_rects(img_shape, p)
    elif p.roi_mode == "auto":
        return detect_roi_regions_auto(stats, keep_ids, img_shape, p)
    elif p.roi_mode == "full":
        return [(0, 0, img_w, img_h)]
    else:
        raise ValueError(f"Unknown roi_mode: {p.roi_mode!r} (expected 'fixed', 'auto', or 'full')")


def filter_keep_ids_by_roi(stats, keep_ids, roi_rects):
    """Drop any component whose centroid does not fall inside at least
    one accepted ROI rectangle."""
    if not roi_rects:
        return []
    filtered = []
    for i in keep_ids:
        x, y, w, h, area = stats[i]
        cx, cy = x + w / 2.0, y + h / 2.0
        for (x1, y1, x2, y2) in roi_rects:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                filtered.append(i)
                break
    return filtered


# ----------------------------------------------------------------------
def compute_baguette_dimensions(instance_masks, p: Params = Params()):
    dimensions = []
    lengths_px = []
    widths_px = []

    for idx, mask in enumerate(instance_masks):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        length_px = max(w, h)
        width_px = min(w, h)
        length_mm = length_px / p.px_per_mm
        width_mm = width_px / p.px_per_mm

        lengths_px.append(length_px)
        widths_px.append(width_px)

        dimensions.append({
            "instance_id": idx,
            "length_px": length_px,
            "width_px": width_px,
            "length_mm": length_mm,
            "width_mm": width_mm,
        })

    avg_length_px = np.mean(lengths_px) if lengths_px else 0
    avg_width_px = np.mean(widths_px) if widths_px else 0
    avg_length_mm = avg_length_px / p.px_per_mm
    avg_width_mm = avg_width_px / p.px_per_mm

    return dimensions, avg_length_px, avg_width_px, avg_length_mm, avg_width_mm


# ----------------------------------------------------------------------
# LENGTH SQUEEZE (Step 10b) - trims oversized-length instances down
# ----------------------------------------------------------------------
def _target_length_mm_for_squeeze(length_mm, p: Params):
    """Linear, inverted mapping from the input band
    [length_squeeze_min_mm, length_squeeze_max_mm] to the output band
    [length_squeeze_target_min_mm, length_squeeze_target_max_mm].
    length_squeeze_min_mm -> length_squeeze_target_max_mm (smallest cut)
    length_squeeze_max_mm -> length_squeeze_target_min_mm (biggest cut)
    """
    lo, hi = p.length_squeeze_min_mm, p.length_squeeze_max_mm
    if hi <= lo:
        return p.length_squeeze_target_max_mm
    frac = (length_mm - lo) / (hi - lo)
    frac = min(1.0, max(0.0, frac))
    t_lo, t_hi = p.length_squeeze_target_min_mm, p.length_squeeze_target_max_mm
    return t_hi - frac * (t_hi - t_lo)


def _trim_mask_top_to_length(mask, target_length_px):
    """Cuts pixels off the TOP (smallest-y end) of the mask's vertical
    extent until its vertical extent equals target_length_px. Width
    (extent in x, at any given row) is left completely untouched - only
    whole rows are zeroed out from the top, nothing is trimmed
    sideways.
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return mask

    y_min, y_max = int(ys.min()), int(ys.max())
    current_length_px = (y_max - y_min) + 1
    target_length_px = max(1, int(round(target_length_px)))

    if current_length_px <= target_length_px:
        return mask  # already shorter than the target, nothing to cut

    cut_px = current_length_px - target_length_px
    new_top = y_min + cut_px

    trimmed = mask.copy()
    trimmed[:new_top, :] = 0
    return trimmed


def squeeze_long_instances(instance_masks, p: Params):
    """For every instance whose measured length falls in
    [length_squeeze_min_mm, length_squeeze_max_mm], trim pixels off the
    top of its mask so the new length lands in
    [length_squeeze_target_min_mm, length_squeeze_target_max_mm].
    Width is never modified. Instances outside the input band are
    returned unchanged. No-op entirely if length_squeeze_enabled is
    False.
    """
    if not p.length_squeeze_enabled:
        return instance_masks

    dimensions, *_ = compute_baguette_dimensions(instance_masks, p)
    dim_by_id = {d["instance_id"]: d for d in dimensions}

    squeezed = []
    for idx, mask in enumerate(instance_masks):
        dim = dim_by_id.get(idx)
        if dim is None:
            squeezed.append(mask)
            continue

        length_mm = dim["length_mm"]
        if p.length_squeeze_min_mm <= length_mm <= p.length_squeeze_max_mm:
            target_mm = _target_length_mm_for_squeeze(length_mm, p)
            target_px = target_mm * p.px_per_mm
            mask = _trim_mask_top_to_length(mask, target_px)

        squeezed.append(mask)

    return squeezed


def smooth_instance(labels, label_id, p: Params):
    inst = np.uint8(labels == label_id) * 255
    contours, _ = cv2.findContours(
        inst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return inst
    c = max(contours, key=cv2.contourArea)

    filled = np.zeros_like(inst)
    cv2.drawContours(filled, [c], -1, 255, thickness=-1)

    smooth_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, p.smooth_kernel_size
    )
    smoothed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, smooth_k)
    return smoothed


def maybe_split_merged(inst_mask, median_width, p: Params):
    x, y, w, h = cv2.boundingRect(inst_mask)
    if median_width <= 0 or w <= median_width * p.split_width_ratio:
        return [inst_mask]

    dist = cv2.distanceTransform(inst_mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    dist_norm = dist_norm.astype(np.uint8)

    _, sure_fg = cv2.threshold(
        dist_norm, 0.5 * dist_norm.max(), 255, cv2.THRESH_BINARY
    )
    sure_fg = np.uint8(sure_fg)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    if n_markers <= 2:
        return [inst_mask]

    markers = markers + 1
    unknown = cv2.subtract(inst_mask, sure_fg)
    markers[unknown == 255] = 0

    color_img = cv2.cvtColor(inst_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_img, markers)

    pieces = []
    for lbl in range(2, n_markers + 1):
        piece = np.uint8(markers == lbl) * 255
        if cv2.countNonZero(piece) > 0:
            pieces.append(piece)
    return pieces if pieces else [inst_mask]


def segment_baguettes(img, p: Params = Params()):
    raw = raw_binary_mask(img, p)
    bridged = bridge_and_denoise(raw, p)
    labels, cc_stats, centroids, keep_ids = filter_components(bridged, p)

    roi_rects = get_roi_rects(cc_stats, keep_ids, img.shape, p)
    keep_ids = filter_keep_ids_by_roi(cc_stats, keep_ids, roi_rects)

    if keep_ids:
        median_width = float(
            np.median([cc_stats[i, cv2.CC_STAT_WIDTH] for i in keep_ids])
        )
    else:
        median_width = 0.0

    instance_masks = []
    for i in keep_ids:
        smoothed = smooth_instance(labels, i, p)
        for piece in maybe_split_merged(smoothed, median_width, p):
            if cv2.countNonZero(piece) >= p.min_area:
                instance_masks.append(piece)

    final_mask = np.zeros_like(raw)
    stats_list = []
    for idx, inst in enumerate(instance_masks):
        final_mask = cv2.bitwise_or(final_mask, inst)
        x, y, w, h = cv2.boundingRect(inst)
        area = cv2.countNonZero(inst)
        M = cv2.moments(inst)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] else y + h // 2
        stats_list.append(
            {
                "instance_id": idx,
                "x": x, "y": y, "w": w, "h": h,
                "area": area, "centroid_x": cx, "centroid_y": cy,
            }
        )

    return final_mask, instance_masks, stats_list, roi_rects


def colored_instance_mask(img_shape, instance_masks):
    colored = np.zeros(img_shape, dtype=np.uint8)
    rng = np.random.default_rng(7)
    for inst in instance_masks:
        color = tuple(int(c) for c in rng.integers(40, 255, size=3))
        colored[inst > 0] = color
    return colored


def filter_by_absolute_size(instance_masks, stats_list, dimensions, p: Params):
    kept_masks, kept_stats, kept_dims = [], [], []
    for mask, stat, dim in zip(instance_masks, stats_list, dimensions):
        if not (p.min_length_mm <= dim["length_mm"] <= p.max_length_mm):
            continue
        if not (p.min_width_mm <= dim["width_mm"] <= p.max_width_mm):
            continue
        kept_masks.append(mask)
        kept_stats.append(stat)
        kept_dims.append(dim)
    return kept_masks, kept_stats, kept_dims


def remove_small_instances(
    instance_masks,
    stats_list,
    p: Params = Params(),
    length_ratio=0.8,
    width_ratio=0.8,
):
    dimensions, avg_length_px, avg_width_px, avg_length_mm, avg_width_mm = (
        compute_baguette_dimensions(instance_masks, p)
    )

    instance_masks, stats_list, dimensions = filter_by_absolute_size(
        instance_masks, stats_list, dimensions, p
    )

    if instance_masks:
        lengths_px = [d["length_px"] for d in dimensions]
        widths_px = [d["width_px"] for d in dimensions]
        avg_length_px = float(np.mean(lengths_px))
        avg_width_px = float(np.mean(widths_px))
        avg_length_mm = avg_length_px / p.px_per_mm
        avg_width_mm = avg_width_px / p.px_per_mm
    else:
        avg_length_px = avg_width_px = 0.0
        avg_length_mm = avg_width_mm = 0.0

    if len(instance_masks) >= p.min_instances_for_relative_filter:
        length_threshold = avg_length_px * length_ratio
        width_threshold = avg_width_px * width_ratio

        filtered_masks, filtered_stats, filtered_dims = [], [], []
        for mask, stat, dim in zip(instance_masks, stats_list, dimensions):
            if (
                dim["length_px"] >= length_threshold
                and dim["width_px"] >= width_threshold
            ):
                filtered_masks.append(mask)
                filtered_stats.append(stat)
                filtered_dims.append(dim)
        instance_masks, stats_list, dimensions = filtered_masks, filtered_stats, filtered_dims

    return instance_masks, stats_list, dimensions, avg_length_mm, avg_width_mm


def clean_rough_instances(instance_masks, area_drop_threshold=0.05, kernel_scale=0.35):
    cleaned = []
    for mask in instance_masks:
        orig_area = cv2.countNonZero(mask)
        if orig_area == 0:
            cleaned.append(mask)
            continue

        x, y, w, h = cv2.boundingRect(mask)
        k = max(3, int(w * kernel_scale))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            cleaned.append(mask)
            continue

        c = max(contours, key=cv2.contourArea)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [c], -1, 255, thickness=-1)

        new_area = cv2.countNonZero(filled)
        area_ratio = new_area / orig_area

        if area_ratio >= (1 - area_drop_threshold):
            cleaned.append(mask)
        elif area_ratio >= 0.5:
            smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            smoothed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, smooth_k)
            cleaned.append(smoothed)
        else:
            cleaned.append(mask)

    return cleaned


# ----------------------------------------------------------------------
# CLEAN LABEL RENDERING
# ----------------------------------------------------------------------
def _fit_text_scale(text, target_width_px, font, thickness, min_scale, max_scale):
    lo, hi = min_scale, max_scale
    best = min_scale
    for _ in range(12):
        mid = (lo + hi) / 2
        (tw, th), _ = cv2.getTextSize(text, font, mid, thickness)
        if tw <= target_width_px:
            best = mid
            lo = mid
        else:
            hi = mid
    return best


def _put_centered_text(img, text, center_x, center_y, font, scale, thickness, color, outline_color):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    org = (int(center_x - tw / 2), int(center_y + th / 2))
    cv2.putText(img, text, org, font, scale, outline_color, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_roi_outlines(img, roi_rects, p: Params = Params()):
    for (x1, y1, x2, y2) in roi_rects:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                       p.roi_outline_color, p.roi_outline_thickness, cv2.LINE_AA)
    return img


def draw_overlay(img, instance_masks, p: Params = Params(), roi_rects=None):
    overlay = img.copy()
    rng = np.random.default_rng(42)

    for idx, inst in enumerate(instance_masks, start=1):
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        overlay[inst > 0] = color

        contours, _ = cv2.findContours(
            inst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        (rcx, rcy), (w, h), angle = rect
        x, y, bw, bh = cv2.boundingRect(cnt)

        length_px = max(w, h)
        width_px = min(w, h)
        length_mm = length_px / p.px_per_mm
        width_mm = width_px / p.px_per_mm

        cx = x + bw / 2
        cy = y + bh / 2

        target_w = max(10, int(bw * 1.2))

        if p.draw_axis_lines:
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            mids = [((box[i] + box[(i + 1) % 4]) / 2).astype(int) for i in range(4)]
            edge1 = np.linalg.norm(box[0] - box[1])
            edge2 = np.linalg.norm(box[1] - box[2])
            if edge1 >= edge2:
                length_start, length_end = mids[3], mids[1]
                width_start, width_end = mids[0], mids[2]
            else:
                length_start, length_end = mids[0], mids[2]
                width_start, width_end = mids[1], mids[3]
            cv2.line(overlay, tuple(length_start), tuple(length_end), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(overlay, tuple(width_start), tuple(width_end), (0, 0, 255), 1, cv2.LINE_AA)

        id_text = str(idx)
        dim_text = f"{length_mm:.0f}x{width_mm:.0f}"

        id_scale = _fit_text_scale(
            id_text, target_w, p.label_font, p.label_id_thickness,
            p.label_min_scale, p.label_max_scale,
        )
        dim_scale = _fit_text_scale(
            dim_text, target_w, p.label_font, p.label_dim_thickness,
            p.label_min_scale, p.label_max_scale,
        )
        scale = min(id_scale, dim_scale)

        (_, id_th), _ = cv2.getTextSize(id_text, p.label_font, scale, p.label_id_thickness)
        line_gap = int(id_th * 1.6)

        dim_y = int(cy + line_gap * 0.65)

        _put_centered_text(overlay, dim_text, cx, dim_y, p.label_font, scale * 1,
                            p.label_dim_thickness, p.label_color, p.label_outline_color)

    if roi_rects:
        draw_roi_outlines(overlay, roi_rects, p)

    blended = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
    return blended


# ----------------------------------------------------------------------
def process_one(path, outdir, p: Params = Params()):
    img = load_image(path)
    final_mask, instance_masks, stats_list, roi_rects = segment_baguettes(img, p)

    instance_masks = clean_rough_instances(instance_masks)
    instance_masks = squeeze_long_instances(instance_masks, p)
    instance_masks, stats_list, dimensions, avg_length_mm, avg_width_mm = (
        remove_small_instances(
            instance_masks, stats_list, p, length_ratio=0.7, width_ratio=0.7,
        )
    )

    final_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for mask in instance_masks:
        final_mask = cv2.bitwise_or(final_mask, mask)

    for stat, dim in zip(stats_list, dimensions):
        stat["length_px"] = round(dim["length_px"], 2)
        stat["width_px"] = round(dim["width_px"], 2)
        stat["length_mm"] = round(dim["length_mm"], 2)
        stat["width_mm"] = round(dim["width_mm"], 2)

    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(outdir, exist_ok=True)

    mask_path = os.path.join(outdir, f"{base}_mask.png")
    overlay_path = os.path.join(outdir, f"{base}_overlay.png")
    instances_path = os.path.join(outdir, f"{base}_instances.png")
    csv_path = os.path.join(outdir, f"{base}_stats.csv")

    cv2.imwrite(mask_path, final_mask)
    cv2.imwrite(overlay_path, draw_overlay(img, instance_masks, p, roi_rects=roi_rects))
    cv2.imwrite(instances_path, colored_instance_mask(img.shape, instance_masks))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance_id", "x", "y", "w", "h",
                "area", "centroid_x", "centroid_y",
                "length_px", "width_px", "length_mm", "width_mm",
            ],
        )
        writer.writeheader()
        writer.writerows(stats_list)

    print(
        f"[{base}] detected {len(instance_masks)} baguette instances in "
        f"{len(roi_rects)} ROI region(s) (roi_mode={p.roi_mode}) "
        f"(avg {avg_length_mm:.1f}mm x {avg_width_mm:.1f}mm, "
        f"px_per_mm={p.px_per_mm}) -> "
        f"{mask_path}, {overlay_path}, {instances_path}, {csv_path}"
    )
    return final_mask, instance_masks, stats_list, roi_rects


# ----------------------------------------------------------------------
# VIDEO PROCESSING
# ----------------------------------------------------------------------
def process_frame(frame, p: Params = Params()):
    final_mask, instance_masks, stats_list, roi_rects = segment_baguettes(frame, p)

    instance_masks = clean_rough_instances(instance_masks)
    instance_masks = squeeze_long_instances(instance_masks, p)
    instance_masks, stats_list, dimensions, avg_length_mm, avg_width_mm = (
        remove_small_instances(
            instance_masks, stats_list, p, length_ratio=0.7, width_ratio=0.7,
        )
    )

    for stat, dim in zip(stats_list, dimensions):
        stat["length_px"] = round(dim["length_px"], 2)
        stat["width_px"] = round(dim["width_px"], 2)
        stat["length_mm"] = round(dim["length_mm"], 2)
        stat["width_mm"] = round(dim["width_mm"], 2)

    return instance_masks, stats_list, dimensions, avg_length_mm, avg_width_mm, roi_rects


def draw_frame_header(img, count, p: Params = Params(), avg_length_mm=None,
                       avg_width_mm=None, frame_idx=None, total_frames=None):
    parts = []
    if frame_idx is not None:
        if total_frames:
            parts.append(f"Frame {frame_idx}/{total_frames}")
        else:
            parts.append(f"Frame {frame_idx}")
    parts.append(f"Baguettes: {count}")
    if avg_length_mm is not None and count > 0:
        parts.append(f"L~{avg_length_mm:.0f}mm W~{avg_width_mm:.0f}mm")
    text = "   ".join(parts)

    font = p.label_font
    scale = 0.9
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 10
    cv2.rectangle(img, (0, 0), (tw + 2 * pad, th + baseline + 2 * pad), (0, 0, 0), -1)
    cv2.putText(img, text, (pad, th + pad), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return img


def process_video(
    video_path,
    outdir,
    p: Params = Params(),
    save_frames=True,
    out_video_name=None,
    frame_skip=1,
    csv_log=True,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    os.makedirs(outdir, exist_ok=True)
    frames_dir = os.path.join(outdir, "frames")
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    out_video_path = os.path.join(outdir, out_video_name or "processed_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps / max(1, frame_skip)
    writer = cv2.VideoWriter(out_video_path, fourcc, out_fps, (W, H))

    csv_path = os.path.join(outdir, "video_stats.csv")
    csv_file = open(csv_path, "w", newline="") if csv_log else None
    csv_writer = None
    if csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame_idx", "instance_id", "x", "y", "w", "h",
                "area", "centroid_x", "centroid_y",
                "length_px", "width_px", "length_mm", "width_mm",
            ],
        )
        csv_writer.writeheader()

    frame_idx = 0
    processed = 0
    counts = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            (instance_masks, stats_list, dimensions,
             avg_length_mm, avg_width_mm, roi_rects) = process_frame(frame, p)

            overlay = draw_overlay(frame, instance_masks, p, roi_rects=roi_rects)
            draw_frame_header(
                overlay, len(instance_masks), p,
                avg_length_mm=avg_length_mm, avg_width_mm=avg_width_mm,
                frame_idx=frame_idx, total_frames=total_frames,
            )

            writer.write(overlay)
            if save_frames:
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:05d}.png"), overlay)
            if csv_writer:
                for stat in stats_list:
                    row = {"frame_idx": frame_idx, **stat}
                    csv_writer.writerow(row)

            counts.append(len(instance_masks))
            processed += 1
            if processed % 25 == 0:
                print(f"  processed frame {frame_idx} ({len(instance_masks)} baguettes)...")

        frame_idx += 1

    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()

    avg_count = float(np.mean(counts)) if counts else 0.0
    print(
        f"[video] processed {processed}/{frame_idx} frames "
        f"(frame_skip={frame_skip}), avg {avg_count:.1f} baguettes/frame -> "
        f"{out_video_path}"
        + (f", {frames_dir}/" if save_frames else "")
        + (f", {csv_path}" if csv_log else "")
    )
    return out_video_path


def _parse_roi_arg(s):
    """Parses '--roi x1,y1,x2,y2' into a (x1,y1,x2,y2) int tuple."""
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--roi expects 'x1,y1,x2,y2', got: {s!r}"
        )
    try:
        x1, y1, x2, y2 = (int(v.strip()) for v in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--roi values must be integers, got: {s!r}"
        )
    return (x1, y1, x2, y2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="image file or folder of images")
    parser.add_argument("--video", help="video file to process instead of images")
    parser.add_argument("--outdir", default="./baguette_output")
    parser.add_argument("--px-per-mm", type=float, default=Params.px_per_mm)
    parser.add_argument("--roi-mode", choices=["fixed", "auto", "full"],
                         default=Params.roi_mode,
                         help="'fixed' (default): use --roi rectangle(s), no per-frame "
                              "detection. 'auto': old automatic row-band detection. "
                              "'full': no ROI restriction.")
    parser.add_argument("--roi", action="append", type=_parse_roi_arg, default=None,
                         metavar="x1,y1,x2,y2",
                         help="fixed ROI rectangle in pixel coords, repeatable for "
                              "multiple row-blocks (e.g. split by a physical seam). "
                              "Only used when --roi-mode=fixed. If omitted, "
                              "Params.fixed_rois in the script is used instead.")
    parser.add_argument("--min-length-mm", type=float, default=Params.min_length_mm,
                         help="reject instances shorter than this (mm), independent of frame average")
    parser.add_argument("--max-length-mm", type=float, default=Params.max_length_mm,
                         help="reject instances longer than this (mm), independent of frame average")
    parser.add_argument("--min-width-mm", type=float, default=Params.min_width_mm,
                         help="reject instances narrower than this (mm), independent of frame average")
    parser.add_argument("--max-width-mm", type=float, default=Params.max_width_mm,
                         help="reject instances wider than this (mm), independent of frame average")
    parser.add_argument("--frame-skip", type=int, default=1,
                         help="process every Nth frame of the video (default: every frame)")
    parser.add_argument("--no-save-frames", action="store_true",
                         help="video mode: don't save every processed frame as a PNG, only the output video")
    parser.add_argument("--no-csv", action="store_true",
                         help="video mode: skip writing the per-frame stats CSV")
    parser.add_argument("--no-length-squeeze", action="store_true",
                         help="disable the length-squeeze step (Step 10b): by default, "
                              f"instances measuring {Params.length_squeeze_min_mm}-"
                              f"{Params.length_squeeze_max_mm}mm long are trimmed from "
                              f"the top down to {Params.length_squeeze_target_min_mm}-"
                              f"{Params.length_squeeze_target_max_mm}mm; width is never touched.")
    args = parser.parse_args()

    p = Params()
    p.px_per_mm = args.px_per_mm
    p.min_length_mm = args.min_length_mm
    p.max_length_mm = args.max_length_mm
    p.min_width_mm = args.min_width_mm
    p.max_width_mm = args.max_width_mm
    p.roi_mode = args.roi_mode
    if args.roi:
        p.fixed_rois = args.roi
    if args.no_length_squeeze:
        p.length_squeeze_enabled = False

    if p.roi_mode == "fixed" and not p.fixed_rois:
        print(
            "roi_mode is 'fixed' but no ROI was given. Pass one or more "
            "--roi x1,y1,x2,y2 flags, or edit Params.fixed_rois in the script.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.video:
        process_video(
            args.video, args.outdir, p,
            save_frames=not args.no_save_frames,
            frame_skip=max(1, args.frame_skip),
            csv_log=not args.no_csv,
        )
        return

    if not args.input:
        print("Provide either --input (image/folder) or --video (video file)")
        sys.exit(1)

    if os.path.isdir(args.input):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        files = sorted(f for f in os.listdir(args.input) if f.lower().endswith(exts))
        if not files:
            print(f"No images found in {args.input}")
            sys.exit(1)
        for f in files:
            process_one(os.path.join(args.input, f), args.outdir, p)
    else:
        process_one(args.input, args.outdir, p)


if __name__ == "__main__":
    main()