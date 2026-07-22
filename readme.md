# Donut Mask Segmentation

Automatic segmentation of donut objects from RGB images using classical computer vision (no deep learning). The pipeline locates donut candidates, extracts ring-shaped masks, validates their geometry, and refines the final segmentation through a post-processing stage.

## Demo

📹 *Video demo placeholder — add your link here:*
`[Watch the demo video](YOUR_VIDEO_LINK_HERE)`

---

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

```bash
pip install opencv-python numpy
```

## Usage

```bash
python donut_segment.py --input path/to/image_or_folder --outdir ./output
```

| Flag | Description | Default |
|---|---|---|
| `--input` | Image file or folder of images | required |
| `--outdir` | Output directory for masks / overlays / stats | `./output` |
| `--min-area` | Minimum accepted component area (px²) | see config |
| `--hough-dp` / `--hough-min-dist` / `--hough-param1` / `--hough-param2` | Hough circle detection tuning | see config |
| `--min-hole-ratio` / `--max-hole-ratio` | Valid donut hole-to-outer-area ratio bounds | see config |
| `--min-circularity` | Minimum accepted circularity (0–1) | see config |
| `--max-concentricity-offset` | Max allowed center offset between hole and outer ring | see config |

> Adjust flag names above to match your actual `argparse` definitions if they differ — keep this table in sync with the script's `--help` output.

---

## Pipeline

```
Input Image
      │
      ▼
Hough Circle Detection      → locates donut candidates
      │
      ▼
ROI Extraction               → crops a local region per candidate
      │
      ▼
LAB Color Thresholding (Local Otsu)
      │
      ▼
Morphological Filtering      → denoise + close small gaps
      │
      ▼
Connected Components         → keep component nearest detected circle
      │
      ▼
Shape Validation             → hole ratio, circularity, concentricity
      │
      ▼
Ring Mask Generation          → per-instance donut mask
      │
      ▼
Merge All Masks               → combine into one frame-level mask
      │
      ▼
Post-Processing               → clean, smooth, preserve holes
      │
      ▼
Final Binary Mask
```

---

## Detection Process

1. **Hough Circle Transform** — detects candidate donut centers/radii in the full image.
2. **ROI extraction** — crops a padded region around each candidate for local processing (avoids global-threshold bias from other donuts/background).
3. **Local LAB thresholding** — converts the ROI to LAB, applies Otsu thresholding on the L channel to separate donut from background/plate.
4. **Morphological cleanup** — opening/closing to remove speckle noise and fill small gaps in the raw threshold mask.
5. **Connected components** — keeps only the component whose centroid is nearest the Hough-detected circle center, discarding unrelated blobs in the ROI.
6. **Hole recovery** — flood-fills from the ROI border to recover the donut's central hole, which is otherwise indistinguishable from background.
7. **Shape validation** — a candidate is accepted only if it passes all three:
   - **Hole ratio**: hole area / outer area falls within an expected donut range.
   - **Circularity**: `4π × area / perimeter²` close to 1 (rejects irregular blobs).
   - **Concentricity**: hole center and outer-ring center are close together (rejects crescents/partial rings).
8. **Ring mask generation** — builds a clean binary ring mask (outer filled, hole subtracted) per validated instance.
9. **Merge** — all accepted instance masks are OR'd into one frame-level binary mask.

---

## Post-Processing

Improves the merged mask's quality by removing artifacts, smoothing boundaries, and correctly preserving donut holes.

### 1. Morphological opening
Removes small isolated noise / stray foreground speckles left over from thresholding.

### 2. Morphological closing
Fills small gaps and reconnects foreground pixels that thresholding split apart.

### 3. Dilation
Slightly expands object boundaries for better edge continuity going into contour detection.

### 4. Connected components (area filter)
Labels all remaining regions and drops any component below the minimum area threshold, keeping only plausible donut-sized regions.

### 5. Contour detection (`RETR_CCOMP`)
Extracts contours with hierarchy so outer boundaries and inner holes are distinguished rather than treated as one blob.

### 6. Contour smoothing (`approxPolyDP`)
Reduces jagged, staircase-like edges from the raster mask while preserving the overall ring shape.

### 7. Hole preservation
Outer contours are filled white; inner (hole) contours are then punched out as black — this is what keeps the donut's hole from being accidentally filled in during closing/dilation.

### 8. Final output
A clean binary mask with reduced noise, smoothed boundaries, and correctly preserved inner holes.

---

## Output Files

| File | Description |
|---|---|
| `*_mask.png` | Final merged binary mask |
| `*_overlay.png` | Original image with detected donuts overlaid/labeled |
| `*_instances.png` | Color-coded per-instance mask |
| `*_stats.csv` | Per-instance geometry: center, radius, hole ratio, circularity, area |

---

## AI & Backend Checklist

Use this before merging or deploying a new version of the pipeline.

### Detection accuracy
- [ ] Hough parameters tuned/re-validated against current camera resolution and lighting
- [ ] False positives (non-donut circles) checked against a labeled validation set
- [ ] False negatives (missed donuts) checked, especially at image edges / overlapping donuts
- [ ] Shape validation thresholds (hole ratio, circularity, concentricity) re-checked after any parameter change

### Post-processing correctness
- [ ] Hole preservation verified — no instances with holes accidentally filled
- [ ] Contour smoothing doesn't distort small donuts disproportionately
- [ ] Morphological kernel sizes appropriate for current image resolution (re-tune if resolution changes)
- [ ] No cross-instance mask bleeding after dilation/closing

### Performance & robustness
- [ ] Runtime benchmarked on target hardware (per-image and, if applicable, per-video-frame)
- [ ] Pipeline handles empty/no-detection frames without crashing
- [ ] Pipeline handles malformed/corrupt input images gracefully (clear error, no silent failure)
- [ ] Batch/folder mode tested on a full representative dataset, not just single images

### Backend / integration
- [ ] Output file naming and CSV schema match what the consuming backend service expects
- [ ] API/CLI contract (arguments, defaults, exit codes) documented and unchanged unless versioned
- [ ] Logging in place for detection counts, rejected candidates, and processing time per image
- [ ] Config values (thresholds, kernel sizes, area limits) externalized, not hardcoded, for easy re-tuning per deployment
- [ ] Version/tag recorded alongside output artifacts for traceability

### Testing
- [ ] Unit tests cover shape validation logic (hole ratio, circularity, concentricity) independently of image I/O
- [ ] Regression test set of known images with expected donut counts kept up to date
- [ ] CI runs the pipeline end-to-end on sample data before merge

---

## Libraries

- OpenCV
- NumPy