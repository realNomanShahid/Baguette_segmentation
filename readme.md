# Donut Mask Segmentation

## Overview

This task performs automatic segmentation of donut  objects from RGB images using classical computer vision techniques. The pipeline detects donut candidates, extracts their ring-shaped masks, validates their geometry, and refines the final segmentation using post-processing operations.

---

## Pipeline

```
Input Image
      │
      ▼
Hough Circle Detection
      │
      ▼
ROI Extraction
      │
      ▼
LAB Color Thresholding (Local Otsu)
      │
      ▼
Morphological Filtering
      │
      ▼
Connected Components
      │
      ▼
Shape Validation
      │
      ▼
Ring Mask Generation
      │
      ▼
Merge All Masks
      │
      ▼
Post-Processing
      │
      ▼
Final Binary Mask
```

---

## Detection Process

- Detect donut candidates using the Hough Circle Transform.
- Extract a local ROI around each detected circle.
- Perform local thresholding on the LAB color space.
- Apply morphological operations to remove small noise and fill small gaps.
- Keep only the connected component nearest to the detected circle.
- Recover the donut hole using flood-fill.
- Validate each donut using:
  - Hole ratio
  - Circularity
  - Concentricity
- Generate an individual ring mask for each valid donut.
- Merge all valid donut masks into a single binary mask.

---

# Post-Processing

The post-processing stage improves the quality of the final segmentation mask by removing artifacts, smoothing boundaries, and preserving donut holes.

## Steps

### 1. Morphological Opening
- Removes small isolated noise.
- Eliminates tiny foreground pixels.

### 2. Morphological Closing
- Fills small gaps.
- Connects nearby foreground pixels.

### 3. Dilation
- Slightly expands the object boundary.
- Improves edge continuity.

### 4. Connected Components
- Labels all connected regions.
- Removes components smaller than the minimum area threshold.
- Keeps only valid donut regions.

### 5. Contour Detection
- Finds contours using `RETR_CCOMP`.
- Preserves contour hierarchy to distinguish:
  - Outer boundary
  - Inner donut hole

### 6. Contour Smoothing
- Smooths contour boundaries using `approxPolyDP`.
- Reduces jagged edges while maintaining shape.

### 7. Hole Preservation
- Outer contours are filled with white.
- Inner contours (holes) are filled with black.
- Preserves the donut structure.

### 8. Final Output
- Produces a clean binary mask with:
  - Reduced noise
  - Smooth boundaries
  - Preserved inner holes

---


## Post-Processing Workflow

![Post-Processing Workflow](docs/workflow.jpg)



## Libraries

- OpenCV
- NumPy