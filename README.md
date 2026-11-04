# CVAT_utils

Utilities for loading annotation files that were created with the CVAT annotation toolkit (https://www.cvat.ai/).
Standalone loader for CVAT XML annotation exports. Parses polygons, provides bounding boxes for them, and renders pixel masks for each frame.
Used for image stream annotations that were annotated with polygons.

**Dependencies:** `numpy`, `xmltodict`, `Pillow`

---

## Loading

### `CVATLoaderUtil(xmlPath)`
Loads a CVAT XML file, parses all frames, and removes duplicate boxes (IoU >= 0.5).

### `loadAnnotations()`
Internal parser. Reads polygons, computes bounding boxes, and renders binary masks for every frame.

---

## Frame access

### `getFileNames()` / `getFirstFileName()` / `getLastFileName()`
List of frame names in annotation order, or the first/last alphabetically.

### `getPolygonsByImgName(fileName)`
Polygon vertex lists for one frame, as stored in the XML.

### `getBoxesByFileName(fileName)` / `getBoxesByImgName(fileName)`
Bounding boxes `[x1, y1, x2, y2]` for one frame. Returns `[]` if the frame is missing.

### `getMaskByFileName(fileName)` / `getMaskByIdx(idx)`
Binary pixel mask (shape: width x height) for a frame. Returns a zero mask if not found.

### `getSeparateMasksByFileName(fileName)`
One mask per polygon in a frame, instead of the merged result.

### `getShape()`
Returns `(width, height)` of the annotated frames.

---

## All-frame access

### `getPolygonsPerFrame()` / `getBoxesPerFrame()`
A list with one entry per frame: all polygons or all boxes.

---

## Checks

### `hasData()` / `hasEntryForFileName(fileName)` / `firstAnnotatedFrameName()` / `__len__`
Whether data exists, whether a specific frame is present, which frame has the first real polygon, and how many frames are loaded.

---

## Modification

### `removeDuplicates(iouThreshold=0.5)`
Drops boxes whose IoU with a neighbor exceeds the threshold. Keeps the first box in each cluster.

### `cropToRegion(x1, y1, x2, y2)`
Crops all polygons, boxes, and masks to a rectangle. Boxes smaller than 50% of their original area (or < 10 px) are dropped.

### `boxesToEllipses(alsoPolys=False, sizeFactors=(1,1))`
Replaces box-shaped polygons (3-7 vertices) with fitted ellipses. With `alsoPolys=True`, all polygons are replaced. Rebuilds masks.

### `deleteEntry(fileName)`
Removes a frame from the annotation set.

---

## Analysis

### `getAnnotationArea(intervalBorder=1.0)`
Median, min, and max area (in pixels) covered by annotation polygons. `intervalBorder < 1.0` clips to a percentile range.

### `getSideFactors(intervalBorder=1.0)`
Median, min, and max aspect ratio of polygon bounding boxes.

---

## Helper functions (module-level)

### `maskImgFromPolygons(polygons, width, height)`
Renders a list of polygons into one merged binary mask.

### `maskFromPolygon(data)`
Renders a single polygon into a binary array via PIL.

### `drawEllipseFromBox(mask, box, sizeFactors)`
Draws a filled ellipse into a mask. Returns the updated mask and an approximate polygon (8 points).

### `resizeBoxSeparate(x1, y1, x2, y2, factorX, factorY)`
Scales a box around its center.

### `boxIoU(boxA, boxB)` / `boxIoUVector(box, boxes)`
Intersection-over-union for two boxes, or one box against many.

### `calcCliques(edges, nodes)`
Groups connected node indices into cliques.

### `mergeableBoxCliquesByIoU(boxes, iouThreshold)`
Finds groups of boxes that overlap above the threshold.

### `scale(x, out_range=(-1, 1))`
Linearly maps an array into the given range.
