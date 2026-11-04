# -*- coding: utf-8 -*-
"""Tests for the standalone CVAT loader."""

import os
import os.path as osp
import sys
import unittest

import numpy as np

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))

from CVAT_utils.cvat_loader import (
    CVATLoaderUtil,
    boxIoU,
    boxIoUVector,
    calcCliques,
    drawEllipseFromBox,
    maskImgFromPolygons,
    mergeableBoxCliquesByIoU,
    resizeBoxSeparate,
    scale,
)

XML_PATH = osp.join(osp.dirname(__file__), "annotations_example.xml")


class TestCVATLoaderUtil(unittest.TestCase):

    def setUp(self):
        self.loader = CVATLoaderUtil(XML_PATH)

    def test_init_loads_frames(self):
        """All 4 frames from the example XML end up in the loader."""
        self.assertEqual(len(self.loader), 4)
        self.assertTrue(self.loader.hasData())

    def test_owner_read(self):
        """The owner username is pulled from the XML meta block."""
        self.assertEqual(self.loader.owner, "test.user")

    def test_file_names(self):
        """getFileNames returns every frame name present in the XML."""
        names = self.loader.getFileNames()
        self.assertEqual(len(names), 4)
        self.assertIn("frame000.png", names)
        self.assertIn("frame003.png", names)

    def test_first_last_filename(self):
        """Alphabetical ordering gives frame000 first and frame003 last."""
        self.assertEqual(self.loader.getFirstFileName(), "frame000.png")
        self.assertEqual(self.loader.getLastFileName(), "frame003.png")

    def test_shape(self):
        """Frame dimensions match the XML attributes (100x50)."""
        self.assertEqual(self.loader.getShape(), (100, 50))

    def test_polygons_per_frame(self):
        """Each frame holds the expected number of polygons (2, 1, 3, 0)."""
        polys = self.loader.getPolygonsPerFrame()
        self.assertEqual(len(polys), 4)
        self.assertEqual(len(polys[0]), 2)
        self.assertEqual(len(polys[1]), 1)
        self.assertEqual(len(polys[2]), 3)
        self.assertEqual(len(polys[3]), 0)

    def test_boxes_per_frame(self):
        """Box counts match polygon counts per frame."""
        boxes = self.loader.getBoxesPerFrame()
        self.assertEqual(len(boxes), 4)
        self.assertEqual(len(boxes[0]), 2)
        self.assertEqual(len(boxes[3]), 0)

    def test_boxes_by_filename(self):
        """Boxes have 4 coords and x1 < x2, y1 < y2."""
        boxes = self.loader.getBoxesByFileName("frame000.png")
        self.assertEqual(len(boxes), 2)
        for box in boxes:
            self.assertEqual(len(box), 4)
            self.assertLess(box[0], box[2])
            self.assertLess(box[1], box[3])

    def test_boxes_by_filename_missing(self):
        """A nonexistent frame returns an empty list, not an error."""
        self.assertEqual(self.loader.getBoxesByFileName("nope.png"), [])

    def test_mask_by_filename(self):
        """The mask has the right shape and contains set pixels."""
        mask = self.loader.getMaskByFileName("frame000.png")
        self.assertEqual(mask.shape[0], 100)
        self.assertEqual(mask.shape[1], 50)
        self.assertGreater(mask.sum(), 0)

    def test_mask_by_filename_missing(self):
        """A missing frame yields a zero-filled mask."""
        mask = self.loader.getMaskByFileName("nope.png")
        self.assertEqual(mask.sum(), 0)

    def test_mask_by_idx(self):
        """Index 0 gives a real mask; an out-of-range index gives zeros."""
        mask = self.loader.getMaskByIdx(0)
        self.assertGreater(mask.sum(), 0)
        mask_last = self.loader.getMaskByIdx(99)
        self.assertEqual(mask_last.sum(), 0)

    def test_polygons_by_img_name(self):
        """frame002 has 3 polygons, each with at least 3 vertices."""
        polys = self.loader.getPolygonsByImgName("frame002.png")
        self.assertEqual(len(polys), 3)
        for poly in polys:
            self.assertGreaterEqual(len(poly), 3)

    def test_boxes_by_img_name(self):
        """frame001 has exactly 1 box stored."""
        boxes = self.loader.getBoxesByImgName("frame001.png")
        self.assertEqual(len(boxes), 1)

    def test_has_entry(self):
        """Existing frames are found; made-up names are not."""
        self.assertTrue(self.loader.hasEntryForFileName("frame000.png"))
        self.assertFalse(self.loader.hasEntryForFileName("missing.png"))

    def test_first_annotated_frame(self):
        """The first frame with actual polygon data is frame000."""
        self.assertEqual(
            self.loader.firstAnnotatedFrameName(), "frame000.png")

    def test_delete_entry(self):
        """Removing a frame drops it from the annotation set."""
        self.loader.deleteEntry("frame001.png")
        self.assertEqual(len(self.loader), 3)
        self.assertFalse(self.loader.hasEntryForFileName("frame001.png"))

    def test_delete_entry_missing(self):
        """Deleting a frame that does not exist leaves the count unchanged."""
        self.loader.deleteEntry("nonexistent.png")
        self.assertEqual(len(self.loader), 4)

    def test_annotation_area(self):
        """Polygon areas are positive and min <= median <= max."""
        med, lo, hi = self.loader.getAnnotationArea()
        self.assertGreater(med, 0)
        self.assertGreaterEqual(hi, lo)

    def test_side_factors(self):
        """Aspect ratios are positive and ordered correctly."""
        med, lo, hi = self.loader.getSideFactors()
        self.assertGreater(med, 0)
        self.assertGreaterEqual(hi, lo)

    def test_separate_masks(self):
        """Each polygon in frame000 produces its own non-empty mask."""
        masks = self.loader.getSeparateMasksByFileName("frame000.png")
        self.assertEqual(len(masks), 2)
        for m in masks:
            self.assertGreater(m.sum(), 0)

    def test_crop_to_region(self):
        """After cropping, the shape reflects the new boundaries."""
        self.loader.cropToRegion(0, 0, 50, 30)
        self.assertEqual(self.loader.getShape(), (50, 30))

    def test_boxes_to_ellipses(self):
        """Ellipse conversion keeps masks as uint8 with the same shape."""
        self.loader.boxesToEllipses(alsoPolys=True)
        for name in self.loader.getFileNames():
            mask = self.loader.getMaskByFileName(name)
            self.assertEqual(mask.dtype, np.uint8)


class TestBoxIoU(unittest.TestCase):

    def test_identical_boxes(self):
        """Two identical boxes have IoU of exactly 1."""
        self.assertAlmostEqual(boxIoU([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_no_overlap(self):
        """Disjoint boxes have IoU of 0."""
        self.assertEqual(boxIoU([0, 0, 5, 5], [10, 10, 20, 20]), 0.0)

    def test_partial_overlap(self):
        """A 5x5 overlap of two 10x10 boxes offset by 5 gives 25/175."""
        iou = boxIoU([0, 0, 10, 10], [5, 5, 15, 15])
        self.assertAlmostEqual(iou, 25.0 / 175.0)

    def test_vector(self):
        """Vectorized version matches the scalar for both overlap cases."""
        result = boxIoUVector(
            [0, 0, 10, 10], [[0, 0, 10, 10], [50, 50, 60, 60]])
        self.assertAlmostEqual(result[0], 1.0)
        self.assertEqual(result[1], 0.0)


class TestCalcCliques(unittest.TestCase):

    def test_no_edges(self):
        """Without edges every node is its own clique."""
        cliques = calcCliques([], [0, 1, 2])
        self.assertEqual(sorted(sorted(c) for c in cliques), [[0], [1], [2]])

    def test_one_group(self):
        """A chain of edges merges all nodes into a single clique."""
        cliques = calcCliques([(0, 1), (1, 2)], [0, 1, 2])
        self.assertEqual(len(cliques), 1)
        self.assertEqual(sorted(cliques[0]), [0, 1, 2])

    def test_mixed(self):
        """One connected pair plus an isolated node gives two cliques."""
        cliques = calcCliques([(0, 1)], [0, 1, 2])
        self.assertEqual(len(cliques), 2)


class TestMergeableBoxCliquesByIoU(unittest.TestCase):

    def test_overlapping(self):
        """Two overlapping boxes merge; a distant one stays separate."""
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
        cliques = mergeableBoxCliquesByIoU(boxes, iouThreshold=0.5)
        self.assertEqual(len(cliques), 2)

    def test_no_overlap(self):
        """Well-separated boxes each form their own clique."""
        boxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
        cliques = mergeableBoxCliquesByIoU(boxes, iouThreshold=0.5)
        self.assertEqual(len(cliques), 2)


class TestResizeBoxSeparate(unittest.TestCase):

    def test_identity(self):
        """Factors of 1.0 leave the box unchanged."""
        result = resizeBoxSeparate(0, 0, 10, 10, 1.0, 1.0)
        self.assertEqual(result, (0, 0, 10, 10))

    def test_shrink(self):
        """Halving both sides of a 10x10 box centered at 5,5 gives [2,2,7,7]."""
        result = resizeBoxSeparate(0, 0, 10, 10, 0.5, 0.5)
        self.assertEqual(result, (2, 2, 7, 7))


class TestDrawEllipseFromBox(unittest.TestCase):

    def test_returns_mask_and_poly(self):
        """The output mask has set pixels and the polygon has >= 7 points."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        newMask, poly = drawEllipseFromBox(mask, [5, 5, 15, 15])
        self.assertEqual(newMask.shape, mask.shape)
        self.assertGreater(newMask.sum(), 0)
        self.assertGreaterEqual(len(poly), 7)


class TestMaskImgFromPolygons(unittest.TestCase):

    def test_single_polygon(self):
        """A square polygon produces a non-empty mask with max value 1."""
        poly = [[10, 10], [20, 10], [20, 20], [10, 20]]
        mask = maskImgFromPolygons([poly], 30, 30)
        self.assertGreater(mask.sum(), 0)
        self.assertEqual(mask.max(), 1)

    def test_empty(self):
        """No polygons means an all-zero mask."""
        mask = maskImgFromPolygons([], 10, 10)
        self.assertEqual(mask.sum(), 0)


class TestScale(unittest.TestCase):

    def test_normal_range(self):
        """Mapping [0, 10] to [-1, 1] puts the endpoints at -1 and 1."""
        x = np.array([0.0, 5.0, 10.0])
        result = scale(x, (-1, 1))
        self.assertAlmostEqual(result[0], -1.0)
        self.assertAlmostEqual(result[2], 1.0)

    def test_constant(self):
        """A constant array passes through unchanged (no division by zero)."""
        x = np.array([5.0, 5.0, 5.0])
        result = scale(x)
        np.testing.assert_array_equal(result, x)


class TestFileMissing(unittest.TestCase):

    def test_missing_file(self):
        """A nonexistent path prints an error and leaves the loader empty."""
        loader = CVATLoaderUtil("/nonexistent/path.xml")
        self.assertFalse(loader.hasData())
        self.assertEqual(len(loader), 0)


if __name__ == "__main__":
    unittest.main()
