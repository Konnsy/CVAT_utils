# -*- coding: utf-8 -*-
"""
Standalone CVAT annotation loader.

Required libraries:
    * numpy (pip install numpy)
    * xmltodict (pip install xmltodict)
    * PIL (pip install Pillow)
"""

import collections
import math
import os.path as osp
import statistics
import sys

import numpy as np
import xmltodict
from PIL import Image, ImageDraw


class CVATLoaderUtil:
    """Loads a CVAT XML export and provides per-frame polygons, boxes, and masks."""

    def __init__(self, xmlPath: str):
        """Parses the CVAT XML file and stores annotation data per frame."""
        self.xmlPath = xmlPath
        self.height = 0
        self.width = 0
        self.minId = -1
        self.maxId = -1
        self.owner = ""
        self.annotationData: collections.OrderedDict = collections.OrderedDict()

        if not osp.exists(xmlPath):
            print(f"Error: Cannot find {xmlPath}")
            return

        self.loadAnnotations()
        self.removeDuplicates()

    def loadAnnotations(self) -> None:
        """Reads polygons, boxes, and masks from the CVAT XML into memory."""
        try:
            with open(self.xmlPath, encoding="utf-8") as fd:
                doc = xmltodict.parse(fd.read())

            self.owner = doc["annotations"]["meta"]["task"]["owner"]["username"]

            for imgEntry in doc["annotations"]["image"]:
                self.height = int(imgEntry["@height"])
                self.width = int(imgEntry["@width"])
                fileName = imgEntry["@name"]
                imgId = int(imgEntry["@id"])

                if self.minId == -1:
                    self.minId = imgId
                    self.maxId = imgId
                else:
                    self.minId = min(imgId, self.minId)
                    self.maxId = max(imgId, self.maxId)

                polygons = []
                if "polygon" in imgEntry:
                    polyEntries = imgEntry["polygon"]
                    if "@label" in polyEntries:
                        polygons.append(self._parsePoints(
                            polyEntries["@points"]))
                    else:
                        for polyEntry in polyEntries:
                            polygons.append(
                                self._parsePoints(polyEntry["@points"]))

                boxes = []
                for poly in polygons:
                    xPos = sorted(pt[0] for pt in poly)
                    yPos = sorted(pt[1] for pt in poly)
                    boxes.append([xPos[0], yPos[0], xPos[-1], yPos[-1]])

                mask = maskImgFromPolygons(
                    polygons, self.width, self.height)

                assert len(polygons) == len(boxes)
                self.annotationData[fileName] = [imgId, polygons, boxes, mask]

        except Exception as exc:
            print(f"Error: Exception during loading of {self.xmlPath}: {exc}")
            return

        if len(self.annotationData) == 0:
            print(
                f"Warning: No fitting annotations found in {self.xmlPath}")

    def _parsePoints(self, pointsString: str) -> list:
        """Splits a semicolon-separated point string into a list of [x, y] floats."""
        return [
            [float(e) for e in pt.split(",")]
            for pt in pointsString.split(";")
        ]

    def removeDuplicates(self, iouThreshold: float = 0.5) -> None:
        """Drops boxes that overlap a neighbor beyond the IoU threshold."""
        for fileName in list(self.annotationData.keys()):
            entry = self.annotationData[fileName]
            polygons, boxes = entry[1], entry[2]

            pairs = []
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if boxIoU(boxes[i], boxes[j]) >= iouThreshold:
                        pairs.append((i, j))

            cliques = calcCliques(pairs, list(range(len(boxes))))
            toRemove = set()
            for clique in cliques:
                if len(clique) > 1:
                    toRemove.update(clique[1:])

            for idx in sorted(toRemove, reverse=True):
                del polygons[idx]
                del boxes[idx]

    def cropToRegion(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Crops all annotations to a rectangle, rebuilding masks at the new size."""
        if x2 == -1:
            x2 = self.width
        if y2 == -1:
            y2 = self.height

        newAnnot: collections.OrderedDict = collections.OrderedDict()
        for key, dat in self.annotationData.items():
            polysNew, boxesNew = [], []
            for polygon, box in zip(dat[1], dat[2]):
                boxNew = [
                    min(box[0], x2), min(box[1], y2),
                    min(box[2], x2), min(box[3], y2),
                ]
                boxNew = [
                    max(0, boxNew[0] - x1), max(0, boxNew[1] - y1),
                    max(0, boxNew[2] - x1), max(0, boxNew[3] - y1),
                ]
                area = (box[2] - box[0]) * (box[3] - box[1])
                areaCrop = (boxNew[2] - boxNew[0]) * (boxNew[3] - boxNew[1])
                if areaCrop < area * 0.5 or areaCrop < 10:
                    continue
                boxesNew.append(boxNew)

                polyNew = [
                    [max(0, min(pt[0], x2) - x1),
                     max(0, min(pt[1], y2) - y1)]
                    for pt in polygon
                ]
                polysNew.append(polyNew)

            mergedMask = np.zeros(
                (self.width, self.height), dtype=np.uint8)
            for poly in polysNew:
                m = maskImgFromPolygons([poly], self.width, self.height)
                mergedMask = ((m + mergedMask) > 0).astype(np.uint8)

            newAnnot[key] = [dat[0], polysNew, boxesNew, mergedMask]

        self.width = x2 - x1
        self.height = y2 - y1
        self.annotationData = newAnnot

    def boxesToEllipses(
        self, alsoPolys: bool = False, sizeFactors: tuple = (1, 1)
    ) -> None:
        """Replaces box-shaped polygons with fitted ellipses and rebuilds masks."""
        for fileName in self.annotationData.keys():
            imgId, polygons, boxes, mask = self.annotationData[fileName]
            if len(polygons) == 0 and len(boxes) > 0:
                polygons = [[] for _ in boxes]

            mergedMask = np.zeros_like(mask)
            modPolygons = []

            for poly, box in zip(polygons, boxes):
                box = [box[1], box[0], box[3], box[2]]
                if (len(poly) == 0 or 3 <= len(poly) < 8 or alsoPolys):
                    mergedMask, polyEl = drawEllipseFromBox(
                        mergedMask, box, sizeFactors)
                    modPolygons.append(polyEl)
                else:
                    m = maskImgFromPolygons(
                        [poly], mergedMask.shape[0], mergedMask.shape[1])
                    mergedMask = ((m + mergedMask) > 0).astype(np.uint8)
                    modPolygons.append(poly)

            mergedMask = mergedMask.astype(np.uint8)
            self.annotationData[fileName] = [
                imgId, modPolygons, boxes, mergedMask]

    def getAnnotationArea(self, intervalBorder: float = 1.0) -> tuple:
        """Returns median, min, and max area covered by annotation polygons."""
        assert 0.0 <= intervalBorder <= 1.0
        areas = []
        for fileName in self.annotationData.keys():
            maskShape = self.annotationData[fileName][3].shape
            for poly in self.annotationData[fileName][1]:
                m = maskImgFromPolygons(
                    [poly], maskShape[0], maskShape[1])
                areas.append(int((m > 0).sum()))

        areas = [a for a in areas if a > 0]
        if not areas:
            return 0, 0, 0
        med = statistics.median(areas)
        if intervalBorder < 1.0:
            lo = int(np.percentile(areas, int(intervalBorder * 100)))
            hi = int(np.percentile(areas, int((1 - intervalBorder) * 100)))
        else:
            lo, hi = min(areas), max(areas)
        return med, lo, hi

    def getSideFactors(self, intervalBorder: float = 1.0) -> tuple:
        """Returns median, min, and max aspect ratio of polygon bounding boxes."""
        assert 0.0 <= intervalBorder <= 1.0
        factors = []
        for fileName in self.annotationData.keys():
            for poly in self.annotationData[fileName][1]:
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                side1 = max(xs) - min(xs)
                side2 = max(ys) - min(ys)
                if side1 > side2:
                    factors.append(side1 / (side2 + sys.float_info.epsilon))
                else:
                    factors.append(side2 / (side1 + sys.float_info.epsilon))

        if not factors:
            return 0, 0, 0
        med = statistics.median(factors)
        if intervalBorder < 1.0:
            lo = float(np.percentile(factors, int(intervalBorder * 100)))
            hi = float(np.percentile(factors, int((1 - intervalBorder) * 100)))
        else:
            lo, hi = min(factors), max(factors)
        return med, lo, hi

    def getPolygonsPerFrame(self) -> list:
        """Returns a list of polygon lists, one per annotated frame."""
        return [
            self.annotationData[name][1]
            for name in self.annotationData.keys()
        ]

    def getBoxesPerFrame(self) -> list:
        """Returns a list of box lists, one per annotated frame."""
        return [
            self.annotationData[name][2]
            for name in self.annotationData.keys()
        ]

    def getFileNames(self) -> list:
        """Returns the list of frame names in annotation order."""
        return list(self.annotationData.keys())

    def getFirstFileName(self) -> str:
        """Returns the alphabetically first frame name."""
        names = sorted(self.annotationData.keys())
        if names:
            return names[0]
        return ""

    def getLastFileName(self) -> str:
        """Returns the alphabetically last frame name."""
        names = sorted(self.annotationData.keys())
        if names:
            return names[-1]
        return ""

    def hasData(self) -> bool:
        """True if at least one frame has annotation data."""
        return len(self.annotationData) > 0

    def hasEntryForFileName(self, fileName: str) -> bool:
        """True if the given frame exists in the annotation set."""
        return fileName in self.annotationData

    def getPolygonsByImgName(self, fileName: str) -> list:
        """Returns stored polygon vertices for a frame as-is."""
        return self.annotationData[fileName][1]

    def getBoxesByImgName(self, fileName: str) -> list:
        """Returns stored bounding boxes for a frame."""
        return self.annotationData[fileName][2]

    def getShape(self) -> tuple:
        """Returns (width, height) of the annotated frames."""
        return (self.width, self.height)

    def getBoxesByFileName(self, fileName: str) -> list:
        """Returns stored boxes for a frame, or an empty list if missing."""
        if self.hasEntryForFileName(fileName):
            return self.annotationData[fileName][2]
        return []

    def getMaskByFileName(self, fileName: str) -> np.ndarray:
        """Returns the pixel mask for a frame, or a zero mask if missing."""
        if self.hasEntryForFileName(fileName):
            return self.annotationData[fileName][3]
        return np.zeros((self.width, self.height), dtype=np.uint8)

    def getSeparateMasksByFileName(self, fileName: str) -> list:
        """Returns one binary mask per polygon in a frame."""
        masks = []
        for poly in self.annotationData[fileName][1]:
            masks.append(maskImgFromPolygons(
                [poly], self.width, self.height))
        return masks

    def getMaskByIdx(self, idx: int) -> np.ndarray:
        """Returns the mask for the frame at the given sorted index."""
        keys = sorted(self.annotationData.keys())
        if len(keys) > idx:
            fileName = keys[idx]
            if self.hasEntryForFileName(fileName):
                return self.annotationData[fileName][3]
        return np.zeros((self.width, self.height), dtype=np.uint8)

    def firstAnnotatedFrameName(self) -> str:
        """Returns the first frame that actually contains polygon data."""
        for name in self.annotationData.keys():
            if len(self.annotationData[name][1]) > 0:
                return name
        return ""

    def deleteEntry(self, fileName: str) -> None:
        """Removes a frame from the annotation set."""
        if fileName in self.annotationData:
            del self.annotationData[fileName]

    def __len__(self) -> int:
        """Number of annotated frames."""
        return len(self.annotationData)


def maskImgFromPolygons(
    polygons: list, im_width: int, im_height: int
) -> np.ndarray:
    """Builds a merged binary mask from a list of polygon vertex lists."""
    masks = []
    for iPoly, polygon in enumerate(polygons):
        masks.append(
            maskFromPolygon((polygon, im_width, im_height),
                            pixelValue=iPoly + 1))

    mergedMask = np.zeros((im_width, im_height), dtype=np.uint16)
    for mask in masks:
        cover = (mergedMask == 0).astype(np.uint16)
        mask = np.multiply(mask, cover)
        mergedMask = np.add(mergedMask, mask)
    return mergedMask.astype(np.uint16)


def maskFromPolygon(
    data: tuple, pixelValue: int = 1, heightFirst: bool = True
) -> np.ndarray:
    """Renders a single polygon into a binary image array via PIL."""
    polygon, height, width = data
    polygon = [list(e) for e in polygon]
    xs = [e[0] for e in polygon]
    ys = [e[1] for e in polygon]
    polygon = list(zip(xs, ys))

    img = Image.new("L", (height, width), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img)

    if heightFirst:
        mask = mask.transpose(1, 0)
    return mask


def drawEllipseFromBox(
    mask: np.ndarray, box: list, sizeFactors: tuple = (1.0, 1.0)
) -> tuple:
    """Draws a filled ellipse for a box into the mask; returns (mask, polygon)."""
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = resizeBoxSeparate(
        x1, y1, x2, y2, sizeFactors[0], sizeFactors[1])
    center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    boxSides = (x2 - x1, y2 - y1)

    img = Image.fromarray((mask * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.ellipse([int(x1), int(y1), int(x2), int(y2)], fill=255)
    arr = np.array(img)
    arr = (arr > 0).astype(np.uint8)

    steps = 7
    poly = []
    for iAngle in range(steps + 1):
        angle = (2 * math.pi / steps) * iAngle
        px = (boxSides[1] * 0.5) * math.cos(angle) + center[1]
        py = (boxSides[0] * 0.5) * math.sin(angle) + center[0]
        poly.append([px, py])

    return arr, poly


def resizeBoxSeparate(
    x1: float, y1: float, x2: float, y2: float,
    factorX: float = 1.0, factorY: float = 1.0
) -> tuple:
    """Scales a box around its center by separate x and y factors."""
    xd = x2 - x1
    yd = y2 - y1
    mx = x1 + xd * 0.5
    my = y1 + yd * 0.5
    x1 = int(mx - xd * 0.5 * factorX)
    x2 = int(mx + xd * 0.5 * factorX)
    y1 = int(my - yd * 0.5 * factorY)
    y2 = int(my + yd * 0.5 * factorY)
    return x1, y1, x2, y2


def boxIoU(boxA: list, boxB: list) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    dx = min(boxA[2], boxB[2]) - max(boxA[0], boxB[0])
    dy = min(boxA[3], boxB[3]) - max(boxA[1], boxB[1])
    if dx < 0 or dy < 0:
        return 0.0
    interArea = dx * dy
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)


def boxIoUVector(box: list, boxes: list) -> np.ndarray:
    """IoU of one box against many, as a numpy array."""
    if len(boxes) == 0:
        return np.array([], dtype=float)
    box = np.asarray(box, dtype=float)
    boxes = np.asarray(boxes, dtype=float)
    dx = np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])
    dy = np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1])
    inter = np.where((dx < 0) | (dy < 0), 0.0, dx * dy)
    areaA = (box[2] - box[0]) * (box[3] - box[1])
    areaB = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (areaA + areaB - inter)


def calcCliques(edges: list, nodes: list) -> list:
    """Groups connected nodes into cliques via iterative merging."""
    nodes = sorted(nodes)
    con: dict = {}
    for a, b in edges:
        if a in con:
            con[a].add(b)
        else:
            con[a] = {b}

    changed = True
    while changed:
        changed = False
        for idx1 in range(len(nodes) - 1, 0, -1):
            for idx2 in range(min(len(nodes), idx1)):
                if (nodes[idx1] in con and nodes[idx2] in con
                        and nodes[idx1] in con[nodes[idx2]]):
                    con[nodes[idx2]].update(con[nodes[idx1]])
                    del con[nodes[idx1]]
                    changed = True

    inClique = set()
    for valList in con.values():
        inClique.update(valList)
    inClique.update(con.keys())

    cliques = []
    for node in nodes:
        if node in con:
            cliques.append([node] + list(con[node]))
        elif node not in inClique:
            cliques.append([node])
    return cliques


def mergeableBoxCliquesByIoU(
    boxes: list, iouThreshold: float
) -> list:
    """Finds groups of boxes that overlap above the IoU threshold."""
    pairs = []
    for i in range(len(boxes)):
        ious = boxIoUVector(boxes[i], boxes[i + 1:])
        for offset in (ious >= iouThreshold).nonzero()[0]:
            pairs.append((i, i + 1 + int(offset)))
    return calcCliques(pairs, list(range(len(boxes))))


def scale(x: np.ndarray, out_range: tuple = (-1, 1)) -> np.ndarray:
    """Normalizes an array to the given output range."""
    lo, hi = np.min(x), np.max(x)
    if (hi - lo) == 0:
        return x
    y = (x - (hi + lo) / 2) / (hi - lo)
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
