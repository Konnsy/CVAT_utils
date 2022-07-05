import numpy as np
import os.path as osp
import xmltodict
import collections
from matplotlib.path import Path
import errno
import os

"""
Important functions:
    • getFileNames(): return all file names mentioned in the annotations
    • getBoxesByFileName(fileName): return boxes for a certain fileName
    • getMaskByFileName(fileName): return the pixel mask for a certain fileName
"""
class CVATLoader:
    def __init__(self, xmlPath):
        self.xmlPath = xmlPath
        self.height = 0
        self.width = 0
        self.minId = -1
        self.maxId = -1
        self.annotationData = collections.OrderedDict()  # fileName -> [id, polygons, boxes, mask]

        if not osp.exists(xmlPath):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), xmlPath)

        self.loadAnnotations()

    def getFileNames(self):
        return list(self.annotationData.keys())

    def hasData(self):
        return len(self.annotationData) > 0

    def hasEntryForFileName(self, fileName):
        return fileName in self.annotationData

    def getPolygonsByImgName(self, fileName):
        return self.annotationData[fileName][1]

    def getShape(self, fileName):
        return (self.width, self.height)

    def getBoxesByFileName(self, fileName):
        return self.annotationData[fileName][2]

    def getMaskByFileName(self, fileName):
        return self.annotationData[fileName][3]
        
    def loadAnnotations(self):
        try:
            with open(self.xmlPath) as fd:
                doc = xmltodict.parse(fd.read())

            imgInfo = []

            for imgEntry in doc['annotations']['image']:
                self.height = int(imgEntry['@height'])
                self.width = int(imgEntry['@width'])
                fileName = imgEntry['@name']
                id = int(imgEntry['@id'])

                if self.minId == -1:
                    self.minId = id
                    self.maxId = id
                else:
                    self.minId = min(id, self.minId)
                    self.maxId = max(id, self.maxId)

                polygons = []
                if 'polygon' in imgEntry:
                    polyEntries = imgEntry['polygon']

                    if '@label' in polyEntries:
                        pointsString = polyEntries['@points']
                        points = []

                        for pointString in pointsString.split(';'):
                            # store this point as a list of float values
                            points.append(list(map(lambda e: float(e), pointString.split(','))))

                        polygons.append(points)
                    else:
                        for polyEntry in polyEntries:
                            pointsString = polyEntry['@points']
                            points = []

                            for pointString in pointsString.split(';'):
                                # store this point as a list of float values
                                points.append(list(map(lambda e: float(e), pointString.split(','))))

                            polygons.append(points)
                        polygons.append(points)

                boxes = []

                if len(polygons) > 0:
                    for poly in polygons:
                        xPos = list(map(lambda points: points[0], poly))
                        xPos.sort()
                        yPos = list(map(lambda points: points[1], poly))
                        yPos.sort()
                        xMin, xMax = xPos[0], xPos[-1]
                        yMin, yMax = yPos[0], yPos[-1]
                        boxes.append([xMin, yMin, xMax, yMax])

                mask = maskImgFromPolygons(polygons, self.width, self.height)

                assert len(polygons) == len(boxes)
                self.annotationData[fileName] = [id, polygons, boxes, mask]

        except:
            print("Error: Exception during loading of {}".format(self.xmlPath))
            return

        if len(self.annotationData) == 0:
            print("Warning: There was no exception, but no fitting annotations were found in {}".format(self.xmlPath))


def maskImgFromPolygons(polygons, im_width, im_height):
    # sequential conversion to masks
    masks = []
    for iPoly, polygon in enumerate(polygons):
        masks.append(maskFromPolygon((polygon, im_width, im_height), pixelValue=iPoly + 1))

    # merge the masks collected this frame
    mergedMask = np.zeros(shape=(im_width, im_height), dtype=np.uint16)
    for mask in masks:
        # mergedMask = np.logical_or(mergedMask, mask)
        cover = np.equal(mergedMask, 0)
        cover = cover.astype(np.uint16)
        mask = np.multiply(mask, cover)  # blank out mask values at places already covered in the mergedMask
        mergedMask = np.add(mergedMask, mask)

    mergedMask = mergedMask.astype(np.uint16)
    return mergedMask


def maskFromPolygon(data, pixelValue=1, heightFirst=True):
    polygon, height, width = data

    polygon = list(map(lambda e: list(e), polygon))

    if len(polygon) == 0:
        if heightFirst:
            return np.zeros(shape=(height, width), dtype=np.uint8)
        else:
            return np.zeros(shape=(width, height), dtype=np.uint8)

    poly_path = Path(polygon)
    x, y = np.mgrid[:height, :width]
    coords = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    mask = poly_path.contains_points(coords)
    mask = mask.reshape(height, width)

    mask = mask.astype(np.uint8)
    mask *= pixelValue

    if not heightFirst:
        mask = mask.transpose(1, 0)

    return mask
