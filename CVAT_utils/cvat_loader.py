"""
required libraries:
    * matplotlib (pip install matplotlib)
    * numpy (pip install numpy)
    * torch (https://pytorch.org/get-started/locally/)
    * xmltodict (pip install xmltodict)
    * PIL (pip install Pillow)
"""

import numpy as np
import os.path as osp
import xmltodict
import collections
import torch
import sys
from matplotlib.path import Path
from PIL import Image, ImageDraw

"""
Important functions:
    • getFileNames(): return all file names mentioned in the annotations
    • getBoxesByFileName(fileName): return boxes for a certain fileName
    • getMaskByFileName(fileName): return the pixel mask for a certain fileName
"""

class CVATLoaderUtil:
    def __init__(self, xmlPath):
        self.xmlPath = xmlPath
        self.height = 0
        self.width = 0
        self.minId = -1
        self.maxId = -1
        self.annotationData = collections.OrderedDict()  # fileName -> [id, polygons, boxes, mask]
        self.scaleTo = []

        if not osp.exists(xmlPath):
            print("Error: Cannot find {}".format(xmlPath))
            return

        self.loadAnnotations()


    def convertToOtherWindowSize(self, newWindowSize):
        if newWindowSize == 100:
            return

        self.stretch = newWindowSize / 100
        self.convertAnnotationLengthsByFactor()
        
        def convertAnnotationLengthsByFactor(self):        
        iouThreshold = 0.1
        distTolerance = 10
        minLenTraces = 15
         
        if self.stretch <= 0:
            raise ValueError("window size factors must be values > 0!")

        factor = self.stretch
        fileNames = list(self.annotationData.keys())

        # calculate traces of the current annotations
        bbt = self.boxTracerFromAnnotations(fileNames, minLenTraces, iouThreshold, distTolerance)
        oldTraces = bbt.getCalculatedTraces(withFillUps=True, onlyCompleted=True, alsoInvalidated=False)
        
        annot = copy.deepcopy(self.annotationData)

        # fileName -> [id, polygons, boxes, mask]
        self.annotationData = collections.OrderedDict()
        for fileName in fileNames:
            self.annotationData[fileName] = [id, [], [], np.zeros((self.width, self.height), dtype=np.uint8)]

        for framesOld, boxesOld in oldTraces:          
            assert len(boxesOld) > 0

            # old meta info for the whole trace
            startOld = min(framesOld)
            endOld = max(framesOld)
            spanOld = endOld - startOld     
            centerOld = int((startOld + endOld) * 0.5)
            
            # new meta info for the whole trace
            centerNew = centerOld
            spanNew = int(spanOld * factor + 0.5)
            startNew = centerNew - math.ceil(spanNew * 0.5)
            endNew = centerNew + math.floor(spanNew * 0.5)

            centerNew = max(0, min(len(fileNames)-1, centerNew))
            startNew = max(0, min(len(fileNames)-1, startNew))
            endNew = max(0, min(len(fileNames)-1, endNew))

            for frameNew in range(startNew, endNew):
                relPosition = float(frameNew - startNew) / spanNew
                newName = fileNames[frameNew]                
                frameOld = int(startOld * (1.0 - relPosition) + endOld * relPosition)
                frameOldInner = frameOld - startOld
                oldName = fileNames[frameOld]

                # smoothen annotations
                colFrom = max(0, frameOldInner - 2)
                colTo = min(spanOld, frameOldInner + 2)
                colBoxes = [boxesOld[frcb] for frcb in range(colFrom, colTo+1)]
                colBox = [
                            statistics.mean([c[0] for c in colBoxes]),
                            statistics.mean([c[1] for c in colBoxes]),
                            statistics.mean([c[2] for c in colBoxes]),
                            statistics.mean([c[3] for c in colBoxes]),
                         ]

                # elements: id, polygons, boxes, mask
                self.annotationData[newName][2].append(colBox)

        self.boxesToEllipses(alsoPolys=True)
        
        fileNames = list(self.annotationData.keys())
        for fileName in fileNames:
            polys = self.annotationData[fileName][1]
            boxes = self.annotationData[fileName][2]
            boxesNew = []
            polysNew = []

            # take the first box and poly of each clique of boxes
            # determined by their iou values
            cliques = mergeableBoxCliquesByIoU(boxes, iouThreshold=iouThreshold)            
            for clique in cliques:
                boxesNew.append(boxes[clique[0]])
                polysNew.append(polys[clique[0]])                

            self.annotationData[fileName][1] = polysNew
            self.annotationData[fileName][2] = boxesNew
            
            
    def boxesToEllipses(self, alsoPolys=False, sizeFactors=(1,1)):
        for fileName in self.annotationData.keys():
            [id, polygons, boxes, mask] = self.annotationData[fileName]
            if len(polygons)==0 and len(boxes)>0:
                polygons = [ [] for _ in boxes]

            mergedMask = np.zeros_like(mask)
            modPolygons = []

            for poly, box in zip(polygons, boxes):
                box = [box[1], box[0], box[3], box[2]]
                if len(poly)==0 or (len(poly) in list(range(3,8))) or alsoPolys:
                    # polygon representing a box
                    mergedMask, polyEl = drawEllipseFromBox(mergedMask, box, sizeFactors)
                    modPolygons.append(polyEl)
                else:
                    # not representing a box
                    mask = maskImgFromPolygons([poly], mergedMask.shape[0], mergedMask.shape[1])
                    mergedMask = ((mask + mergedMask) > 0).astype(np.uint8)
                    modPolygons.append(poly)
    
            mergedMask = mergedMask.transpose(1,0)
            self.annotationData[fileName] = [id, modPolygons, boxes, mergedMask]

    def getFileNames(self):
        return list(self.annotationData.keys())

    def getFirstFileName(self):
        names = sorted(self.annotationData.keys())
        if len(names) > 0:
            return names[0]

    def getLastFileName(self):
        names = sorted(self.annotationData.keys())
        if len(names) > 0:
            return names[-1]

    def hasData(self):
        return len(self.annotationData) > 0

    def hasEntryForFileName(self, fileName):
        return fileName in self.annotationData

    def getPolygonsByImgName(self, fileName):
        polys = self.annotationData[fileName][1]
        polys = [[ [pt[1], pt[0]] for pt in poly] for poly in polys]
        return polys

    def getBoxesByImgName(self, fileName):
        polys = self.getPolygonsByImgName(fileName)
        boxes = []

        for poly in polys:
            x_min = min([p[0] for p in poly])
            x_max = max([p[0] for p in poly])
            y_min = min([p[1] for p in poly])
            y_max = max([p[1] for p in poly])
            boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.tensor(boxes)
        return boxes

    def getShape(self, fileName):
        return (self.width, self.height)

    def getBoxesByFileName(self, fileName):
        return self.annotationData[fileName][2]

    def getMaskByFileName(self, fileName):
        return self.annotationData[fileName][3]

    def loadAnnotations(self):
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
                            # store this point as a list
                            points.append(list(map(lambda e: int(float(e)), pointString.split(','))))

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

            mask = (maskImgFromPolygons(polygons, self.width, self.height) > sys.float_info.epsilon).astype(np.uint8)
            mask = mask.transpose(1,0)
            mask = torch.from_numpy(mask)

            assert len(polygons) == len(boxes)
            self.annotationData[fileName] = [id, polygons, boxes, mask]

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
    polygon = [list(e) for e in polygon]
    polygon_x = [e[0] for e in polygon]
    polygon_y = [e[1] for e in polygon]
    polygon = list(zip(polygon_x, polygon_y))

    img = Image.new('L', (height, width), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img)

    if heightFirst:
        mask = mask.transpose(1, 0)

    return mask

    #if len(polygon) == 0:
    #    if heightFirst:
    #        return np.zeros(shape=(height, width), dtype=np.uint8)
    #    else:
    #        return np.zeros(shape=(width, height), dtype=np.uint8)

    #poly_path = Path(polygon)
    #x, y = np.mgrid[:height, :width]
    #coords = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    #mask = poly_path.contains_points(coords)
    #mask = mask.reshape(height, width)

    #mask = mask.astype(np.uint8)
    #mask *= pixelValue

    #if not heightFirst:
    #    mask = mask.transpose(1, 0)

    #return mask


def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)

    if (domain[1] - domain[0]) == 0:
        return x

    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
