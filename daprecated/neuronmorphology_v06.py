import json
import numpy as np
from convertAllenSpace import convertAllenSpace


class NeuronMorphology:
    metaData = None
    customTypes = None
    customProps = None
    lines = None
    points = None
    _objectProps = None

    # IMPORTANT: unitOrientationOrigin assumes that neuron is in Allen Space.
    # Extension to generic space to be made when needed.
    def __init__(self, neuronDict, unitOrientationOrigin):
        self.unitOrientationOrigin = unitOrientationOrigin
        self.metaData = neuronDict["metaData"]
        self.customTypes = neuronDict["customTypes"]
        self.customProps = neuronDict["customProperties"]
        self.lines = neuronDict["treeLines"]["data"]
        self.lineColumns = neuronDict["treeLines"]["columns"]
        self.points = np.array(neuronDict["treePoints"]["data"])
        self.pointColumns = neuronDict["treePoints"]["columns"]

    def getType2id(self):
        type2id = {"soma": 1, "axon": 2, "dendrite": 3, "apical": 4}
        for geom, types in self.customTypes.items():
            for tp in types:
                type2id[tp.name] = tp.id
        return type2id

    def initObjectProperties(self):
        propertyAssignments = self.customProps["for"]
        objectProperties = {}
        for index, assignment in enumerate(propertyAssignments):
            props = assignment["set"]
            if "objects" in assignment:
                for objId in assignment["objects"]:
                    if not objId in objectProperties:
                        objectProperties[objId] = props.copy()
                    else:
                        objectProperties[objId].update(props)
        self._objectProps = objectProperties

    def getObjectProperties(self):
        if self._objectProps is None:
            self.initObjectProperties()
        return self._objectProps

    # note: this removes point properties, if any
    def setObjectProperties(self, objectProperties):
        self._objectProps = objectProperties
        idLists = {}
        for objectId, props in objectProperties.items():
            for k, v in props.items():
                propKV = json.dumps([k, v])
                if propKV not in idLists:
                    idLists[propKV] = []
                idLists[propKV].append(objectId)

        propLists = {}
        for propKV, ids in idLists.items():
            idsKey = json.dumps(ids)
            if idsKey not in propLists:
                propLists[idsKey] = []
            propLists[idsKey].append(propKV)

        propertyAssignments = []
        for idsKey, propKVs in propLists.items():
            ids = json.loads(idsKey)
            propertyAssignment = {"objects": ids}
            propertyAssignment["set"] = {}
            for propKV in propKVs:
                [k, v] = json.loads(propKV)
                propertyAssignment["set"][k] = v  # set key-value property

            propertyAssignments.append(propertyAssignment)

        if len(propertyAssignments):
            self.customProps["for"] = propertyAssignments

    def getLine2children(self, mixTypes=False):
        line2children = {}
        for lineIdx, line in enumerate(self.lines):
            (tp, firstPoint, numPoints, parentLineIdx, negOffset) = line
            if lineIdx != parentLineIdx:
                parentLine = self.lines[parentLineIdx]
                if mixTypes or tp == parentLine[0]:
                    if not parentLineIdx in line2children:
                        line2children[parentLineIdx] = []
                    line2children[parentLineIdx].append(lineIdx)
        return line2children

    def _propagateDistance(
        self, distances, branchOrders, lineLengths, parentLineIdx, line2children
    ):
        if parentLineIdx not in line2children:
            return
        parentLine = self.lines[parentLineIdx]
        prevPointIdx = (
            parentLine[1] + parentLine[2] - 1 - parentLine[4]
        )  # firstPoint+nPoints-1-negOffset
        lineIndices = line2children[parentLineIdx]
        branchOrder = branchOrders[prevPointIdx]
        if len(lineIndices) > 1:
            branchOrder += 1
        for lineIdx in lineIndices:
            (tp, firstPointIdx, nPoints, parentLineIdx, negOffset) = self.lines[
                lineIdx
            ]
            if nPoints == 0:
                continue
            prevPoint = self.points[prevPointIdx][0:3]
            dst = distances[prevPointIdx]
            for p in range(firstPointIdx, firstPointIdx + nPoints):
                point = self.points[p][0:3]
                dst += np.linalg.norm(point - prevPoint)
                distances[p] = dst
                branchOrders[p] = branchOrder
                prevPoint = point

            lineLengths[lineIdx] = (
                distances[firstPointIdx + nPoints - 1] - distances[prevPointIdx]
            )
            if lineLengths[lineIdx] < 0:
                print(
                    "Error, lineLength of line {} is negative: {}".format(
                        lineIdx, lineLengths[lineIdx]
                    )
                )

            self._propagateDistance(
                distances, branchOrders, lineLengths, lineIdx, line2children
            )

    def getSomaLineIdx(self):
        somaType = 1
        somaPointIdx = None
        somaLineIdx = None
        for lineIdx, line in enumerate(self.lines):
            if line[0] == somaType:
                somaPointIdx = line[1]
                somaLineIdx = lineIdx
                break
        return somaLineIdx

    def getPointStatistics(self, somaLineIdx=None):
        if somaLineIdx is None:
            somaLineIdx = self.getSomaLineIdx()
        if somaLineIdx is None:
            somaLineIdx = 0
        somaDistances = np.zeros((len(self.points)))
        branchOrders = np.zeros((len(self.points)))
        lineLengths = np.zeros((len(self.lines)))
        line2children = self.getLine2children(mixTypes=True)
        self._propagateDistance(
            somaDistances, branchOrders, lineLengths, somaLineIdx, line2children
        )

        return somaDistances, branchOrders, lineLengths, somaLineIdx

    def _propagateNearestTerminal(
        self, lineIdx, nearestTerminals, somaDistances, line2children
    ):
        if not nearestTerminals[lineIdx]:
            nearestTerminal = 0
            lowestDistance = None
            if lineIdx in line2children:
                for childIdx in line2children[lineIdx]:
                    terminalIdx = self._propagateNearestTerminal(
                        childIdx, nearestTerminals, somaDistances, line2children
                    )
                    line = self.lines[terminalIdx]
                    distance = somaDistances[line[1] + line[2] - 1]
                    if lowestDistance is None or distance < lowestDistance:
                        lowestDistance = distance
                        nearestTerminal = terminalIdx
            else:
                nearestTerminal = lineIdx
            nearestTerminals[lineIdx] = nearestTerminal
        return nearestTerminals[lineIdx]

    def getNearestTerminals(self, somaDistances, somaLineIdx):
        nearestTerminals = np.zeros((len(self.lines)), np.uint32)
        line2children = self.getLine2children(mixTypes=True)
        self._propagateNearestTerminal(
            somaLineIdx, nearestTerminals, somaDistances, line2children
        )
        return nearestTerminals

    def getReorientedPoint(self, pointIdx, unitOrientationOrigin):
        A = convertAllenSpace(self.unitOrientationOrigin, unitOrientationOrigin)
        point = self.points[pointIdx, :]
        point[3] = 1.0
        return point @ A.T

    def reorient(self, unitOrientationOrigin):
        A = convertAllenSpace(self.unitOrientationOrigin, unitOrientationOrigin)
        warpedPoints = self.points.copy()
        warpedPoints[:, 3] = 1.0
        warpedPoints = warpedPoints @ A.T
        self.unitOrientationOrigin = unitOrientationOrigin

        # use cubic root of determinant of affine matrix to determine radius change
        warpedPoints[:, 3] = np.linalg.det(A) ** (1 / 3) * self.points[:, 3]
        self.points = warpedPoints

    """
    return all isolated pieces of neurite of the given types,
    where (1, 2, 3, 4) refers to soma, axons, dendrite and apical dendrite, respectively
    """

    def getPiecesOfNeurite(self, neuriteTypes=(1, 2, 3, 4)):
        lines = self.lines

        pieces = {}
        for lineId, line in enumerate(lines):
            # only consider points that are either soma-tree (not existent in Neurolucida), axon, dendrite or apical dendrite
            if line[0] in neuriteTypes:
                pieces[lineId] = {}

        for lineId, piece in pieces.items():
            line = lines[lineId]
            parentLineId = line[3]
            if parentLineId:
                parentLine = lines[parentLineId]
                if not parentLineId in pieces:
                    print("Error, expecting parentLineId in pieces.")
                pieces[parentLineId][lineId] = piece
        # now remove all pieces that have a parent
        for lineId, piece in list(pieces.items()):
            line = lines[lineId]
            parentLineId = line[3]
            if parentLineId:
                # has parent
                pieces.pop(lineId)

        return pieces

    def addPoint(self, point):
        self.points = np.vstack((self.points, point))
        return len(self.points) - 1

    def addLine(self, tp, firstPointIdx, numPoints, parentLineId, negOffset=0):
        self.lines.append(
            [tp, firstPointIdx, numPoints, parentLineId, negOffset]
        )
        return len(self.lines) - 1

    def addNeuriteType(self, geom, attrs):
        # create a new custom type for stitches
        customTypes = self.customTypes
        maxTypeId = 0
        for geom, typeDefs in customTypes.items():
            if not isinstance(typeDefs, (list, tuple)):
                typeDefs = [typeDefs]
            for tp in typeDefs:
                if tp["id"] > maxTypeId:
                    maxTypeId = tp["id"]
        newTypeId = maxTypeId + 1
        if geom in customTypes:
            if not isinstance(customTypes[geom], (list, tuple)):
                customTypes[geom] = [customTypes[geom]]
        else:
            self.customTypes[geom] = []
        newType = attrs.copy()
        newType["id"] = newTypeId
        customTypes[geom].append(newType)
        return newTypeId

    """
    Return a copy of the neuron with a subsample of points, keeping the branching structure intact.
    Only include points that are at least minDistance apart from each other (along the neurite).
    """

    def subsampledCopy(self, selectedTypes, minDistance=0):
        type2id = self.getType2id()
        selectedTypeIds = []
        for tp in selectedTypes:
            if isinstance(tp, str):
                # convert to id
                selectedTypeIds.append(type2id[tp])
            else:
                # in case the selectedType is already specified as an integer, just use it
                selectedTypeIds.append(tp)

        distances, branchingOrders, lineLengths, somaIdx = (
            self.getPointStatistics()
        )
        newPoints = [[0, 0, 0, 0]]  # first point is ignored
        newLines = [[0, 0, 0, 0, 0]]  # first line is ignored
        line2new = {0: 0}
        for lineIdx, line in enumerate(self.lines):
            firstIncluded = 0
            numIncluded = 0
            if line[0] in selectedTypeIds:
                parentLineIdx = line[3]
                if parentLineIdx:
                    parentLine = self.lines[parentLineIdx]
                    # first point is the last point of the parent
                    prevPointIdx = (
                        parentLine[1] + parentLine[2] - 1 - parentLine[4]
                    )
                else:
                    # first point is the first point of the line if it has no parent
                    prevPointIdx = line[1]

                dst0 = distances[prevPointIdx]
                firstPointIdx = line[1]
                lastPointIdx = firstPointIdx + line[2] - 1
                for i in range(firstPointIdx, lastPointIdx + 1):
                    # always include last point of line (=terminal or branch point)
                    if distances[i] - dst0 > minDistance or i == lastPointIdx:
                        if not firstIncluded:
                            firstIncluded = len(newPoints)
                        numIncluded += 1
                        newPoints.append(self.points[i, :])
                        dst0 = distances[i]

                line2new[lineIdx] = len(newLines)
                newLines.append(
                    [line[0], firstIncluded, numIncluded, line[3], line[4]]
                )

        # parent line field should use new line indices
        for line in newLines:
            if line[3] in line2new:
                line[3] = line2new[line[3]]
            else:
                line[3] = 0  # no parent

        return NeuronMorphology(
            dict(
                metaData=self.metaData,
                customTypes=self.customTypes,
                customProperties={},  # these are no longer valid
                treeLines=dict(columns=self.lineColumns, data=newLines),
                treePoints=dict(columns=self.pointColumns, data=newPoints),
            ),
            self.unitOrientationOrigin,
        )

    def asDict(self):
        return dict(
            metaData=self.metaData,
            customTypes=self.customTypes,
            customProperties=self.customProps,
            treeLines=dict(columns=self.lineColumns, data=self.lines),
            treePoints=dict(
                columns=self.pointColumns, data=self.points.tolist()
            ),
        )

    # UNTESTED!
    def asSwcDict(self, includeLines=None):
        newLines = self.lines
        # newObjectProperties = {}
        # newPointProperties = {}

        if includeLines:
            # overwrite newLines to include only lineIds from includeLines
            newLines = np.zeros([len(includeLines) + 1, 5], np.uint32)
            newLineIds = {}
            for i, lineId in enumerate(includeLines):
                line = self.lines[lineId]
                newLineId = i + 1
                newLineIds[lineId] = newLineId
                newLines[newLineId, :] = line

            # fix parentLineId
            for newLineId in range(1, newLines.shape[0]):
                newLine = newLines[i, :]
                newLine[3] = (
                    newLineIds[newLine[3]] if newLine[3] in newLineIds else 0
                )

        swcPoints = []
        newPointIds = {}
        newPointId = 1  # start counting at 1
        for lineId in range(1, newLines.shape[0]):
            parentPointId = 0
            tp, firstPoint, numPoints, parentLineId, negOffset = newLines[
                lineId, :
            ]
            if parentLineId > 0:
                parentLine = newLines[parentLineId, :]
                parentPointId = parentLine[1] + parentLine[2] - 1 - negOffset
                # parent firstPoint + numPoints -1 - negative offset

            for pointId in range(firstPoint, firstPoint + numPoints):
                """
                if (pointId in objectProperties) {
                  if (pointId !== firstPoint) console.log('Object property assigned to point '+pointId+', but object starts at '+firstPoint);
                  newObjectProperties[firstPoint] = objectProperties[firstPoint];
                }
                if (pointId in pointProperties) newPointProperties[newPointId] = pointProperties[pointId];
                """
                newPointIds[pointId] = newPointId
                point = self.points[pointId, :]
                swcPoints.push(
                    [
                        newPointId,
                        tp,
                        point[0],
                        point[1],
                        point[2],
                        point[3],
                        parentPointId,
                    ]
                )
                parentPointId = pointId
                newPointId += 1

        # make the switch from parentPointId to newParentPointId
        for point in swcPoints:
            point[6] = newPointIds[point[6]] if point[6] in newPointIds else -1
        # TODO: FIX customProperties, for now make them empty
        # const customProperties = tree_class.compressProperties(newObjectProperties,newPointProperties);
        customProperties = {}
        return dict(points=swcPoints, customProperties=customProperties)
