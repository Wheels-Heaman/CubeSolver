# 8 Scramble Moves 
# Average of 26 Moves over 1,000,000 solves 
# Last test 1.14024 seconds per solve with Rendering Cube over 1000 solves at Solve and Scramble Speed 1               
# Last test 0.0020355 seconds per solve without Rendering Cube over 1,000,000 solves 

# Import All Modules

import numpy as np
import matplotlib.pyplot as plt
import random, time, cv2
from matplotlib import widgets
from PIL import ImageGrab

# Import Functions from other Python Scripts

from projection import Quaternion, project_points
from webcamFive import createRects, drawRects, calculateAverages, drawColors, calibrateColors

# Variable Constants 
PrintScramble = True 
PrintSolve = True 
DemoMode = True # Use if you don't have your own cube 
CameraNumber = 0 
NoOfSolves = 10 # Number of Solves Per Test 
ScrambleSpeed = 4 # No of Steps to Break 90 Degree Rotation (Greatly Effects Running Speed)...
SolveSpeed = 4 # Set Speed Value to 0 to Not Render Rotations 
NoOfScrambleMoves = 8 # Shouldn't be less than 8 to be Legal Scramble 

# Color Values for matplotlib
White = "w"
Yellow = "#ffcf00"
Orange = "#ff6f00"
Red = "#cf0000"
Green = "#009f0f"
Blue = "#00008f"
Black = "black"
Gray = "gray"

ArrayOfAllColors = [White,Yellow,Orange,Red,Green,Blue,Black,Gray]
AllColorNames = ["White", "Yellow", "Orange", "Red", "Green", "Blue", "Black", "Gray"]

# Cube Constants
NoOfFaces = 6
CubeSize = 2
CubeColors = [Yellow, Blue, Red, Green, Orange, White]
NoOfStickersPerFace = CubeSize * CubeSize

# Arrays For Scramble
CubesFaces = ["F", "B", "L", "R", "U", "D"]
CubesTurns = [-1, 1, 2]

OppositeY = [3,2,1,0]

# Move Arrays For Sticker Dict 
UMoveArray = [[0,1], [0,0], [0,2], [0,1], [0,3], [0,2], [0,0], [0,3], [1,0], [2,0], [2,0], [3,0], [3,0], [4,0], [4,0], [1,0], [1,1], [2,1], [2,1], [3,1], [3,1], [4,1], [4,1], [1,1]]
RMoveArray = [[2,1], [2,0], [2,2], [2,1], [2,3], [2,2], [2,0], [2,3], [0,1], [1,1], [1,1], [5,1], [5,1], [3,3], [3,3], [0,1], [0,2], [1,2], [1,2], [5,2], [5,2], [3,0], [3,0], [0,2]]
FMoveArray = [[1,1], [1,0], [1,2], [1,1], [1,3], [1,2], [1,0], [1,3], [2,0], [0,3], [5,1], [2,0], [4,2], [5,1], [0,3], [4,2], [0,2], [4,1], [4,1], [5,0], [5,0], [2,3], [2,3], [0,2]]
LMoveArray = [[4,1], [4,0], [4,2], [4,1], [4,3], [4,2], [4,0], [4,3], [0,0], [3,2], [3,2], [5,0], [5,0], [1,0], [1,0], [0,0], [0,3], [3,1], [3,1], [5,3], [5,3], [1,3], [1,3], [0,3]]
DMoveArray = [[5,1], [5,0], [5,2], [5,1], [5,3], [5,2], [5,0], [5,3], [1,3], [4,3], [2,3], [1,3], [3,3], [2,3], [4,3], [3,3], [1,2], [4,2], [2,2], [1,2], [3,2], [2,2], [4,2], [3,2]]
BMoveArray = [[3,1], [3,0], [3,2], [3,1], [3,3], [3,2], [3,0], [3,3], [0,0], [2,1], [2,1], [5,2], [5,2], [4,3], [4,3], [0,0], [0,1], [2,2], [2,2], [5,3], [5,3], [4,0], [4,0], [0,1]]

# Algorithms Move List

# 3x3
JAPerm = [["R", -1], ["U", 1], ["L", -1], ["U", 2], ["R", 1], ["U", -1], ["R", -1], ["U", 2], ["L", 1], ["R", 1]]
YPerm = [["F", 1], ["R", 1], ["U", -1], ["R", -1], ["U", -1], ["R", 1], ["U", 1], ["R", -1], ["F", -1], ["R", 1], ["U", 1], ["R", -1], ["U", -1], ["R", -1], ["F", 1], ["R", 1], ["F", -1]]

# 2x2
DoubleJAPerm = [["R", 2], ["U", -1], ["F", 2], ["D", 2], ["R", 2], ["U", -1], ["R", 2]]
DoubleYPerm = [["R", 2], ["F", 2], ["R", 2]]
YJAPerm = [["R", 1], ["U", -1], ["R", 1], ["F", 2], ["R", -1], ["U", 1], ["R", -1]]

# Class Used for Cube Rendering 
class Cube:
    # Defines Constants for Cube Rendering Class 
    defaultPlasticColor = Black
    defaultFaceColors = [White, Yellow, Orange, Red, Green, Blue, Gray, "none"]
    baseFace = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=float)
    stickerWidth = 0.85
    stickerMargin = 0.5 * (1. - stickerWidth)
    stickerThickness = 0.000001

    (d1, d2, d3) = (1 - stickerMargin, 1 - 2 * stickerMargin,1 + stickerThickness)

    baseSticker = np.array([[d1, d2, d3], [d2, d1, d3], [-d2, d1, d3], [-d1, d2, d3], [-d1, -d2, d3], [-d2, -d1, d3], [d2, -d1, d3], [d1, -d2, d3], [d1, d2, d3]], dtype=float)
    baseFaceCentroid = np.array([[0, 0, 1]])
    baseStickerCentroid = np.array([[0, 0, 1 + stickerThickness]])

    # Define rotation angles and axes for the six sides of the cube
    x, y, z = np.eye(3)
    rotations = [Quaternion.from_v_theta(x, theta) for theta in (np.pi / 2, -np.pi / 2)]
    rotations += [Quaternion.from_v_theta(y, theta) for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]

    # Define face movements
    facesDict = dict(F=z, B=-z, R=x, L=-x, U=y, D=-y)

    # Initializes Cube Class Variables 
    def __init__(self):
        self.Cube = CubeSize
        self.plasticColor = self.defaultPlasticColor
        self.faceColors = self.defaultFaceColors
        self.initArrays()

    # Creates Arrays for Cube Rendering 
    def initArrays(self):
        cubieWidth = 2. / CubeSize 
        translations = np.array([[[-1 + (i + 0.5) * cubieWidth, -1 + (j + 0.5) * cubieWidth, 0]] for i in range(CubeSize) for j in range(CubeSize)])

        faceCentroids = []
        faces = []
        stickerCentroids = []
        stickers = []
        colors = []

        factor = np.array([1. / CubeSize, 1. / CubeSize, 1])

        for i in range(6):
            M = self.rotations[i].as_rotation_matrix()
            facesT = np.dot(factor * self.baseFace + translations, M.T)
            stickersT = np.dot(factor * self.baseSticker + translations, M.T)
            faceCentroidsT = np.dot(self.baseFaceCentroid + translations, M.T)
            stickerCentroudsT = np.dot(self.baseStickerCentroid + translations, M.T)
            colorsI = i + np.zeros(faceCentroidsT.shape[0], dtype=int)

            faceCentroidsT = np.hstack([faceCentroidsT.reshape(-1, 3), colorsI[:, None]])
            stickerCentroudsT = stickerCentroudsT.reshape((-1, 3))

            faces.append(facesT)
            faceCentroids.append(faceCentroidsT)
            stickers.append(stickersT)
            stickerCentroids.append(stickerCentroudsT)
            colors.append(colorsI)

        self.faceCentroids = np.vstack(faceCentroids)
        self.faces = np.vstack(faces)
        self.stickerCentroids = np.vstack(stickerCentroids)
        self.stickers = np.vstack(stickers)
        self.colors = np.concatenate(colors)

    # Rotates Face by Altering Arrays for Cube Rendering 
    def rotateFace(self, f, n=1, layer=0):
        v = self.facesDict[f]
        r = Quaternion.from_v_theta(v, n * np.pi / 2)
        M = r.as_rotation_matrix()

        proj = np.dot(self.faceCentroids[:, :3], v)
        cubieWidth = 2. / CubeSize
        flag = ((proj > 0.9 - (layer + 1) * cubieWidth) & (proj < 1.1 - layer * cubieWidth))

        for x in [self.stickers, self.stickerCentroids, self.faces]:
            x[flag] = np.dot(x[flag], M.T)
        self.faceCentroids[flag, :3] = np.dot(self.faceCentroids[flag, :3], M.T)


    # Draws the Cube to the Screen 
    def drawInteractiveCube(self):
        figure = plt.figure(figsize=(8, 8))
        figure.add_axes(InteractiveCube(self))
        figure.patch.set_facecolor('grey')
        figure.canvas.set_window_title('Cube Solver')

        return figure

# Class Used to Manage Solving 
class InteractiveCube(plt.Axes):
    def __init__(self, cube=None, interactive=True, view=(0, 0, 10), figure=None, rect=[0, 0.16, 1, 0.84], **kwarguments):

        if cube is None: # If Cube Parameter Was Not Passed 
            self.cube = Cube(3) # Cube is 3x3
        elif isinstance(cube, Cube): 
            self.cube = cube 
        else:
            self.cube = Cube(cube)

        # Defines Rendering Angle 
        self.view = view
        self.startRotation = Quaternion.from_v_theta((1, -1, 0), -np.pi / 6)

        if figure is None:
            figure = plt.gcf()

        # Removes Default Key Press Events 
        callbacks = figure.canvas.callbacks.callbacks
        del callbacks['key_press_event']

        # Defines Defaults and Draws Axes
        kwarguments.update(dict(aspect=kwarguments.get('aspect', 'equal'), xlim=kwarguments.get('xlim', (-2.0, 2.0)), ylim=kwarguments.get('ylim', (-2.0, 2.0)), frameon=kwarguments.get('frameon', False), xticks=kwarguments.get('xticks', []), yticks=kwarguments.get('yticks', [])))
        super(InteractiveCube, self).__init__(figure, rect, **kwarguments)
        self.xaxis.set_major_formatter(plt.NullFormatter())
        self.yaxis.set_major_formatter(plt.NullFormatter())

        self.startXLim = kwarguments['xlim']
        self.startYLim = kwarguments['ylim']

        # View Rotation Directions and Speeds 
        self.DirectionUpDown = (1, 0, 0)
        self.stepUpDown = 0.01
        self.DirectionLeftRight = (0, -1, 0)
        self.stepLeftRight = 0.01

        self.DirectionLeftRightOpposite = (0, 0, 1)

        # Event Varaibles 
        self.active = False
        self.buttonOne = False 
        self.buttonTwo = False  
        self.mouseCoords = None  
        self.shift = False  

        self.currentRotation = self.startRotation  #current rotation state
        self.facePolys = None
        self.stickerPolys = None

        # Define Events and Call Appropriate Function 
        self.figure.canvas.mpl_connect('button_press_event', self.mousePress)
        self.figure.canvas.mpl_connect('button_release_event',  self.mouseRelease)
        self.figure.canvas.mpl_connect('motion_notify_event', self.mouseMotion)
        self.figure.canvas.mpl_connect('key_press_event', self.keyPress)
        self.figure.canvas.mpl_connect('key_release_event', self.keyRelease)

        self.createStickerDict() 
        self.solveMoves = [] # Stores Moves used to Sovle Scramble 

        self.initWidgets()
        self.drawCube()

    # Creates Buttons For GUI
    def initWidgets(self):
        self.axesReset = self.figure.add_axes([0.75, 0.05, 0.2, 0.075])
        self.buttonReset = widgets.Button(self.axesReset, 'Reset View', color = "0.3", hovercolor = "0.35")
        self.buttonReset.on_clicked(self.resetView)

        self.axesScramble = self.figure.add_axes([0.55, 0.05, 0.2, 0.075])
        self.buttonScramble = widgets.Button(self.axesScramble, "Scramble", color = "0.3", hovercolor = "0.35")
        self.buttonScramble.on_clicked(self.scramble)

        self.axesSolve = self.figure.add_axes([0.35, 0.05, 0.2, 0.075])
        self.buttonSolve = widgets.Button(self.axesSolve, "Solve", color = "0.3", hovercolor = "0.35")
        self.buttonSolve.on_clicked(self.initSolve)

    # Creates Dictionary that Stores Sticker Locations  
    def createStickerDict(self):
        self.stickerDict = {}
        for x in range(len(CubeColors)):
            for y in range(NoOfStickersPerFace):
                self.stickerDict[x,y] = CubeColors[x]
        self.cubeStateDict = dict(self.stickerDict)

    # Corrects Order of Stickers from Sticker Dict to Bridge between Cube Solving System and Cube Rendering System 
    def correctStickerOrder(self, stickersArray):
        copyStickerArray = list(stickersArray)
        stickersArray = np.asarray(stickersArray)
        copyStickerArray = np.asarray(copyStickerArray)
        arrayOfFaces = np.array_split(stickersArray, NoOfFaces)
        copyArrayOfFaces = np.array_split(copyStickerArray, NoOfFaces)
        
        arrayOfFaces[1] = copyArrayOfFaces[5]
        arrayOfFaces[5] = copyArrayOfFaces[1]
        arrayOfFaces[3] = copyArrayOfFaces[2]
        arrayOfFaces[4] = copyArrayOfFaces[3]
        arrayOfFaces[2] = copyArrayOfFaces[4]

        for x in range(NoOfFaces):
            faceCopy = list(arrayOfFaces[x])
            arrayOfFaces[x][1] = faceCopy[0]
            arrayOfFaces[x][3] = faceCopy[1]
            arrayOfFaces[x][0] = faceCopy[3]
            arrayOfFaces[x][2] = faceCopy[2]
        
        stickersArray = []
        for x in range(len(arrayOfFaces)):
            currentArray = arrayOfFaces[x]
            for y in range(len(currentArray)):
                stickersArray.append(currentArray[y])

        return stickersArray

    # Correct Order of Stickers from Stickers Returned by Color Detection System to Bridge Gap with Cube Rendering System
    def correctDetectedColors(self, detectedColors):
        copy = list(detectedColors)
        detectedColors[1] = copy[2]
        detectedColors[2] = copy[3]
        detectedColors[3] = copy[1]
    
        return detectedColors

    # Used for DrawCube Function
    def project(self, pts):
        return project_points(pts, self.currentRotation, self.view, [0, 1, 0])

    # Draws Cube to GUI
    def drawCube(self):
        stickers = self.project(self.cube.stickers)[:, :, :2]
        faces = self.project(self.cube.faces)[:, :, :2]
        faceCentroids = self.project(self.cube.faceCentroids[:, :3])
        stickerCentroids = self.project(self.cube.stickerCentroids[:, :3])
        plasticColor = self.cube.plasticColor

        # Takes Sticker Dictionary and Coverts Values to Format the Draw Function can Understand 
        colors = []
        for x in range(len(CubeColors)):
            for y in range(NoOfStickersPerFace):
                colors.append(self.stickerDict[x,y])
        colors = self.correctStickerOrder(colors)

        facesZOrders = -faceCentroids[:, 2]
        stickerZOrders = -stickerCentroids[:, 2]

        if self.facePolys is None:
            self.facePolys = []
            self.stickerPolys = []

            for i in range(len(colors)):
                fp = plt.Polygon(faces[i], facecolor=plasticColor,
                                 zorder=facesZOrders[i])
                sp = plt.Polygon(stickers[i], facecolor=colors[i],
                                 zorder=stickerZOrders[i])

                self.facePolys.append(fp)
                self.stickerPolys.append(sp)
                self.add_patch(fp)
                self.add_patch(sp)
        else:
            for i in range(len(colors)):
                self.facePolys[i].set_xy(faces[i])
                self.facePolys[i].set_zorder(facesZOrders[i])
                self.facePolys[i].set_facecolor(plasticColor)

                self.stickerPolys[i].set_xy(stickers[i])
                self.stickerPolys[i].set_zorder(stickerZOrders[i])
                self.stickerPolys[i].set_facecolor(colors[i])

        self.figure.canvas.draw()

    # Rotate View of Cube
    def rotate(self, rotation):
        self.currentRotation = self.currentRotation * rotation

    # Rotates Faces for Rendering System 
    def rotateFaceRendered(self, face, turns=1, layer=0, steps=SolveSpeed):
        steps *= turns
        if steps < 0:
            steps *= -1
        if not np.allclose(turns, 0):
            for i in range(steps):
                self.cube.rotateFace(face, turns * 1. / steps, layer=layer)
                self.drawCube()

    # Rotates Faces for Solving System
    def rotateFaceInternal(self, face, turns=1, layer=0, steps=SolveSpeed):
        steps *= turns
        if steps < 0:
            steps *= -1
        if not np.allclose(turns, 0):
            self.rotateStickerDict(face, turns)
            if turns == 2:
                self.rotateStickerDict(face, turns)
            self.solveMoves.append([face, turns])

    # Rotates Cube for Cube Rendering System 
    def rotateCubeRendered(self, axis, turns=1, steps=SolveSpeed):
        steps *= turns
        if steps < 0:
            steps *= -1
        if axis == "X":
            face = "R"
        elif axis == "Z":
            face = "F"
        else:
            face = "U"
        if not np.allclose(turns, 0):
            for i in range(steps):
                for layer in range(CubeSize):
                    self.cube.rotateFace(face, turns * 1. / steps, layer=layer)
                # Subtle Difference In Indentation Means All Faces Revolve Together
                self.drawCube()

    # Rotates Cube for Solving System 
    def rotateCubeInternal(self, axis, turns=1, steps=SolveSpeed):
        steps *= turns
        if steps < 0:
            steps *= -1
        if axis == "X":
            face = "R"
        elif axis == "Z":
            face = "F"
        else:
            face = "U"
        if not np.allclose(turns, 0):
            for layer in range(CubeSize):
                self.rotateStickerDict(face, turns, layer)
            if turns == 2:
                for layer in range(CubeSize):
                    self.rotateStickerDict(face, turns, layer)
            
            self.solveMoves.append([axis, turns])

    # Updates Sticker Dict by Perfoming Array Value Swaps Stored in Move Arrays
    def rotateStickerDict(self, face, direction, layer = 0):  
        if face == "U":
            if layer != 0:
                if (layer + 1) == CubeSize:
                    moveArray = DMoveArray
                    direction *= -1
            else:
                moveArray = UMoveArray   

        elif face == "R":
            if layer != 0:
                if (layer + 1) == CubeSize:
                    moveArray = LMoveArray
                    direction *= -1
            else:
                moveArray = RMoveArray
        elif face == "F":
            if layer != 0:
                if (layer + 1) == CubeSize:
                    moveArray = BMoveArray
                    direction *= -1
            else:
                moveArray = FMoveArray 
        elif face == "L":
            moveArray = LMoveArray
        elif face == "D":
            moveArray = DMoveArray
        elif face == "B":
            moveArray = BMoveArray

        cubeStateCopy = dict(self.cubeStateDict)
        if direction > 0:
            for stickerIndexNo in range(0, len(moveArray), 2):
                stickerIndexOne = moveArray[stickerIndexNo]
                stickerIndexTwo = moveArray[stickerIndexNo + 1]
                self.cubeStateDict[stickerIndexOne[0], stickerIndexOne[1]] = cubeStateCopy[stickerIndexTwo[0], stickerIndexTwo[1]]

        else:
            for stickerIndexNo in range(len(moveArray) -1, 0, -2):
                stickerIndexOne = moveArray[stickerIndexNo]
                stickerIndexTwo = moveArray[stickerIndexNo -1]
                self.cubeStateDict[stickerIndexOne[0], stickerIndexOne[1]] = cubeStateCopy[stickerIndexTwo[0], stickerIndexTwo[1]]

    # Resets View To Default
    def resetView(self, *arguments):
        self.set_xlim(self.startXLim)
        self.set_ylim(self.startYLim)
        self.currentRotation = self.startRotation
        self.drawCube()

    # Returns Positions of All White Pieces
    def getBottomPositions(self, color):
        positions = [] 
        for x in range(NoOfFaces):
            for y in range(NoOfStickersPerFace):
                if self.cubeStateDict[x,y] == color:
                    if x != 5:
                        positions.append([x,y])
    
        return positions

    # Checks if Sticker Directly Bellow Passed Index Already Contains a Bottom Color Sticker (Bottom Color may Vary if Color Neutral)
    def getClearBottom(self, bottomColor):
        clearBottom = []
        for x in range(NoOfStickersPerFace):
            if self.cubeStateDict[5, x] != bottomColor:
                clearBottom.append(x)

        return clearBottom

    # Solves Corner Passed (First Identifies Where the Corner is then Executes "Face Neutral" Algorithms)
    def placeCorner(self, position):
        if position[0] == 0:
            if position[1] == 0:
                face = "L"
            elif position[1] == 1:
                face = "B"
            elif position[1] == 2:
                face = "R"
            else:
                face = "F"
            self.rotateFaceInternal(face, 1)
            self.rotateFaceInternal("U", 2)
            self.rotateFaceInternal(face, -1)
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(face, 1)
            self.rotateFaceInternal("U", 1)
            self.rotateFaceInternal(face, -1)
        elif position[1] == 1:
            if position[0] == 1:
                face = "R"
            elif position[0] == 2:
                face = "B"
            elif position[0] == 3:
                face = "L"
            else:
                face = "F"
            self.rotateFaceInternal("U", 1)
            self.rotateFaceInternal(face, 1)
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(face, -1)
        elif position[1] == 0:
            if position[0] == 1:
                face = "L"
            elif position[0] == 2:
                face = "F"
            elif position[0] == 3:
                face = "R"
            else:
                face = "B"
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(face, -1)
            self.rotateFaceInternal("U", 1)
            self.rotateFaceInternal(face, 1)
        elif position[1] == 2:
            if position[0] == 1:
                face = "R"
            elif position[0] == 2:
                face = "B"
            elif position[0] == 3:
                face = "L"
            else:
                face = "F"
            self.rotateFaceInternal(face, 1)
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(face, -1)
            self.rotateFaceInternal("U", 1)
            self.rotateFaceInternal(face, 1)
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(face, -1)
        elif position[1] == 3:
            if position[0] == 1:
                face = "L"
            elif position[0] == 2:
                face = "F"
            elif position[0] == 3:
                face = "R"
            else:
                face = "B"
            self.rotateFaceInternal(face, -1)
            self.rotateFaceInternal("U", 1)
            self.rotateFaceInternal(face, 1)
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(face, -1)
            self.rotateFaceInternal("U", 1)
            self.rotateFaceInternal(face, 1)

    # Returns Color of Front Facing Stickers of Level Passed ("top" or "bottom")
    def getFrontColor(self, level):
        if level == "top":
            y = 0, 1
            face = "U"
        else:
            y = 2, 3
            face = "D"
        frontColor = None
        noCorrect = 0
        for x in range(1,5):
            if self.cubeStateDict[x, y[0]] == self.cubeStateDict[x, y[1]]:
                frontColor = self.cubeStateDict[x, y[0]]
                noCorrect += 1
        if frontColor == self.cubeStateDict[1, y[0]] or frontColor == None:
            return noCorrect
        else:
            if noCorrect != 4:
                while self.cubeStateDict[1, y[0]] != frontColor:
                    self.rotateFaceInternal(face, 1)
     
            return noCorrect
        
    # Returns No Of "topColor" Stickers on Top ("top Color" is dependent on what was Identified as most Effiecient Bottom Color )
    def getTopOnTop(self, topColor):
        topNo = 0
        for y in range(0,4):
            if self.cubeStateDict[0, y] == topColor:
                topNo += 1

        return topNo

    # Solves Top (First Identifies Orientation then Executes "Face Neautral" Algorithm)
    def performOLL(self, algorithm, topColorIndex):
        if algorithm == "AntiSune":
            if topColorIndex == 0:
                face = "R"
            elif topColorIndex == 1:
                face = "F"
            elif topColorIndex == 2:
                face = "L"
            else:
                face = "B"
            self.rotateFaceInternal(face, -1)
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(face, 1)
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(face, -1)
            self.rotateFaceInternal("U", 2)
            self.rotateFaceInternal(face, 1)

        elif algorithm == "Sune":
            if topColorIndex == 3:
                face = "R"
            elif topColorIndex == 0:
                face = "F"
            elif topColorIndex == 1:
                face = "L"
            else:
                face = "B"
            self.rotateFaceInternal(face, 1)
            self.rotateFaceInternal("U", 1)
            self.rotateFaceInternal(face, -1)
            self.rotateFaceInternal("U", 1)
            self.rotateFaceInternal(face, 1)
            self.rotateFaceInternal("U", 2)
            self.rotateFaceInternal(face, -1)

        elif algorithm == "Headlights":
            if topColorIndex == 0:
                faceOne = "F"
                faceTwo = "R"
            elif topColorIndex == 1:
                faceOne = "L"
                faceTwo = "F"
            elif topColorIndex == 2:
                faceOne = "B"
                faceTwo = "L"
            else:
                faceOne = "R"
                faceTwo = "B"
            self.rotateFaceInternal(faceOne, 1)
            self.rotateFaceInternal("U", 1)
            self.rotateFaceInternal(faceTwo, 1)
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(faceTwo, -1)
            self.rotateFaceInternal(faceOne, -1)

        elif algorithm == "Chamelon":
            if topColorIndex == 0:
                faceOne = "L"
                faceTwo = "B"
            elif topColorIndex == 1:
                faceOne = "B"
                faceTwo = "R"
            elif topColorIndex == 2:
                faceOne = "R"
                faceTwo = "F"
            else:
                faceOne = "F"
                faceTwo = "L"
            self.rotateFaceInternal(faceOne, 1)
            self.rotateFaceInternal("U", 1)
            self.rotateFaceInternal(faceOne, -1)
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(faceOne, -1)
            self.rotateFaceInternal(faceTwo, 1)
            self.rotateFaceInternal(faceOne, 1)
            self.rotateFaceInternal(faceTwo, -1)

        elif algorithm == "OppositeCorners":
            if topColorIndex == 0:
                faceOne = "F"
                faceTwo = "R"
            elif topColorIndex == 1:
                faceOne = "R"
                faceTwo = "B"
            elif topColorIndex == 2:
                faceOne = "B"
                faceTwo = "L"
            else:
                faceOne = "L"
                faceTwo = "F"
            self.rotateFaceInternal(faceOne, 1)
            self.rotateFaceInternal(faceTwo, 1)
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(faceTwo, -1)
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(faceTwo, 1)
            self.rotateFaceInternal("U", 1)
            self.rotateFaceInternal(faceTwo, -1)
            self.rotateFaceInternal(faceOne, -1)

        elif algorithm == "Symmetric":
            if topColorIndex == 2:
                face = "R" 
            else:
                face = "B"
            self.rotateFaceInternal(face, 2)
            self.rotateFaceInternal("U", 2)
            self.rotateFaceInternal(face, 1)
            self.rotateFaceInternal("U", 2)
            self.rotateFaceInternal(face, 2)
        else:
            if topColorIndex == 0:
                faceOne = "R"
                faceTwo = "B"
            elif topColorIndex == 1:
                faceOne = "B"
                faceTwo = "L"
            elif topColorIndex == 2:
                faceOne = "L"
                faceTwo = "F"
            else:
                faceOne = "F"
                faceTwo = "R"

            self.rotateFaceInternal(faceOne, 1)
            self.rotateFaceInternal(faceTwo, 1)
            self.rotateFaceInternal("U", 1)
            self.rotateFaceInternal(faceTwo, -1)
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(faceTwo, 1)
            self.rotateFaceInternal("U", 1)
            self.rotateFaceInternal(faceTwo, -1)
            self.rotateFaceInternal("U", -1)
            self.rotateFaceInternal(faceOne, -1)

    # Prints Moves Executes for Task (Solve or Scramble)
    def printMovesExecuted(self, moves):
        movesPrint = ""
        for move in moves:
            if move[1] == -1:
                direction = "'"
            elif move[1] == 1:
                direction = ""
            else:
                direction = str(move[1])
            movesPrint += (move[0] + direction + " ")
        print movesPrint

    # Scrambles Cube (Needs Improving)
    def scramble(self, *arguments):
        scramble = []

        # No need to check first face
        face = CubesFaces[random.randint(0, 5)]
        noOfTurns = CubesTurns[random.randint(0, 2)]
        self.rotateFaceRendered(face, noOfTurns, steps=ScrambleSpeed)
        self.rotateFaceInternal(face, noOfTurns, steps=ScrambleSpeed)
        scramble.append([face, noOfTurns])

        for n in range(NoOfScrambleMoves - 1):
            legalFace = True 
            face = CubesFaces[random.randint(0, 5)]
            noOfTurns = CubesTurns[random.randint(0, 2)]
            index = CubesFaces.index(face)
            odd = index % 2
            if odd:
                oppositeFaceIndex = index - 1
            else:
                oppositeFaceIndex = index + 1

            # Checks that Face ISN'T same as Previous
            if face == scramble[n][0]:    
                legalFace = False

            # If Previous Face is Opposite Checks that 2 Previous ISN'T same (Has same effect as selecting two sames faces in a row)
            if CubesFaces[oppositeFaceIndex] == scramble[n][0]:
                if scramble[n -1][0] == face:
                    legalFace = False

            # Selects Random Face Until Legal Face is Found
            while legalFace == False:
                legalFace = True
                face = CubesFaces[random.randint(0, 5)]
                index = CubesFaces.index(face)
                odd = index % 2
                if odd:
                    oppositeFaceIndex = index - 1
                else:
                    oppositeFaceIndex = index + 1

                # Checks that Face ISN'T same as Previous
                if face == scramble[n][0]:
                    legalFace = False

                # If Previous Face is Opposite Checks that 2 Previous ISN'T same (Has same effect as selecting two sames faces in a row)
                if CubesFaces[oppositeFaceIndex] == scramble[n][0]:
                    if scramble[n -1][0] == face:
                        legalFace = False
            self.rotateFaceRendered(face, noOfTurns, steps=ScrambleSpeed)
            self.rotateFaceInternal(face, noOfTurns, steps=ScrambleSpeed)
            scramble.append([face, noOfTurns])

        if PrintScramble:
            print "Scramble"
            self.printMovesExecuted(scramble)

    # Used For Testing and Error Handling 
    def checkIfSolved(self):
        for x in range(NoOfFaces):
            for y in range(NoOfStickersPerFace):
                newY = y-1
                if newY == -1:
                    newY = NoOfStickersPerFace - 1
                if self.cubeStateDict[x,newY] != self.cubeStateDict[x,y]:
                    print False
                    return False

        print True 
        return True 

    # Identifies which Color is most Efficient to Sovle on the Bottom (When Color Neatural) 
    def getBottomColor(self):
        greatestNoCube = 0
        bestColorCube = None
        faceNo = None
        for x in range(NoOfFaces):
            colorNumberArray = [[Yellow, 0], [Blue, 0], [Red, 0], [Green, 0], [Orange, 0], [White, 0]]
            for y in range(NoOfStickersPerFace):
                for z in range(NoOfFaces):
                    if colorNumberArray[z][0] == self.cubeStateDict[x,y]:
                        colorNumberArray[z][1] += 1

            bestColorFace, greatestNoFace = max(colorNumberArray, key=lambda item: item[1])
            if greatestNoFace > greatestNoCube:
                greatestNoCube = greatestNoFace
                bestColorCube = bestColorFace
                faceNo = x

        if faceNo != 5:
            if faceNo == 0:
                self.rotateCubeInternal("X", 2)
            elif faceNo == 1:
                self.rotateCubeInternal("X", -1)
            elif faceNo == 3:
                self.rotateCubeInternal("X", 1)
            elif faceNo == 2:
                self.rotateCubeInternal("Z", 1)
            else:
                self.rotateCubeInternal("Z", -1)

        return bestColorCube

    # Solves Cube Using Begginners Method
    def isBelowClear(self, working, bottomColor):
        if working[0] == 0:
            if working[1] == 0:
                belowY = 3
            elif working[1] == 1:
                belowY = 2
            elif working[1] == 2:
                belowY = 1
            else:
                belowY = 0
        else:
            if working[0] == 1:
                if working[1] == 0 or working[1] == 3:
                    belowY = 0
                elif working[1] == 1 or working[1] == 2:
                    belowY = 1

            if working[0] == 2:
                if working[1] == 0 or working[1] == 3:
                    belowY = 1
                elif working[1] == 1 or working[1] == 2:
                    belowY = 2

            if working[0] == 3:
                if working[1] == 0 or working[1] == 3:
                    belowY = 2
                elif working[1] == 1 or working[1] == 2:
                    belowY = 3

            if working[0] == 4:
                if working[1] == 0 or working[1] == 3:
                    belowY = 3
                elif working[1] == 1 or working[1] == 2:
                    belowY = 0
 

        if self.cubeStateDict[5, belowY] != bottomColor:
            return True
        else:
            return False


    # First Groups Together Like Moves for Efficieny Then Corrects Errors Such as Triple Moves and 0/4 Moves (Caused By Combining Like Moves)
    def amendSolveInefficiencies(self): # Too Many If Else Statements But More Compact Code Was Hell to Write so This is What it Came To (Not Proud)
        workingArray = self.solveMoves
        newArray = []
        x = 0
        while x < (len(workingArray)):
            if x < len(workingArray) - 1: # Prevents Looking Past Ultimate Element 
                if workingArray[x][0] == workingArray[x+1][0]: # Two in a Row 
                    if x < len(workingArray) - 2: # Prevents Looking Past Ultimate Element 
                        if workingArray[x][0] == workingArray[x+2][0]: # Three in a Row 
                            if x < len(workingArray) - 3: # Prevents Looking Past Ultimate Element 
                                if workingArray[x][0] == workingArray[x+3][0]: # Four in a Row! (Due to Algs Finishing How the Next Starts, I Swear Im NOT and Idiot)
                                    sumOfRot = workingArray[x][1] + workingArray[x+1][1] + workingArray[x+2][1] + workingArray[x+3][1]
                                    if sumOfRot != 4 or sumOfRot != 0:
                                        if sumOfRot == 3:
                                            sumOfRot = -1
                                        newArray.append([workingArray[x][0], sumOfRot])    
                                    x+=4
                                else:
                                    sumOfRot = workingArray[x][1] + workingArray[x+1][1] + workingArray[x+2][1]
                                    if sumOfRot != 4 or sumOfRot != 0:
                                        if sumOfRot == 3:
                                            sumOfRot = -1
                                        newArray.append([workingArray[x][0], sumOfRot])     
                                    x+=3
                            else:
                                sumOfRot = workingArray[x][1] + workingArray[x+1][1] + workingArray[x+2][1]
                                if sumOfRot != 4 or sumOfRot != 0:
                                    if sumOfRot == 3:
                                        sumOfRot = -1
                                    newArray.append([workingArray[x][0], sumOfRot])     
                                x+=3
                        else:
                            sumOfRot = workingArray[x][1] + workingArray[x+1][1]
                            if sumOfRot != 4 or sumOfRot != 0:
                                if sumOfRot == 3:
                                    sumOfRot = -1
                                newArray.append([workingArray[x][0], sumOfRot])     
                            x+=2
                    else:
                        sumOfRot = workingArray[x][1] + workingArray[x+1][1]
                        if sumOfRot != 4 or sumOfRot != 0:
                            if sumOfRot == 3:
                                sumOfRot = -1
                            newArray.append([workingArray[x][0], sumOfRot])     
                        x+=2
                else: # None in a Row 
                    newArray.append(workingArray[x])
                    x+=1 
            else:
                newArray.append(workingArray[x])
                x+=1
       
        self.solveMoves = newArray

       
    # Solves Until Encounters Error (SHOULD Prevent the Program from Crashing)
    def initSolve(self, *arguments):
        broken = self.solve()
        while broken or not self.checkIfSolved:
            broken = self.solve()

    # Solves the Cube 
    def solve(self):
        self.solveMoves = []
        bottomColor = self.getBottomColor() # Decides Bottom Color 
        bottomPositions = self.getBottomPositions(bottomColor) # Returns Bottom Color Positions (Not Solved)

        numberOfUMoves = 0 
        while len(bottomPositions) > 0: # While there are still Bottom Color Stickers Not Sovled 
            position = bottomPositions[0]
            while not self.isBelowClear(position, bottomColor): # Rotates U Face until A[propriate for Corner Solve
                numberOfUMoves += 1 
                if numberOfUMoves > 3:
                    return True
                self.rotateFaceInternal("U", 1)
                bottomPositions = self.getBottomPositions(bottomColor)
                position = bottomPositions[0]

            self.placeCorner(position)
            bottomPositions = self.getBottomPositions(bottomColor)

        noCorrectBottom = self.getFrontColor("bottom") # How Solved are the Bottom Sides 

        # Looks Messy but more Compact the Other Methods 
        if bottomColor == Yellow:
            topColor = White
        elif bottomColor == White:
            topColor = Yellow
        elif bottomColor == Blue:
            topColor = Green
        elif bottomColor == Green:
            topColor = Blue
        elif bottomColor == Red:
            topColor = Orange
        else:
            topColor = Red

        # Solves Top 
        topOnTop = self.getTopOnTop(topColor) # How Solved is the Top 
        if topOnTop != 4: 
            if topOnTop == 1:
                algorithm = "Sune"
                for z in range(4):
                    if self.cubeStateDict[0, z] == topColor:
                        topColorIndex = z
                    if self.cubeStateDict[z + 1, 0] == topColor:
                        algorithm = "AntiSune"
            elif topOnTop == 2:
                algorithm = "Chamelon"
                for z in range(4):
                    z2 = z - 1
                    if z2 == -1:
                        z2 = 3
                    z3 = z - 2
                    if z3 == -1:
                        z3 = 3
                    elif z3 == -2:
                        z3 = 2
                    if self.cubeStateDict[0, z] == topColor and self.cubeStateDict[0, z2] == topColor:
                        topColorIndex = z
                    elif self.cubeStateDict[0, z] == topColor and self.cubeStateDict[0, z3] == topColor:
                        algorithm = "OppositeCorners"
                        for z in range(4):
                            if self.cubeStateDict[z + 1, 0] == topColor:
                                topColorIndex = z
                    if self.cubeStateDict[z + 1, 0] == topColor and self.cubeStateDict[z + 1, 1] == topColor:
                        algorithm = "Headlights"

            else:
                algorithm = "Symmetric"
                totalDouble = 0
                for z in range(4):
                    if self.cubeStateDict[z + 1, 0] == topColor and self.cubeStateDict[z + 1, 1] == topColor:
                        topColorIndex = z 
                        totalDouble += 1
                if totalDouble != 2:
                    algorithm = "NonSymmetric"

            self.performOLL(algorithm, topColorIndex)

        noCorrectTop = self.getFrontColor("top") # How Solves are the Top Sides

        # Solves PBL 
        if noCorrectBottom == 4: # If Bottom Solved 
            if noCorrectTop == 1:
                self.performAlgorithmInternal(JAPerm)
            elif noCorrectTop == 0:
                self.performAlgorithmInternal(YPerm)
        elif noCorrectBottom == 1:
            if noCorrectTop == 4: # If Top Solved
                self.rotateFaceInternal("B", 2)
                self.performAlgorithmInternal(JAPerm)
                self.rotateFaceInternal("U", -1)
                self.rotateFaceInternal("B", 2)
            elif noCorrectTop == 1:
                self.performAlgorithmInternal(DoubleJAPerm)
            else:
                ## Correct for Cube Rotation 
                self.rotateFaceInternal("F", 2)
                self.rotateFaceInternal("B", 2)
                self.performAlgorithmInternal(YJAPerm)


        else: # If Bottom Not Solved
            if noCorrectTop == 4: # If Top Solved 
                ## Correct for Cube Rotation 
                self.rotateFaceInternal("F", 2)
                self.rotateFaceInternal("B", 2)
                self.performAlgorithmInternal(YPerm)

            elif noCorrectTop == 0:
                self.performAlgorithmInternal(DoubleYPerm)
            else:
                self.performAlgorithmInternal(YJAPerm)

        while self.cubeStateDict[1,0] != self.cubeStateDict[1,3]:
            self.rotateFaceInternal("U", 1)

        # Checks over Solve and Makes Improvements 
        self.amendSolveInefficiencies()

        if PrintSolve:
            # Don't Count Cube Rotations as Moves 
            moveDeduction = 0
            for x in range(len(self.solveMoves)):
                if self.solveMoves[x][0] == "X" or self.solveMoves[x][0] == "Y" or self.solveMoves[x][0] == "Z":
                    moveDeduction += 1
            print "Solved in:", len(self.solveMoves) - moveDeduction
            self.printMovesExecuted(self.solveMoves)

        self.performAlgorithmRendered(self.solveMoves)

    # Once the Solve is Found, Executes Moves for Rendered Cube
    def performAlgorithmRendered(self, algorithm):
        for move in algorithm:
            if move[0] == "X" or move[0] == "Y" or move[0] == "Z":
                self.rotateCubeRendered(move[0], move[1])
            else:
                self.rotateFaceRendered(move[0], move[1])

    # Once the Solve is Found, Executes Moves for "Internal" System 
    def performAlgorithmInternal(self, algorithm):
        for move in algorithm:
            if move[0] == "X" or move[0] == "Y" or move[0] == "Z":
                self.rotateCubeInternal(move[0], move[1])
            else:
                self.rotateFaceInternal(move[0], move[1])

    # Handles Key Press Events 
    def keyPress(self, event):
        if event.key == 'shift':
            self.shift = True
        elif event.key == 'right':
            if self.shift:
                DirectionLeftRight = self.DirectionLeftRightOpposite
            else:
                DirectionLeftRight = self.DirectionLeftRight
            self.rotate(Quaternion.from_v_theta(DirectionLeftRight, 5 * self.stepLeftRight))
        elif event.key == 'left':
            if self.shift:
                DirectionLeftRight = self.DirectionLeftRightOpposite
            else:
                DirectionLeftRight = self.DirectionLeftRight
            self.rotate(Quaternion.from_v_theta(DirectionLeftRight, -5 * self.stepLeftRight))
        elif event.key == 'up':
            self.rotate(Quaternion.from_v_theta(self.DirectionUpDown, 5 * self.stepUpDown))
        elif event.key == 'down':
            self.rotate(Quaternion.from_v_theta(self.DirectionUpDown, -5 * self.stepUpDown))
        elif event.key.upper() in 'LRUDBF':
            if self.shift:
                direction = -1
            else:
                direction = 1

         
            self.rotateFaceInternal(event.key.upper(), direction)
            self.rotateFaceRendered(event.key.upper(), direction)
        elif event.key.upper() == "A":
            self.scramble()
        elif event.key.upper() == "S":
            self.initSolve()
        elif event.key.upper() == "T":
            totalTime = 0
            totalNoOfMoves = 0
            for x in range(NoOfSolves):
                self.scramble()
                start = time.time()
                self.initSolve()
                totalNoOfMoves += len(self.solveMoves)
                totalTime += (time.time() - start)
                start = time.time()
                if self.checkIfSolved:
                    print x + 1, " Cubes Solved"
            print "Average Time Per Solve:", totalTime/ NoOfSolves, "Seconds"
            print "Average Number of Moves:", totalNoOfMoves / NoOfSolves
            print " "

        elif event.key.upper() == "C":
            print self.checkIfSolved()

        elif event.key.upper() in "XZY":
            if self.shift:
                direction = -1
            else:
                direction = 1

            self.rotateCube(event.key.upper(), direction)

        # Launches WebCam 
        elif event.key.upper() == "W":
            self.resetView()

            cap = cv2.VideoCapture(CameraNumber)

            TopLeft = 100, 50
            CubieSize = 200
            Padding = 50

            YellowWebcam = 66, 244, 226
            BlueWebcam = 244, 78, 66
            RedWebcam = 66, 66, 244
            GreenWebcam = 98, 244, 66
            OrangeWebcam = 66, 161, 244
            WhiteWebcam = 255, 255, 255

            colorsBGR = [YellowWebcam, BlueWebcam, RedWebcam, GreenWebcam, OrangeWebcam, WhiteWebcam]
            colorsHSV = [[26.159300000000002, 114.65090000000008, 114.3373], [102.20989999999998, 170.32320000000004, 76.107300000000009], [171.25239999999997, 150.37210000000002, 70.242299999999972], [68.361499999999978, 113.70349999999996, 80.432000000000031], [163.27959999999996, 163.37390000000002, 131.10299999999995], [54.942199999999985, 11.840499999999999, 98.711799999999997]]

            colorNames = ["Yellow", "Blue", "Red", "Green", "Orange", "White"]

            rects, sampleRects = createRects(CubeSize, CubieSize, TopLeft, Padding)
            startTime = time.time()
            calculating = False
            calibrating = False
            squareColor = None
            colorsArray = None
            newColors = []
            inDemoMode = False 
            while True:
                if inDemoMode:
                    frame = cv2.imread("exampleImages/exampleImage%s.png" %(faceIndex + 1))
                    if faceIndex > 5:
                        _, frame = cap.read()

                else:
                    _, frame = cap.read()
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                drawRects(frame, NoOfStickersPerFace, rects, sampleRects, WhiteWebcam)
                drawColors(colorsArray, frame, colorsBGR, NoOfStickersPerFace, sampleRects)
                if calculating == True:
                    colorsArray = calculateAverages(hsv, NoOfStickersPerFace, sampleRects, colorsHSV)

                cv2.imshow("Original", frame)

                key = cv2.waitKey(25)
                if key == 113: # Closes Window 
                    cv2.destroyAllWindows()
                    self.cubeStateDict = dict(self.stickerDict)
                    calculating = False
                    calibrating = False
                    break

                elif key == 32: 
                    if calibrating == False:
                        faceIndex = 0
                        calibrating = True
                    else:
                        if faceIndex < 6:
                            newColors = calibrateColors(faceIndex, hsv, newColors, sampleRects)
                            faceIndex += 1
                        else:
                            print newColors
                            calculating = False
                            calibrating = False

                elif key == 13:
                    if calculating == False:
                        calculating = True
                        faceIndex = 0
                        for sticker in self.stickerDict:
                            self.stickerDict[sticker] = Black
                            self.drawCube()
                        if DemoMode:
                            inDemoMode = True 
                    else:
                        if faceIndex < 6:
                            colorsArrayCorrected = self.correctDetectedColors(colorsArray)
                            for x in range(NoOfStickersPerFace):
                                self.stickerDict[faceIndex, x] = CubeColors[colorsArrayCorrected[x]]
                                self.drawCube()
                            faceIndex += 1
                        else:
                            cv2.destroyAllWindows()
                            self.cubeStateDict = dict(self.stickerDict)
                            calculating = False
                            calibrating = False
                            inDemoMode = False 
                            break
        
            self.drawCube()

        self.drawCube()

    # Sets Key State to False
    def keyRelease(self, event):
        if event.key == 'shift':
            self.shift = False
        elif event.key.isdigit():
            self.digitsStates[int(event.key)] = 0

    # Sets Mouse Button Pressed to True
    def mousePress(self, event):
        self.mouseCoords = (event.x, event.y)
        if event.button == 1:
            self.buttonOne = True
        elif event.button == 3:
            self.buttonTwo = True

    # Sets Mouse Button Released to False
    def mouseRelease(self, event):
        self.mouseCoords = None
        if event.button == 1:
            self.buttonOne = False
        elif event.button == 3:
            self.buttonTwo = False

    # Rotates Cube if Left Click Zooms with Right Click
    def mouseMotion(self, event):
        if self.buttonOne or self.buttonTwo:
            dx = event.x - self.mouseCoords[0]
            dy = event.y - self.mouseCoords[1]
            self.mouseCoords = (event.x, event.y)

            if self.buttonOne:
                if self.shift:
                    DirectionLeftRight = self.DirectionLeftRightOpposite
                else:
                    DirectionLeftRight = self.DirectionLeftRight
                rot1 = Quaternion.from_v_theta(self.DirectionUpDown, self.stepUpDown * dy)
                rot2 = Quaternion.from_v_theta(DirectionLeftRight, self.stepLeftRight * dx)

                self.rotate(rot1 * rot2)
                self.drawCube()

            if self.buttonTwo:
                factor = 1 - 0.003 * (dx + dy)
                xlim = self.get_xlim()
                ylim = self.get_ylim()
                self.set_xlim(factor * xlim[0], factor * xlim[1])
                self.set_ylim(factor * ylim[0], factor * ylim[1])

                self.figure.canvas.draw()

if __name__ == '__main__':
    import sys
    c = Cube()
    c.drawInteractiveCube()

    plt.show()
