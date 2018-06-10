import cv2, time
from PIL import ImageGrab
import numpy as np
# # 03432 221 234

# cap = cv2.VideoCapture(0)


# TopLeft = 100, 50
# CubieSize = 200
# Padding = 50

# YellowWebcam = 66, 244, 226
# BlueWebcam = 244, 78, 66
# RedWebcam = 66, 66, 244
# GreenWebcam = 98, 244, 66
# OrangeWebcam = 66, 161, 244
# WhiteWebcam = 255, 255, 255



# NoOfFaces = 6
# CubeSize = 2
# NoOfStickersPerFace = CubeSize * CubeSize
# colorsBGR = [YellowWebcam, BlueWebcam, RedWebcam, GreenWebcam, OrangeWebcam, WhiteWebcam]
# # Blue = 99, 255, 139
# # Red = 174, 208, 119
# # Green = 78, 250, 159
# # Orange = 6, 164, 184
# # White = 40, 21, 161
# # Yellow 36, 175, 171
# # colorsHSV = [(41, 82, 235), (95, 255, 181), (174, 185, 195), (76, 191, 205), (6, 135, 253), (40.5983,   13.9768,  207.6184)]
# # colorsHSV = [(74.87483072916667, 72.45318684895834, 133.87873046875, 0.0), (71.14132486979167, 150.59005208333335, 140.57203776041666, 0.0), (110.75356445312501, 107.35980794270834, 139.44041666666666, 0.0), (59.826923828125004, 118.44148763020834, 143.28761067708334, 0.0), (45.98081380208334, 86.021943359375, 159.42790364583334, 0.0), (37.841748046875004, 61.10380859375, 134.75288736979167, 0.0)]
# colorsHSV = ([  38.8544,   96.5101,  244.0201]),([  95.5691,  254.9239,  197.8534]),([ 176.316 ,  201.8031,  190.3882]),([  75.6778,  193.268 ,  224.6914]),([   2.1281,  113.7182,  254.9999]),([  38.3355,   14.1049,  213.3314])
# colorNames = ["Yellow", "Blue", "Red", "Green", "Orange", "White"]

def createRects(CubeSize, CubieSize, TopLeft, Padding):
	rects = []
	sampleRects = []
	stickerNo = 0
	for x in range(CubeSize):
		for y in range(CubeSize):
			differenceX = x * CubieSize
			differenceY = y * CubieSize
			rects.append([(TopLeft[0] + differenceX, TopLeft[1] + differenceY), (TopLeft[0] + (differenceX + CubieSize), TopLeft[1] + (differenceY + CubieSize))])
			sampleRects.append([(rects[stickerNo][0][0] + Padding, rects[stickerNo][0][1] + Padding), (rects[stickerNo][1][0] - Padding, rects[stickerNo][1][1]- Padding)])
			stickerNo += 1

	return rects, sampleRects
# 

def drawRects(frame, NoOfStickersPerFace, rects, sampleRects, WhiteWebcam):
	for x in range(NoOfStickersPerFace):
		cv2.rectangle(frame, rects[x][0], rects[x][1], WhiteWebcam, 2)
		cv2.rectangle(frame, sampleRects[x][0], sampleRects[x][1], WhiteWebcam, 2)

	return frame
	

def calculateAverages(hsv, NoOfStickersPerFace, sampleRects, colorsHSV):
	bestColors = []
	for z in range(NoOfStickersPerFace):
		rect = hsv[sampleRects[z][0][1]:sampleRects[z][1][1], sampleRects[z][0][0]:sampleRects[z][1][0]]
		averageColorPerRow  = np.average(rect, axis=0)
		average = np.average(averageColorPerRow, axis=0)

		bestTotalDifference = 400
		for x in range(6):
			totalDifference = 0
			for y in range(3):
				difference = average[y] - colorsHSV[x][y]
				if difference < 0:
					difference *= -1
				totalDifference += difference
			if totalDifference < bestTotalDifference:
				bestTotalDifference = totalDifference
				bestColor = x

		bestColors.append(bestColor)
	
	return bestColors

def drawColors(colorsArray, frame, colorsBGR, NoOfStickersPerFace, sampleRects):
	if colorsArray != None:
		for x in range(len(colorsArray)):
			colorsArray[x] = colorsBGR[colorsArray[x]]
	
		for x in range(NoOfStickersPerFace):
			cv2.rectangle(frame, sampleRects[x][0], sampleRects[x][1], colorsArray[x], -1)

def calibrateColors(faceIndex, hsv, newColors, sampleRects):
	rect = hsv[sampleRects[0][0][1]:sampleRects[0][1][1], sampleRects[0][0][0]:sampleRects[0][1][0]]
	averageColorPerRow  = np.average(rect, axis=0)
	average = np.average(averageColorPerRow, axis=0)

	newColors.append([average[0], average[1], average[2]])

	return newColors


# rects, sampleRects = createRects()
# startTime = time.time()
# calculating = False
# calibrating = False
# squareColor = None
# colorsArray = None
# faceIndex = 0

# while True:
# 	_, frame = cap.read()
# 	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# 	drawRects()
# 	drawColors(colorsArray)
# 	if calculating == True:
# 		colorsArray = calculateAverages(hsv)

# 	cv2.imshow("Original", frame)

# 	key = cv2.waitKey(25)
# 	if key == 113:
# 		cv2.destroyAllWindows()
# 		break

# 	elif key == 32:
# 		if calibrating == False:
# 		 	calibrating = True
# 		else:
# 			if faceIndex < 6:
# 				colorsHSV = calibrateColors(faceIndex)
# 				faceIndex += 1
# 			else:
# 				print colorsHSV

# 	elif key == 13:
# 		calculating = True
			

# 	print colorsArray

# cv2.destroyAllWindows()
# cap.release()