import os
import cv2
import glob
import string
import requests
import numpy as np
from PIL import Image
from recognition.mse import meanSquaredError, structuralSimilarity

camera = cv2.VideoCapture(0)

cardStore = {}
for file in glob.glob(os.path.dirname(__file__) + '/recognition/cards/*.jpg'):
    cardName = string.capwords(os.path.basename(file)[:-4].replace('-', ' '))
    cardStore[cardName] = {}
    cardStore[cardName]['details'] = None
    cardStore[cardName]['image'] = np.asarray(Image.open(file))

def rotate(l, n):
    return l[-n:] + l[:-n]

def getContours(image, imageContour):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(imageContour, contours, -1, (255, 0, 255), 7)

    longestPerimeter = 0

    points = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)

        if perimeter > longestPerimeter:
            longestPerimeter = perimeter
        else:
            continue

        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        x_, y_, w, h = cv2.boundingRect(approximation)
        # cv2.rectangle(imageContour, (x_, y_), (x_ + w, y_ + h), (0, 255, 0), 5)
        
        # Flatten array and get points
        flattenedApprox = approximation.ravel()
        for i in range(int(len(flattenedApprox) / 2)):
            x = flattenedApprox[2*i]
            y = flattenedApprox[2*i+1]
            # string = str(x) + " " + str(y)
            # cv2.putText(imageContour, string, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
            points.append([x, y])

    return points

def straightenImage(imageContour, points):
    width, height = 168, 234

    if len(points) != 4:
        return imageContour

    # topLeftIndex = 0
    # minDistanceToOrigin = 0
    # for i, point in enumerate(points):
    #     distance = np.linalg.norm(np.array(point))
    #     if distance < minDistanceToOrigin:
    #         minDistanceToOrigin = distance
    #         topLeftIndex = i
    
    # rotate(points, -topLeftIndex)     

    points1 = np.float32(points)

    if np.linalg.norm(np.array(points[0]) - np.array(points[1])) < 300:
        points2 = np.float32([[width,0], [0,0], [0,height], [width,height]])
    else:
        points2 = np.float32([[0,0], [0,height], [width,height], [width,0]])

    matrix = cv2.getPerspectiveTransform(points1, points2)
    return cv2.warpPerspective(imageContour, matrix, (width, height))

def getCardDetails(cardName):
    # Query the scryfall API by card name and cache details
    if not cardStore[cardName]['details']:
        response = requests.get('https://api.scryfall.com/cards/named?exact='+cardName.replace(' ', '+'))
        cardStore[cardName]['details'] = response.json()
    return cardStore[cardName]['details']

def MSEonImage(image):
    mostLikelyCard = { 
        'error': np.infty, 
        'name': ''
    }
    for name, card in cardStore.items():
        try:
            mse = meanSquaredError(image, card['image'])
            # print(f'  {name}: {mse}')
            if mse < mostLikelyCard['error']:
                mostLikelyCard = {
                    'error': mse,
                    'name': name
                }
        except:
            pass
    return mostLikelyCard['name']

def SSIMonImage(image):
    mostLikelyCard = { 
        'likeness': -1, 
        'name': ''
    }
    for name, card in cardStore.items():
        try:
            ssim = structuralSimilarity(image, card['image'])
            # print(f'  {name}: {ssim}')
            if ssim > mostLikelyCard['likeness']:
                mostLikelyCard = {
                    'likeness': ssim,
                    'name': name
                }
        except:
            pass
    return mostLikelyCard['name']

# def OCRonImage(image):
#     text = pytesseract.image_to_string(Image.fromarray(image))
#     return text

while(True):
    success, frame = camera.read()
    baseFrame = frame.copy()
    frameContour = frame.copy()

    frameBlur = cv2.GaussianBlur(frame, (7, 7), 10)
    frameGray = cv2.cvtColor(frameBlur, cv2.COLOR_BGR2GRAY)

    # Canny edge detector
    # threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    # threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    threshold1 = 0
    threshold2 = 70
    frameCanny = cv2.Canny(frameGray, threshold1, threshold2)

    kernel = np.ones((5,5))
    frameDilation = cv2.dilate(frameCanny, kernel, iterations=1)

    points = getContours(frameDilation, frameContour)
    frameStraight = straightenImage(baseFrame, points)

    # print(OCRonImage(frameStraight))

    # msePrediction = MSEonImage(frameStraight)
    # if msePrediction:
    #     print(msePrediction)

    ssimPrediction = SSIMonImage(frameStraight)
    if ssimPrediction:
        points.sort(key=lambda point: point[0] + point[1])
        x, y = points[0]

        cardName = ssimPrediction
        details = getCardDetails(cardName)

        cv2.putText(frameContour, cardName, (x, y - 60), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0))
        cv2.putText(frameContour, 'Price: $' + details['prices']['usd'], (x, y - 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        cv2.putText(frameContour, 'Multiverse ID: ' + str(details['multiverse_ids'][0]), (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('contours', frameContour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(cardStore.keys())
        break

camera.release()
cv2.destroyAllWindows()