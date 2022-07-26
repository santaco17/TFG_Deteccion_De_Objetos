import os
import cv2
import glob
import numpy as np
from natsort import natsorted

dir = os.getcwd();
path = '/fotos/PruebaGrande/output/Prueba/*.png'
coordinates = open('coordinates', 'w')

verticalMax = 200
verticalMin = 5
horizontalMax = 400
horizontalMin = 5

i = 0;
for filename in natsorted(glob.glob(dir + path )):

    im = cv2.imread(filename)

    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    a = np.array(g)
    max = a.max()
    mean = a.mean()

    ########OTSU THRESHOLD########
    (T, threshOtsu) = cv2.threshold(img_gray, max-30, max , cv2.THRESH_OTSU)
    print("[INFO] otsu's thresholding value: {}".format(T))

    ######THRESHOLD BINARIO#######
    ret, threshBinary = cv2.threshold(img_gray, max-15, max, cv2.THRESH_BINARY)
    print("[INFO] binary thresholding value: {}".format(ret))

    ######AUTOMATIC MASK#######

    #mask = cv2.inRange(im, np.array([210, 210, 210]), np.array([255, 255, 255]))

    contoursBinary, hierarchy = cv2.findContours(image=threshBinary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE )
    contoursToTest = [];

    for contours in contoursBinary:
        minContour = np.vstack(contours.min(axis=0))
        maxContour = np.vstack(contours.max(axis=0))
        minMaxContourArray = [minContour, maxContour];
        contoursToTest.append(minMaxContourArray);


    #contoursOtsu, hierarchyOtsu = cv2.findContours(image=threshOtsu, mode=cv2.RETR_EXTERNAL  , method=cv2.CHAIN_APPROX_NONE)

    image_copyBinary = im.copy()
    image_copyBinaryToColor = threshBinary.copy()
    image_copyBinaryToColor=cv2.cvtColor(image_copyBinaryToColor,cv2.COLOR_GRAY2RGB)
    for contours in contoursToTest:

        minContour = contours[0]
        maxContour = contours[1]

        right = minContour[0][0]
        left = maxContour[0][0]

        top = minContour[0][1]
        bottom = maxContour[0][1]

        if left - right <= horizontalMin or bottom - top <= verticalMin or left - right >= horizontalMax or bottom-top >= verticalMax:
            continue;

        coordinates.writelines(str(i) + ' ' + str(right)  + ', '  + str(top) + ', ' + str(left) + ', ' + str(bottom) + ('\n') )
        cv2.rectangle(image_copyBinary, (left, top ), (right, bottom), (0, 0, 255), 2)
        cv2.drawContours(image=image_copyBinaryToColor, contours=contoursBinary, contourIdx=-1, color=(255, 0, 0), thickness=2,
                     lineType=cv2.LINE_AA)

    cv2.imshow('Object detection BINARY' + str(i), image_copyBinary);
    cv2.waitKey(0)
    i = i+1;
    cv2.destroyAllWindows()
