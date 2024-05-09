import numpy as np
import cv2 as cv

# Load image
# img = cv.imread('C:/Users/Ignacio/source/repos/DLIP/PyOpenCvExamples/Lab3/LV3.png',1)  # Color scale image
# original_heigth, original_with = img.shape[:2]
# new_heigth = int((600 / original_with) * original_heigth)
# img2 = cv.resize(img, (600,new_heigth))

# cv.namedWindow('src', cv.WINDOW_AUTOSIZE)
# cv.imshow('src', img2)

#Video
cap = cv.VideoCapture('C:/Users/Ignacio/source/repos/DLIP/PyOpenCvExamples/Lab3/LAB3_Video.mp4')

if not cap.isOpened():
    print("Error al abrir el video")
    exit()

ret, img = cap.read()
original_heigth, original_with = img.shape[:2]
new_heigth = int((600 / original_with) * original_heigth)
img2 = cv.resize(img, (600,new_heigth))

#Select ROI
roi = cv.selectROI('Select ROI', img2, fromCenter=False, showCrosshair=True)
x, y, w, h = roi

cv.destroyAllWindows()

while cap.isOpened():
    ret, img = cap.read()

    img2 = cv.resize(img, (600,new_heigth))

    cv.imshow('Video', img2)

    #Separate original images into colours chanels RGB
    canal_azul, canal_verde, canal_rojo = cv.split(img2)

    #Apply median filter to reduce noise and maintain edges
    median = cv.medianBlur(canal_rojo,11)

    roi_selected = median[y:y+h, x:x+w]

    #Detect the edges of the curve
    edges = cv.Canny(roi_selected, 120, 240,None, 3)
    contours, _=cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS )
    #Create a mask to draw contours
    filled_image = np.ones_like(img2)
    cv.drawContours(filled_image[y:y+h, x:x+w], contours, -1, (255,0,255), thickness=3,lineType=cv.LINE_4 )

    #Paint horizontal lines indicating the levels of tension
    max_y = None
    max_point_y = None

    lvl1 = original_heigth - 250 #Level 1: >250px from the bottom of the image
    lvl1 = int((new_heigth / original_heigth) * lvl1)
    lvl2 = original_heigth - 120 #Level 2: 120~250 px from the bottom of the image / Level 3: < 120 px from the bottom of the image
    lvl2 = int((new_heigth / original_heigth) * lvl2)

    cv.line(filled_image, (0, lvl1), (600, lvl1), (0, 255, 255), thickness=1,lineType=cv.LINE_AA)
    cv.line(filled_image, (0, lvl2), (600, lvl2), (255, 255, 0), thickness=1,lineType=cv.LINE_AA)  

    # FInd the lowest point from the detected edge
    for contorno in contours:
        for point in contorno.squeeze():
            if max_y is None or point[1] > max_y:
                max_y = point[1]
                max_point_y = point

    # Determine the level of tension
    max_y = max_y + y
    if(max_y<=lvl1):
        cv.circle(filled_image[y:y+h, x:x+w], max_point_y, 5, (0, 255, 0), -1) 
        color = (0, 255, 0)
        lvl = 1
    elif(max_y>lvl1 and max_y<lvl2):
        cv.circle(filled_image[y:y+h, x:x+w], max_point_y, 5, (0, 255, 255), -1)  
        color=(0, 255, 255)
        lvl = 2
    else:
        cv.circle(filled_image[y:y+h, x:x+w], max_point_y, 5, (255, 255, 0), -1)
        color=(255, 255, 0)
        lvl = 3

    # Desing grafic elements and print
    #The score value is calculated respect the new image size, thats why numbers are not the same with the original sizes of the image. But they are proportional.
    max_y = original_heigth - max_y
    text1 = f"Score: {max_y}"
    text2 = f"Level: {lvl}"
    cv.putText(filled_image, text1, (400,50), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
    cv.putText(filled_image, text2, (400,100), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

    cv.namedWindow('contours', cv.WINDOW_AUTOSIZE)
    cv.imshow('contours', filled_image)

    alpha = 0.5
    imagen_superpuesta = cv.addWeighted(img2, 1 - alpha, filled_image, alpha, 0)
    cv.namedWindow('imagen_superpuesta', cv.WINDOW_AUTOSIZE)
    cv.imshow('imagen_superpuesta', imagen_superpuesta)

    if cv.waitKey(70) & 0xFF == ord('q'):
        break