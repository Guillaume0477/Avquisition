




import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
k = 0

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rows,cols = gray.shape

tx=15
ty=10
angle=180
echelle=0.5

Mr = cv2.getRotationMatrix2D((cols/2,rows/2),angle,echelle)

Mt = np.float32([[0,0,tx],[0,0,ty]])

M = Mr + Mt

while(True):
    ret, frame = cap.read() #1 frame acquise à chaque iteration
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #conversion en n&b


    edges = cv2.Canny(gray,100,200)

    dst = cv2.warpAffine(gray,M,(cols,rows))

    cv2.imshow('Capture_Video', gray) #affichage
    cv2.imshow('Capture_Contours', edges) #affichage
    cv2.imshow('Capture_Affine', dst) #affichage

    key = cv2.waitKey(1) #on évalue la touche pressée

    hist = cv2.calcHist(gray, [0], None, [256], [0,255])

    plt.plot(hist)
    plt.draw()
    plt.pause(0.001)
    plt.cla()

    if key & 0xFF == ord('q'): #si appui sur'q'
        break #sortie de la boucle while
    
    #time.sleep(1)
    if key == ord('c'):
        #enregistre image

        cv2.imwrite("Capture" + str(k)+".png", gray)
        k+=1

    

cap.release()
cv2.destroyAllWindows()


