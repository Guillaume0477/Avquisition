




import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
k = 0

while(True):
    ret, frame = cap.read() #1 frame acquise à chaque iteration
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #conversion en n&b
    cv2.imshow('Capture_Video', gray) #affichage
    key = cv2.waitKey(1) #on évalue la touche pressée

    hist = cv2.calcHist(gray, [0], None, [256], [0,255])

    plt.plot(hist)
    plt.draw()
    plt.pause(0.001)
    plt.cla()

    if key & 0xFF == ord('q'): #si appui sur'q'
        break #sortie de la boucle while
    if key == ord('c'):
        #enregistre image
        cv2.imwrite("Capture" + str(k)+".png", gray)
        k+=1

    

cap.release()
cv2.destroyAllWindows()


