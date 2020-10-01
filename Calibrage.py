import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import floor, exp
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

    sizeImage = gray.shape

    #Calcul des contours
    edges = cv2.Canny(gray,100,200)
    #Calcul de l'histogramme
    hist = cv2.calcHist([gray], [0], None, [256], [0,255])
    

    dst = cv2.warpAffine(gray,M,(cols,rows))

    #Calcul de la matrice gaussienne
    [X,Y] = np.meshgrid(np.arange(floor(-sizeImage[1]/2),floor(sizeImage[1]/2), 1), np.arange(floor(-sizeImage[0]/2),floor(sizeImage[0]/2), 1))
    A = 1/100
    sig = [100,100]
    G = A*np.exp(-X**2/(sig[0]**2) - Y**2/(sig[1]**2))
    PX = np.sin(X/np.max(X)) #np.sign(X)#
    PY = np.cos(Y/np.max(Y)) #np.sign(Y)#

    Xmap = X+sizeImage[1]/2+np.multiply(G, PX) 
    Ymap = Y+sizeImage[0]/2+np.multiply(G,PY)
    
    ImGauss = cv2.remap(gray, Xmap.astype(np.float32) , Ymap.astype(np.float32), cv2.INTER_LINEAR)

    #Affichage histogramme
    plt.plot(hist)
    plt.draw()

    #Affichage des images
    cv2.imshow('Capture_Video', gray) #affichage
    cv2.imshow('Capture_Contours', edges) #affichage
    cv2.imshow('Gaussian deformation', ImGauss)
    cv2.imshow('Capture_Affine', dst) #affichage


    #Capture interaction utilisateur
    key = cv2.waitKey(1) #on évalue la touche pressée
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


