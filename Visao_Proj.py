#!/usr/bin/env python

## Alan Michel Birbrier
## Jonatas Bazzoli
## Lucas de Aguiar Simões
## 8º N

import cv2 as cv
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as mat
from collections import Counter

# Lê do MNIST e Treina utilizando SVM linear com o HOG
def treinaDigitos():
    conjuto_dados = datasets.fetch_mldata("MNIST Original")#captura MNIST
    car = mat.array(conjuto_dados.data, 'int16')#sava as imagens numpy array
    legenda = mat.array(conjuto_dados.target, 'int')# associa as imagens correspondente ao numero
    lista_do_hog = [] # Armazena os hog de cada imagem 
    for ca in car:
        #Calcula as Caracteristica do hog tamanho de 7x7
        fd = hog(ca.reshape((28, 28)), orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2), visualise=False)
        lista_do_hog.append(fd)
        print(fd)
    hog_car = mat.array(lista_do_hog, 'float64')
    print "contando digitos treinado", Counter(legenda)
    clf = LinearSVC()# Classificão multi-classes
    clf.fit(hog_car, legenda)# Classifica com a respota se é 1 ..10
    #Salva o treino 
    joblib.dump(clf, "digitos_clasificado.pkl", compress=3)

# Faz as claficações com base no treino ultilizando SVN linear com HOG
def digitosReconhecedor(): 
    treino = joblib.load("digitos_clasificado.pkl")#Carrega o Treino
    image = cv.imread("photo_10.jpg")#Carrega a Imagem
    image_cinza = cv.cvtColor(image, cv.COLOR_BGR2GRAY)# Transforma tons de cinza
    # Removendo ruido aplicando filtro Gaussiano de escala de cinza
    image_cinza = cv.GaussianBlur(image_cinza, (5,5),0)
    # Converte a imagem de escala de cinza para binario ou seja binariza
    returno, image_threshold = cv.threshold(image_cinza,95,255, cv.THRESH_BINARY_INV)
    # Calcula o contorno da imagem
    contornos, cnt2 = cv.findContours(image_threshold.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # Faz um retangulo na borda de cada contorno
    retangulos = [cv.boundingRect(contorno) for contorno in contornos]
    
    for retangulo in retangulos:
        cv.rectangle(image, (retangulo[0],retangulo[1]), (retangulo[0] + retangulo[2], retangulo[1]+retangulo[3]),(0,255,0),3)
        legendas = int(retangulo[3]*1.7)# Faz as legendas
        pt1 = int(retangulo[1] + retangulo[3]/2 - legendas/2)
        pt2 = int(retangulo[0] + retangulo[2]/2 - legendas/2)
        roi = image_threshold[pt1:pt1+legendas, pt2:pt2+legendas]
        roi = cv.resize(roi, (28, 28), interpolation=cv.INTER_AREA)# Faz um redicionamento na image para 28x28 e dilata na proxima linha. 
        roi = cv.dilate(roi, (3, 3)) # dilata
        # Calcula as Caracteristica do hog tamanho de 7x7 igual ao que esta no metodo treinaDigitos()
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2), visualise=False)
        # Faz reconhecimento ou aproximação com base no treino
        numero = treino.predict(mat.array([roi_hog_fd], 'float64'))
        # Escreve os numeros preditos na imagen com os retangulos 
        cv.putText(image, str(int(numero[0])), (retangulo[0], retangulo[1]),cv.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    cv.imshow("Resulting Image with Rectangular ROIs", image)# Mostra Imagem
    cv.waitKey() # espera um tecla do teclado


def main():
    treinaDigitos()
    digitosReconhecedor()
main()