#Dynamic Time Warping

import numpy as np
import scipy.io.wavfile as wav
nombres=['zero.wav','one.wav','two.wav','three.wav','four.wav','five.wav','six.wav','seven.wav','eight.wav','nine.wav']
car_pat=[]
ventana=0.02 #20 ms
for nom in nombres:
    Fs,senal=wav.read(nom)
    senal=senal/np.max(np.abs(senal))
    if len(senal.shape)>1:
        senal=senal[:,0]
    longitud=int(ventana*Fs)
    paso_ventana=longitud
    num_ventanas=int(np.ceil(float(len(senal)-longitud)/paso_ventana))+1
    long_padded=num_ventanas*paso_ventana+longitud
    senal_padded=np.append(senal,np.zeros(long_padded-len(senal)))
    energia=[]
    zcr=[]
    for i in range(0,num_ventanas):
        inicio=i*paso_ventana
        fin=inicio+longitud
        ventana_01=senal_padded[inicio,fin]
        energia_ventana=np.sum(ventana_01**2)
        energia.append(energia_ventana)
        cruces=np.sum(np.abs(np.diff(np.sign(ventana_01))))/2
        zcr.append(cruces)
    caracteristicas=np.column_stack((energia,zcr))
    car_pat.append(caracteristicas)


#Agregar ICA o PCA, más rápido que ICA pero más inseguro