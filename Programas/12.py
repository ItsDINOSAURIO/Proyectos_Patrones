# import numpy as np
# import matplotlib.pyplot as plt

# Real=np.linspace(0,1,256)
# a=0.1#Ancho
# b=2.5#Caída
# c=0.5#Centro
# C1=1/(1+(np.abs((Real-c)/a)**(2*b)))
# a=10 #
# c=0.5#Punto de inflexión
# C2=1/(1+(np.exp((-a)*(Real-c))))
# C3=1/(1+(np.exp((a)*(Real-c))))

# plt.figure(0)
# plt.plot(Real,C1)
# plt.plot(Real,C2)
# plt.plot(Real,C3)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

def sigmoid(x,c,a, invertido=False):
    if invertido:
        out=1-(1/(1/1+np.exp(-a*(x-c))))
    else:
        out=1/(1+np.exp(-a*(x-c)))
    return out
def bell (x,a,b,c):
    out=1/(1+np.abs((x-c)/a)**(2*b))
    return out

def fuzzy_sets(pixel):
    Pmin=sigmoid(pixel,0.25,55,True)
    Pbla=bell(pixel,0.15,2.5,0.5)
    Pmed=bell(pixel,0.15,2.5,0.65)
    Pmax=sigmoid(pixel,0.75,55,False)

    #S y V
    '''
    Pmin=sigmoid(pixel,0.15,55,True)
    Pbla=bell(pixel,0.25,2.5,0.5)
    Pmax=sigmoid(pixel,0.5,55,False) #Para V la clara tiene que ir de la mitad para delante
    '''
    return Pmin, Pbla, Pmed,Pmax

def plot_fuzzy_sets():
    pixeles=np.linspace(0,1,256)
    Pmin_values=[]
    Pbla_values=[]
    Pmed_values=[]
    Pmax_values=[]
    for pix in pixeles:
        Pmin,Pbla,Pmed,Pmax=fuzzy_sets(pix)
        Pmin_values.append(Pmin)
        Pbla_values.append(Pbla)
        Pmed_values.append(Pmed)
        Pmax_values.append(Pmax)
    plt.figure(figsize=(10,6))
    plt.plot(pixeles,Pmin_values,label="con_1")
    plt.plot(pixeles,Pbla_values,label="con_2")
    # plt.plot(pixeles,Pmed_values,label="con_3")
    plt.plot(pixeles,Pmax_values,label="con_4")
    plt.legend()
    plt.grid(True)
    plt.show()

    #Proceso de determinación de conjuntos "Fuzzyfication"#
plot_fuzzy_sets()
