# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io,color,transform

# def sigmoid(x,c,a, invertido=False):
#     if invertido:
#         out=1-(1/(1/1+np.exp(-a*(x-c))))
#     else:
#         out=1/(1+np.exp(-a*(x-c)))
#     return out
# def bell (x,a,b,c):
#     out=1/(1+np.abs((x-c)/a)**(2*b))
#     return out

# def Defuzzy(rango,datos):
#     numerador=np.sum(rango*datos)
#     denominador=np.sum(datos)
#     return numerador/denominador if denominador else 0


# Ima=io.imread(r"D:\Upiita\6to\Patrones\piel.jpg")
# Ima=transform.resize(Ima,(480,640,3),anti_aliasing=True)
# hsv=color.rgb2hsv(Ima)
# fil,col,_=hsv.shape
# binario=np.zeros((fil,col))
# reglas=np.zeros(47)

# h_bajo, h_mediobajo,h_medioalto, h_alto=(0.25,55,True),(0.15,2.5,0.35),(0.15,2.5,0.65),(0.75,75,False)
# s_bajo, s_mediobajo,s_medioalto, s_alto=(0.1,55,True),(0.15,2.5,0.25),(0.15,2.5,0.55),(0.65,75,False)
# v_claro, v_medio, v_oscuro=(0.1,55,True),(0.15,2.5,0.4),(0.6,75,False)


# #Generar conjuntos de salida:

# piel=(0.45,55,True)
# no_piel=(0.55,55,False)
# valor=np.linspace(0,1,100)
# piel_fuzzy=sigmoid(valor,*piel)
# no_piel_fuzzy=sigmoid(valor,*no_piel)

# relaciones=np.zeros((len(reglas),len(no_piel_fuzzy)))


# #Sistema difuso Mamdani agarra los mínimos y de todos los mínimos agarra el máximo
# for i in range(fil):
#     for j in range(col):
#         h,s,v=hsv[i,j] #Mundo real
#         #Se mapean el h,s,v actual en todos los conjuntos difusos
#         h_fuzzy=[sigmoid(h,*h_bajo),bell(h,*h_mediobajo),bell(h,*h_medioalto),sigmoid(h,*h_alto)] #El asterísco denota que se toma cada elemento para el correcto uso de la función
#         s_fuzzy=[sigmoid(s,*s_bajo),bell(s,*s_mediobajo),bell(s,*s_medioalto),sigmoid(s,*s_alto)]
#         v_fuzzy=[sigmoid(v,*v_oscuro),bell(v,*v_medio),sigmoid(v,*v_claro)]
# cont=0
# for i in range(3):
#     for j in range(3):
#         for k in range(2):
#             reglas[cont]=np.min([h_fuzzy[i],s_fuzzy[j],v_fuzzy[k]])
#             cont+=1
# for i in range(len(reglas)):
#     relaciones[i,:]=np.minimum(reglas[i],no_piel_fuzzy)

# relaciones[4,:]=np.minimum(reglas[5],piel_fuzzy)

# # print(reglas)
# # print(relaciones)

# salida=np.max(relaciones,axis=0)
# # print(salida)
# centroide=Defuzzy(np.linspace(0.0,1.0,100),salida)

# if(centroide<0.5):
#     binario[i,j]=1

# #La base de conocimiento son todas las implicaciones difusas: es decir todas las combinaciones posibles, para éste caso específico son 48, por lo que se define el universo y se defino todo aquello que es de interes
# plt.figure(0)
# plt.imshow(hsv)

# plt.figure()
# plt.imshow(binario,cmap='gray')

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
 
def sigmoide(x,c,a,invertido=False):
    if invertido:
        out = 1-(1/(1+np.exp(-a*(x-c))))
    else:
        out = 1/(1+np.exp(-a*(x-c)))
    return out
 
def bell(x,a,b,c):
    out = 1/(1+np.abs((x-c)/a)**(2*b))
    return out
 
def defuzzy(rango,datos):
    numerador = np.sum(rango*datos)
    denominador = np.sum(datos)
    return numerador/denominador if denominador else 0
 
ima = io.imread(r"D:\Upiita\6to\Patrones\piel.jpg")
ima = transform.resize(ima,(480,640,3),anti_aliasing=True)
hsv = color.rgb2hsv(ima)
plt.figure()
plt.imshow(hsv)
fil,col,_=hsv.shape
binario=np.zeros((fil,col))
#-----------------------------------------------------------------------------
h_bajo,h_camp1,h_camp2,h_alto=(0.25,55,True),(0.15,2.5,0.4),(0.15,2.5,0.65),(0.75,55,False)
S_bajo,S_camp1,S_camp2,S_alto=(0.1,55,True),(0.15,2.5,0.25),(0.15,2.5,0.55),(0.65,55,False)
V_bajo,V_camp1,V_alto=(0.15,55,True),(0.15,2.5,0.4),(0.6,55,False)
 
piel=(0.45,0.55,True)
no_piel=(0.55,55,False)
 
 
valor=np.linspace(0,1,101)
piel_fuzzy=sigmoide(valor,*piel)
no_piel_fuzzy=sigmoide(valor,*no_piel)
 
 
#------------------------------------------------------------------------------
reglas=np.zeros(48)
for i in range(fil):
    for j in range(col):
        h,s,v=hsv[i,j]  #Si es el valor real, es necesario pasar del real al difuso
        h_fuzzy=[sigmoide(h,*h_bajo),bell(h,*h_camp1),bell(h,*h_camp2),sigmoide(h,*h_alto)]   #El * es para que recorra cada elemento, tipo en matlab
        S_fuzzy=[sigmoide(s,*S_bajo),bell(s,*S_camp1),bell(s,*S_camp2),sigmoide(s,*S_alto)]
        V_fuzzy=[sigmoide(v,*V_bajo),bell(v,*V_camp1),sigmoide(v,*V_alto)]
        #La base de conocimientos son las implicaciones que cumplen las condiciones que necesitamos
#Ahora hay que generar las reglas difusas
        reglas[0]=np.min([h_fuzzy[0],S_fuzzy[0],V_fuzzy[0]])   #Bajo bajo bajo significa que es no piel
        reglas[1]=np.min([h_fuzzy[0],S_fuzzy[0],V_fuzzy[1]])
        reglas[2]=np.min([h_fuzzy[0],S_fuzzy[0],V_fuzzy[2]])
        reglas[3]=np.min([h_fuzzy[0],S_fuzzy[1],V_fuzzy[0]])
        reglas[4]=np.min([h_fuzzy[0],S_fuzzy[1],V_fuzzy[1]])
        reglas[5]=np.min([h_fuzzy[0],S_fuzzy[1],V_fuzzy[2]])
        reglas[6]=np.min([h_fuzzy[0],S_fuzzy[2],V_fuzzy[0]])
        reglas[7]=np.min([h_fuzzy[0],S_fuzzy[2],V_fuzzy[1]])
        reglas[8]=np.min([h_fuzzy[0],S_fuzzy[2],V_fuzzy[2]])
        reglas[9]=np.min([h_fuzzy[0],S_fuzzy[3],V_fuzzy[0]])
        reglas[10]=np.min([h_fuzzy[0],S_fuzzy[3],V_fuzzy[1]])
        reglas[11]=np.min([h_fuzzy[0],S_fuzzy[3],V_fuzzy[2]])
        reglas[12]=np.min([h_fuzzy[1],S_fuzzy[0],V_fuzzy[0]])
        reglas[13]=np.min([h_fuzzy[1],S_fuzzy[0],V_fuzzy[1]])
        reglas[14]=np.min([h_fuzzy[1],S_fuzzy[0],V_fuzzy[2]])
        reglas[15]=np.min([h_fuzzy[1],S_fuzzy[1],V_fuzzy[0]])
        reglas[16]=np.min([h_fuzzy[1],S_fuzzy[1],V_fuzzy[1]])
        reglas[17]=np.min([h_fuzzy[1],S_fuzzy[1],V_fuzzy[2]])
        reglas[18]=np.min([h_fuzzy[1],S_fuzzy[2],V_fuzzy[0]])
        reglas[19]=np.min([h_fuzzy[1],S_fuzzy[2],V_fuzzy[1]])
        reglas[20]=np.min([h_fuzzy[1],S_fuzzy[2],V_fuzzy[2]])
        reglas[21]=np.min([h_fuzzy[1],S_fuzzy[3],V_fuzzy[0]])
        reglas[22]=np.min([h_fuzzy[1],S_fuzzy[3],V_fuzzy[1]])
        reglas[23]=np.min([h_fuzzy[1],S_fuzzy[3],V_fuzzy[2]])
        reglas[24]=np.min([h_fuzzy[2],S_fuzzy[0],V_fuzzy[0]])
        reglas[25]=np.min([h_fuzzy[2],S_fuzzy[0],V_fuzzy[1]])
        reglas[26]=np.min([h_fuzzy[2],S_fuzzy[0],V_fuzzy[2]])
        reglas[27]=np.min([h_fuzzy[2],S_fuzzy[1],V_fuzzy[0]])
        reglas[28]=np.min([h_fuzzy[2],S_fuzzy[1],V_fuzzy[1]])
        reglas[29]=np.min([h_fuzzy[2],S_fuzzy[1],V_fuzzy[2]])
        reglas[30]=np.min([h_fuzzy[2],S_fuzzy[2],V_fuzzy[0]])
        reglas[31]=np.min([h_fuzzy[2],S_fuzzy[2],V_fuzzy[1]])
        reglas[32]=np.min([h_fuzzy[2],S_fuzzy[2],V_fuzzy[2]])
        reglas[33]=np.min([h_fuzzy[2],S_fuzzy[3],V_fuzzy[0]])
        reglas[34]=np.min([h_fuzzy[2],S_fuzzy[3],V_fuzzy[1]])
        reglas[35]=np.min([h_fuzzy[2],S_fuzzy[3],V_fuzzy[2]])
        reglas[36]=np.min([h_fuzzy[3],S_fuzzy[0],V_fuzzy[0]])
        reglas[37]=np.min([h_fuzzy[3],S_fuzzy[0],V_fuzzy[1]])
        reglas[38]=np.min([h_fuzzy[3],S_fuzzy[0],V_fuzzy[2]])
        reglas[39]=np.min([h_fuzzy[3],S_fuzzy[1],V_fuzzy[0]])
        reglas[40]=np.min([h_fuzzy[3],S_fuzzy[1],V_fuzzy[1]])
        reglas[41]=np.min([h_fuzzy[3],S_fuzzy[1],V_fuzzy[2]])
        reglas[42]=np.min([h_fuzzy[3],S_fuzzy[2],V_fuzzy[0]])
        reglas[43]=np.min([h_fuzzy[3],S_fuzzy[2],V_fuzzy[1]])
        reglas[44]=np.min([h_fuzzy[3],S_fuzzy[2],V_fuzzy[2]])
        reglas[45]=np.min([h_fuzzy[3],S_fuzzy[3],V_fuzzy[0]])
        reglas[46]=np.min([h_fuzzy[3],S_fuzzy[3],V_fuzzy[1]])
        reglas[47]=np.min([h_fuzzy[3],S_fuzzy[3],V_fuzzy[2]])
        R0 = np.minimum(reglas[0],no_piel_fuzzy)
        R1 = np.minimum(reglas[1],no_piel_fuzzy)
        R2 = np.minimum(reglas[2],no_piel_fuzzy)
        R3 = np.minimum(reglas[3],no_piel_fuzzy)
        R4 = np.minimum(reglas[4],no_piel_fuzzy)
        R5 = np.minimum(reglas[5],piel_fuzzy)
        R6 = np.minimum(reglas[6],no_piel_fuzzy)
        R7 = np.minimum(reglas[7],no_piel_fuzzy)
        R8 = np.minimum(reglas[8],no_piel_fuzzy)
        R9 = np.minimum(reglas[9],no_piel_fuzzy)
        R10 = np.minimum(reglas[10],no_piel_fuzzy)
        R11 = np.minimum(reglas[11],no_piel_fuzzy)
        R12 = np.minimum(reglas[12],no_piel_fuzzy)
        R13 = np.minimum(reglas[13],no_piel_fuzzy)
        R14 = np.minimum(reglas[14],no_piel_fuzzy)
        R15 = np.minimum(reglas[15],no_piel_fuzzy)
        R16 = np.minimum(reglas[16],no_piel_fuzzy)
        R17 = np.minimum(reglas[17],no_piel_fuzzy)
        R18 = np.minimum(reglas[18],no_piel_fuzzy)
        R19 = np.minimum(reglas[19],no_piel_fuzzy)
        R20 = np.minimum(reglas[20],no_piel_fuzzy)
        R21 = np.minimum(reglas[21],no_piel_fuzzy)
        R22 = np.minimum(reglas[22],no_piel_fuzzy)
        R23 = np.minimum(reglas[23],no_piel_fuzzy)
        R24 = np.minimum(reglas[24],no_piel_fuzzy)
        R25 = np.minimum(reglas[25],no_piel_fuzzy)
        R26 = np.minimum(reglas[26],no_piel_fuzzy)
        R27 = np.minimum(reglas[27],no_piel_fuzzy)
        R28 = np.minimum(reglas[28],no_piel_fuzzy)
        R29 = np.minimum(reglas[29],no_piel_fuzzy)
        R30 = np.minimum(reglas[30],no_piel_fuzzy)
        R31 = np.minimum(reglas[31],no_piel_fuzzy)
        R32 = np.minimum(reglas[32],no_piel_fuzzy)
        R33 = np.minimum(reglas[33],no_piel_fuzzy)
        R34 = np.minimum(reglas[34],no_piel_fuzzy)
        R35 = np.minimum(reglas[35],no_piel_fuzzy)
        R36 = np.minimum(reglas[36],no_piel_fuzzy)
        R37 = np.minimum(reglas[37],no_piel_fuzzy)
        R38 = np.minimum(reglas[38],no_piel_fuzzy)
        R39 = np.minimum(reglas[39],no_piel_fuzzy)
        R40 = np.minimum(reglas[40],no_piel_fuzzy)
        R41 = np.minimum(reglas[41],no_piel_fuzzy)
        R42 = np.minimum(reglas[42],no_piel_fuzzy)
        R43 = np.minimum(reglas[43],no_piel_fuzzy)
        R44 = np.minimum(reglas[44],no_piel_fuzzy)
        R45 = np.minimum(reglas[45],no_piel_fuzzy)
        R46 = np.minimum(reglas[46],no_piel_fuzzy)
        R47 = np.minimum(reglas[47],no_piel_fuzzy)
        Salida=np.max([R0,R1,R2,R3,R4,R5,R6,R7,R7,R8,R9,R10,R11,R12,R13,R14,R15,R16,R17,R18,R19,R20,R21,R22,R23,R24,R25,R26,R27,R28,R29,R30,R31,R32,R33,R34,R35,R36,R37,R38,R39,R40,R41,R42,R43,R44,R45,R46,R47],axis=0)
        centroide=defuzzy(np.linspace(0.0,1.0,101),Salida)
 
        if (centroide<0.5):
            binario[i,j]=1
plt.figure(3)
plt.imshow(binario,cmap='gray')

plt.show()