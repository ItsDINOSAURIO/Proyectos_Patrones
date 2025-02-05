from skimage import color,io,morphology,measure
import numpy as np
import matplotlib.pyplot as plt

imagen=io.imread(r"D:\Upiita\6to\Patrones\Objetos\himym.jpg")

hsv=color.rgb2hsv(imagen)
rango_bajo=np.array([0,0.23,0.75])
rango_alto=np.array([0.1,0.68,1.0])
mascara_piel=((hsv[:,:,0]>=rango_bajo[0]) & (hsv[:,:,0]<=rango_alto[0]) & (hsv[:,:,1]>=rango_bajo[1]) & (hsv[:,:,1]<=rango_alto[1]) & (hsv[:,:,2]>=rango_bajo[2]) & (hsv[:,:,2]<=rango_alto[2]))

plt.figure()
plt.imshow(mascara_piel)

mascara=morphology.remove_small_objects(mascara_piel,min_size=400)
mascara=morphology.binary_closing(mascara,morphology.disk(5))

plt.figure()
plt.imshow(mascara)

etiquetas=measure.label(mascara)
regiones=measure.regionprops(etiquetas)

fig,ax=plt.subplots(1)

ax.imshow(imagen)
for region in regiones:
    if region.area>=500:
        minr,minc,maxr,maxc=region.bbox
        rect=plt.Rectangle((minc,minr),(maxc-minc),(maxr-minr),edgecolor='green',facecolor='none',linewidth=2)
        ax.add_patch(rect)

plt.axis('off')


plt.show()