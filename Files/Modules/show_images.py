#!/usr/bin/env python
# coding: utf-8

# In[ ]:



def show_images(images, labels, caracteres, columnas, filas):

    import matplotlib.pyplot as plt
    
    """ Permite mostrar la primeras n imagenes ( n = columnas x filas)
    Parámetros
    ----------
        images: array 3D con los datos de cada imagen
        labels: array 1D con el numero de clase de cada de las imágenes
        caracteres: diccionario que proprociona el caracter que se corresponde a cada una de las imágenes
        columnas: numero de columnas a mostrar
        filas: numero de filas a mostrar
    """    
    fig, ax = plt.subplots(ncols=columnas,nrows=filas,figsize=(10, 10))
    axes=ax.ravel()
    index = 0   
     
    for x in zip(images, labels):        
        image = x[0]        
        label = caracteres[x[1]]
         
        axes[index].imshow(image,cmap='gray')
        axes[index].set_title(label)
        axes[index].axis("off")
 
        plt.title(label);        
        index += 1
        if index>=(columnas*filas):
            plt.tight_layout()
            plt.show()
            return

