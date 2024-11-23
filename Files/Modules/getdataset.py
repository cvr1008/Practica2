#!/usr/bin/env python
# coding: utf-8

# In[ ]:



def getdataset(images,labels, caracteres, num_pix=16):
    """ Obtiene los arrays de numpy con las imágenes y las etiquetas
    Parámetros
    ----------
        imagenes -- estructura de datos que contiene la información de cada una de las imágenes
        etiquetas -- estructura de datos que contiene la información de la clase a la que
        pertenece cada una de las imágenes
        caracteres -- diccionario que contiene la "traducción" a ASCII de cada una de las etiquetas
        num_pix -- valor de la resolución de la imagen (se debe obtener una imagen num_pix x num_pix)
    
    Devolución
    --------
        X -- array 2D (numero_imagenes x numero_pixeles) con los datos de cada una de las imágenes
        y -- array 1D (numero_imagenes) con el caracter que representa cada una de las imágenes
    """ 
    import numpy as np
    from skimage.transform import resize
    import warnings
    warnings.filterwarnings("ignore")

    X = []
    y = []

    for i, lab in zip(images, labels):
        try:
            # Obtener el carácter correspondiente usando el diccionario `caracteres`
            caracter = caracteres[lab]
            
             # Redimensionar la imagen a `num_pix x num_pix` si es necesario
            img_resized = resize(i, (num_pix, num_pix), anti_aliasing=True)
            
            # Aplanar la imagen en un vector de tamaño `num_pix * num_pix`
            img_vector = img_resized.reshape(-1)  # Convierte a 1D
        
            # Agregar la imagen y la etiqueta a las listas
            X.append(img_vector)
            y.append(caracter)


        except Exception as e:
            print(f"Error al procesar la imagen con etiqueta {lab}: {e}")
            continue  # Saltar imágenes que tengan problemas

    X = np.array(X)
    y = np.array(y)

    


    #Completar el código necesario:
    #  Recorre todos las imágenes y etiquetas
    #  Asigna un caracter al valor de etiqueta
    #  Convierte cada imagen a un vector de tamaño (num_pix * num_pix)
    #  Los datos de salida deben ser un array de numpy.
    #  Si alguna imagen no es correcta, muestra un error y no almacenes ni la imagen ni la etiqueta
    
    return X, y

