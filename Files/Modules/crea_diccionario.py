#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def crea_diccionario(archivo_claves):

    """
    Función que lee un archivo de texto donde cada línea contiene una clave 
    numérica y un valor numérico y obtiene un diccionario en el cual la clave será 
    el indice (del 0 al 46) y el valor será el caracter ASCII que representa ('A', 'B', '0', '1'...).

    Parámetros
    ----------
        archivo_claves: Cadena de caracteres
                        Ruta del archivo de texto que contiene las claves
                        y valores separados por espacios.
                              

    Devolución
    ----------
        dict: Diccionario
              Un diccionario donde las claves son enteros y los valores son 
              caracteres convertidos a partir de los valores numéricos.
    """
    
    with open(archivo_claves, "r") as archivo:
    # Crea un diccionario vacío
        diccionario = {}
        for linea in archivo:
            # Divide la línea en dos partes: clave y valor
            clave, valor = linea.strip().split()  # Separa por espacios en blanco
            # Agrega al diccionario
            diccionario[int(clave)] = chr(int(valor))

    # Muestra el diccionario
    return(diccionario)

