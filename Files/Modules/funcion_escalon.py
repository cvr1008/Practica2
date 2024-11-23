#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def funcion_escalon(a,z=0):
    """ Función de activación 
    Toma la suma ponderada de entradas "a" y devuelve 1: si supera el umbral z, o 0: si no lo supera.

    Parámetros
    ----------
        a -- array con los valores de sumatorio de los elementos de un caso de entrenamiento
        z -- valor del umbral para la función de activación

    Devolución
    --------
        yhat_vec -- array con valores obtenidos de g(f)
    """ 
    # función de activación
    yhat_vec = 1 if a > z else 0
    return yhat_vec

