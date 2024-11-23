#!/usr/bin/env python
# coding: utf-8

# In[ ]:



def predice(w, X, z, funcion_activacion):

    import numpy as np
    """
    Función para la predicción.
    Calcula el net_input para cada dato de prueba.
    Devuelve preddiciones usando la función escalón.
    
    Parámetros:
    ----------
        w : np.ndarray
            Array con los pesos obtenidos en el entrenamiento del perceptrón.
        X : np.ndarray
            Valores de x_i para cada uno de los datos de test.
        z : float
            Valor del umbral para la función de activación.
        funcion_activacion : callable
            Función de activación para el perceptrón.

    Devolución:
    ----------
        y_pred : np.ndarray
            Array con los valores predichos para los datos de test.
    """
    # Calcular net input para cada ejemplo de prueba
    net_input = np.dot(X, w[1:]) + w[0] - z
    
    # Aplicar función de activación a cada valor de net input
    y_pred = np.array([funcion_activacion(x) for x in net_input])
    
    return y_pred

