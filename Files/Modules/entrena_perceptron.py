#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def entrena_perceptron(X, y, z, eta, t, funcion_activacion):
    """
    Entrena un perceptrón simple
    Inicia los parámetros a 0, y para cada dato de cada una de las iteraciones, se calcula la suma de las entradas y el bias.
    Emplea la función escalón como la función de activación.
    Ajusta los pesos si la predicción no es correcta.
    Devuelve pesos ajustados y los errores.

    
    Parámetros
    ----------
        X : array
            Valores de x_i para cada uno de los datos de entrenamiento.
        y : array
            Valor de salida deseada para cada uno de los datos de entrenamiento.
        z : float
            Valor del umbral para la función de activación.
        eta : float
            Coeficiente de aprendizaje.
        t : int
            iteraciones que se quieren realizar con los datos de entrenamiento.
        funcion_activacion : función (aquí será la funcion_escalon)
            Función de activación para el perceptrón.
    
    Devolución
    ----------
        w : array de numpy
            Valores de los pesos del perceptrón.
        J : lista
            Error cuadrático obtenido de comparar la salida deseada con la que se obtiene 
            con los pesos de cada iteración.
    """
    import numpy as np
    
    # Inicialización de los pesos (incluye el bias como el primer peso)
    w = np.zeros(X.shape[1] + 1)  # Agregar un peso para el bias
    n = 0  # Número de iteraciones se inicializa a 0
    
    # Inicialización de variables adicionales
    J = []  # Error cuadrático en cada iteración
    
    while n < t:
        # Inicializar variables para esta iteración
        yhat_vec = np.zeros(len(y))  # Predicciones de cada ejemplo
        
        # Iterar sobre todos los ejemplos
        for i, x in enumerate(X):
            # Calcular el producto interno (net input) más el bias
            net_input = np.dot(x, w[1:]) + w[0] 
            
            # Predicción usando la función de activación
            yhat = funcion_activacion(net_input)
            yhat_vec[i] = yhat
            
            # Actualizar pesos si hay error
            update = eta * (y[i] - yhat)
            w[1:] += update * x  # Actualización de los pesos
            w[0] += update       # Actualización del bias
        
        # Calcular el error cuadrático de la iteración
        errors = (y - yhat_vec) ** 2
        J.append(0.5 * np.sum(errors))
        
        # Incrementar contador de iteraciones
        n += 1
    
    # Devuelve los pesos y el error cuadrático
    return w, J

