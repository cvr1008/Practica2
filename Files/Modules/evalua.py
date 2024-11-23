#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def evalua(y_test, y_pred):
    import numpy as np
    """
    Evalúa el porcentaje de aciertos de un clasificador.
    Compara las predicciones con los valores reales y calcula el acierto.
    
    Parámetros:
    ----------
        y_test : np.ndarray
            Array con los valores reales (etiquetas conocidas) para los datos de test.
        y_pred : np.ndarray
            Array con los valores predichos por el perceptrón para los datos de test.

    Devolución:
    ----------
        acierto : float
            Porcentaje de valores acertados con respecto al total de elementos.
    """
    # Calcular el porcentaje de aciertos
    # Para calcular el porcentaje debería multiplicarse por 100, pero dado que la referencia dada es una proporción calculada sobre 1, se mantiene así.
    acierto = np.mean(y_test == y_pred) 
    
    return acierto

