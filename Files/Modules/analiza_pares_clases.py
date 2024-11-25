#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def analiza_pares_clases(X, y, clases_unicas):
    """
    Analiza los pares de clases y evalúa con el perceptrón de sklearn.
    
    Parámetros:
    -----------
    X : array-like
        Datos de entrada.
    y : array-like
        Etiquetas correspondientes.
    clases_unicas : array-like
        Lista de todas las clases únicas en y.
    
    Devuelve:
    ---------
    None (muestra métricas y gráficas de las matrices de confusión seleccionadas).
    """
    
    
    # Parámetros del perceptrón
    eta = 0.1  # Learning rate
    t = 50  # Número de iteraciones

    
    pares_clases = list(itertools.combinations(clases_unicas, 2))  # Todos los pares posibles
    resultados = []

    for clase1, clase2 in pares_clases:
        print(f"Analizando clases {clase1} vs {clase2}...")
        
        # Filtrar datos para las clases actuales
        X_2C = X[(y == clase1) | (y == clase2)]
        y_2C = y[(y == clase1) | (y == clase2)]
        y_2C_binario = (y_2C == clase1).astype(int)  # Convertir a etiquetas binarias (0, 1)
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_2C, y_2C_binario, stratify=y_2C_binario, train_size=0.7)
        
        # ---- Perceptrón de sklearn ----
        clf = Perceptron(max_iter=t, eta0=eta, random_state=42, shuffle=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Calcular métricas
        exactitud = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average=None).mean()
        sensibilidad = metrics.recall_score(y_test, y_pred, average=None).mean()
        conf_mat = metrics.confusion_matrix(y_test, y_pred)
        
        # Almacenar resultados
        resultados.append({
            "clases": f"{clase1}-{clase2}",
            "exactitud": exactitud,
            "precision": precision,
            "sensibilidad": sensibilidad,
            "conf_matrix": conf_mat
        })

    # Ordenar los resultados por exactitud
    resultados = sorted(resultados, key=lambda x: x["exactitud"])
    n = len(resultados)
    
    # Seleccionar pares con métricas bajas, promedio y altas
    seleccionados = [
        resultados[0],  # El par con exactitud más baja
        resultados[1],  # El segundo más bajo
        resultados[n//2 - 1],  # Cercano al promedio (primero)
        resultados[n//2],  # Cercano al promedio (segundo)
        resultados[-2],  # El segundo más alto
        resultados[-1]   # El más alto
    ]

    # Mostrar matrices seleccionadas
    for resultado in seleccionados:
        print(f"\n--- Clases: {resultado['clases']} ---")
        print(f"Exactitud: {resultado['exactitud']:.4f}")
        print(f"Precisión: {resultado['precision']:.4f}")
        print(f"Sensibilidad: {resultado['sensibilidad']:.4f}")
        
        # Mostrar matriz de confusión
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=resultado["conf_matrix"], 
                                                     display_labels=resultado["clases"].split("-"))
        fig, ax = plt.subplots(figsize=(10, 10))
        cm_display.plot(ax=ax)
        plt.title(f"Matriz de confusión ({resultado['clases']})")
        plt.show()



# In[ ]:


def analiza_pares_clases_entero(X, y, clases_unicas):
    """
    Analiza los pares de clases y evalúa con el perceptrón de sklearn.
    
    Parámetros:
    -----------
    X : array-like
        Datos de entrada.
    y : array-like
        Etiquetas correspondientes.
    clases_unicas : array-like
        Lista de todas las clases únicas en y.
    
    Devuelve:
    ---------
    None (muestra métricas y gráficas de todas las matrices de confusión).
    """
    # Parámetros del perceptrón
    eta = 0.1  # Learning rate
    t = 50  # Número de iteraciones
    
    pares_clases = list(itertools.combinations(clases_unicas, 2))  # Todos los pares posibles
    resultados = []

    for clase1, clase2 in pares_clases:
        print(f"Analizando clases {clase1} vs {clase2}...")

        # Filtrar datos para las clases actuales
        X_2C = X[(y == clase1) | (y == clase2)]
        y_2C = y[(y == clase1) | (y == clase2)]
        y_2C_binario = (y_2C == clase1).astype(int)  # Convertir a etiquetas binarias (0, 1)

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_2C, y_2C_binario, stratify=y_2C_binario, train_size=0.7)

        # ---- Perceptrón de sklearn ----
        clf = Perceptron(max_iter=t, eta0=eta, random_state=42, shuffle=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Calcular métricas
        exactitud = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average=None).mean()
        sensibilidad = metrics.recall_score(y_test, y_pred, average=None).mean()
        conf_mat = metrics.confusion_matrix(y_test, y_pred)

        # Almacenar resultados
        resultados.append({
            "clases": f"{clase1}-{clase2}",
            "exactitud": exactitud,
            "precision": precision,
            "sensibilidad": sensibilidad,
            "conf_matrix": conf_mat
        })

    # Mostrar métricas y matrices de confusión de todos los pares
    for resultado in resultados:
        print(f"\n--- Clases: {resultado['clases']} ---")
        print(f"Exactitud: {resultado['exactitud']:.4f}")
        print(f"Precisión: {resultado['precision']:.4f}")
        print(f"Sensibilidad: {resultado['sensibilidad']:.4f}")

        # Mostrar matriz de confusión
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=resultado["conf_matrix"], 
                                                     display_labels=resultado["clases"].split("-"))
        fig, ax = plt.subplots(figsize=(10, 10))
        cm_display.plot(ax=ax)
        plt.title(f"Matriz de confusión ({resultado['clases']})")
        plt.show()


