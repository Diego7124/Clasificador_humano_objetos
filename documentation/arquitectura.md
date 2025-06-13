#  Arquitectura del Sistema
Este proyecto implementa un sistema de detección de objetos,Humanos o Fondos/Indefinido en tiempo real utilizando un modelo de aprendizaje profundo entrenado con TensorFlow y procesado mediante OpenCV. A continuación la esctructura:

## 📌 Descripción

El sistema ha sido entrenado con imágenes etiquetadas y hace uso de `TensorFlow` y `OpenCV` para realizar detección en vivo desde la webcam.


- **Backend**: Python + TensorFlow
- **Entrada**: Webcam
- **Procesamiento**: CNN con imágenes normalizadas
- **Salida**: Clasificación en tiempo real (humano / objeto/ fondo)



---

##  Componentes Principales

### 1. Dataset
- Contiene imágenes clasificadas en distintas categorías (por ejemplo: `humans/`, `objects/`, `background/`).
- Estructura tipo:
  - `human/` , `objet/` y `objet/`
  - Cada una siendo una subcarpeta.

### 2. Modelo de Detección
- Entrenado previamente con Keras/TensorFlow.
- Arquitectura de red convolucional básica o CNN personalizada.
- Guardado como carpeta que contiene archivos:
  - `saved_model.pb`
  - `variables/`

### 3. Script de Detección en Tiempo Real
- Usa `OpenCV` para capturar video desde la cámara.
- Procesa cada frame en tiempo real y realiza sus predicciones.

---

## 📁 Estructura de Carpetas

│
├── dataset-fondo-objeto-persona/ # Dataset organizado en subcarpetas (persona/, objeto/, fondo/)
│
├── realtimedetection.py # Entrenamiento del modelo
│
└── detectar_en_tiempo_real/ # Script para ejecutar la detección en tiempo real

al ejecutar realtimedetection.py se creara otro archivo con el nombre del modelo ya entrenado.

---


