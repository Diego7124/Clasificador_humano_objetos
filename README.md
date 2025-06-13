# Detección de Humanos en Tiempo Real con TensorFlow y MobileNetV2

Este proyecto implementa un modelo de clasificación binaria para detectar **humanos** frente a **otros objetos** utilizando imágenes personalizadas y una cámara en tiempo real. El modelo está basado en MobileNetV2 preentrenado con ImageNet y adaptado a tres clases: **Persona**, **Objeto**, y **Fondo**.

## 📷 Funcionalidades

- Entrenamiento con imágenes personalizadas (no-humanos vs humanos).
- Modelo ligero con `MobileNetV2` para alto rendimiento.
- Detección en tiempo real con `OpenCV`.
- Clasificación en 3 clases: `Persona`, `Objeto`, `Fondo`.

---


## 🔧 Requisitos

- Python 3.8 o superior
- TensorFlow 2.x
- OpenCV
- NumPy


Instala las dependencias necesarias:
```bash
pip install tensorflow opencv-python numpy matplotlib
```
---

## 🚀 Pasos para Ejecutar

### 1. Clonar el Repositorio

```bash
git clone https://github.com/Diego7124/Clasificador_humano_objetos.git
cd Deteccion_Humanos
```

--- 

descarga el dataset con los datos de entrenamiento:
[DropBox]('https://www.dropbox.com/scl/fi/k8gj1ngrwtniqn4mn5de6/dataset.rar?rlkey=kprsqiz26gkwl5wmi0kweiyj0&st=6er60v02&dl=0')

Pasos
## 1. Descargar el archivo
## 2.Descomprimir el archivo
## 3.Moverlo a la carpeta/directorio donde se clone el repositorio
## 4.Verificar la estructura
## 📁 Estructura de Carpetas

│
├── dataset-fondo-objeto-persona/ # Dataset organizado en subcarpetas (persona/, objeto/, fondo/)
│
├── realtimedetection.py # Entrenamiento del modelo
│
└── detectar_en_tiempo_real/ # Script para ejecutar la detección en tiempo real

al ejecutar realtimedetection.py se creara otro archivo con el nombre del modelo ya entrenado.
## 5. Asegurate que todo este correcto y ejecuta el codigo

```bash
python realtimedetection.py
```
esto empezara el entrenamiento del modelo

### 2. Espera a que el modelo se termine de entrenar para  usarlo.

---

## 🚀 Cómo Ejecutar la deteccion en camara

### 1. Asegurate de que el modelo este entrenado y tengas el archivo en tu carpeta


```bash
python detectar_en_tiempo_real.py
```
esto cargara el  modelo ya entrenado y empezara a ejecutarse la deteccion en tiempo real.

