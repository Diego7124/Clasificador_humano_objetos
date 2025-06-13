#  Algoritmos Utilizados

Este documento describe los algoritmos de entrenamiento y predicción utilizados en el sistema de detección de objetos en tiempo real.

---

##  Modelo de Clasificación

Se utilizó un modelo de redes neuronales convolucionales (CNN) creado con TensorFlow/Keras y una base preentrenada de MobileNet.

### Creacion del Modelo:
```python
#esta funcion carga una base ya entrenada yle indicaremos con nuestras 3 clases los objetos a identificar    
def create_model():
    '''
    Aqui se crea el modelo de clasificacion, vamos usar MobileNet y utilizar una base preentrenada en ImageNet
    '''
    #cargamos el modelo de MobileNet
    modeloBase = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3),
    #eliminamos las capas de salida de la red para que solo se quede con la de nosotros 
    include_top=False,
    #indicamos usar los pesos de imagennet
    weights='imagenet')
    #con esto evitamos volver a reentrenar el modelo
    modeloBase.trainable = False
    #añadimos nuevas capas para las 3 clases que vamos a utilizar:Persona, Objeto, Fondo
    modelo = tf.keras.Sequential([
        modeloBase,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='softmax')  
    ])
```

## Entrenamiento del modelo

```python
def main():
    #preprocesamos los datos de entrenamiento
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=TMN_IMG,
        batch_size=TMN_LOTE,
        label_mode='categorical'
    )

    #preprocesamos los datos de validacion
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=TMN_IMG,
        batch_size=TMN_LOTE,
        label_mode='categorical'
    )

    #usamos autotune para que el entrenamiento se optimize segun el hardware (esto ya es opcional)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    model = create_model()
    model.summary()

    #empezamos a entrenar nuestro modelo
    model.fit(train_ds,validation_data=val_ds,epochs=EPOCAS)


    #guardamos nuestro modelo:
    model.save('modelo_persona_objeto_fondo.h5')
    print("Modelo guardado como 'modelo_persona_objeto_fondo.h5'")

```

## Preprocesamiento de imagenes con cv2 antes de su prediccion

```python
# Cargamos el modelo ya guardado del codigo anterior
model = tf.keras.models.load_model('modelo_persona_objeto_fondo.h5')

# Clases de  los objetos (Estas tienen que ir en el orden que se tengan en la carpeta)
CLASSES = ['fondo', 'objeto', 'persona']

# Tamaño que espera el modelo para procesar las imagenes
IMG_SIZE = (224, 224)


def preprocess(frame):
    # Redimensionar
    img = cv2.resize(frame, IMG_SIZE)
    # Convertimos la imagen de BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalizamos la imagen 
    img = img / 255.0
    # Añadimos nueva dimension al inicio y lo convertimos aa float 32 ya que tensorflow utiliza este tipo
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img
```

## Clasificacion en tiempo real por la camara

```python

#iniciamos un bucle para indicar que analize cada frame y haga la prediccion
while True:
    

    ret, frame = cap.read()
    if not ret:
        print("Hubo un error al leer el frame")
        break

    # Preprocesamos el frame para el modelo y lo guardamos 
    input_img = preprocess(frame)

    # Predecimos los resultados
    predicts = model.predict(input_img)
    class_id = np.argmax(predicts)
    class_name = CLASSES[class_id]
    confidence = predicts[0][class_id]

    
    label = f"{class_name}: {confidence*100:.2f}%"
    cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
   #mostramos las predicciones en pantalla
    cv2.imshow('Clasificacion en tiempo real', frame)
```


