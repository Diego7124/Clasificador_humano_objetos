import tensorflow as tf
from tensorflow.keras import layers
#definimos nuestras constantes como: medidas de la imagen y tamano del lote para procesar, epocas que entrenara nuestro modelo y el directorio donde se encuentra
TMN_IMG = (224, 224) #TAMANO DE IMAGEN
TMN_LOTE = 32 #tamano de imagenes a procesar
EPOCAS = 10  
DATASET_DIR = 'dataset'  #en esta ruta/carpeta se encuentran las imagenes para su entrenamiento

#esta funcion carga una base ya entrenada yle indicaremos con nuestras 3 clases los objetos a identificar    
def create_model():
    
    #Aqui se crea el modelo de clasificacion, vamos usar MobileNet y utilizar una base preentrenada en ImageNet

    #cargamos el modelo de MobileNet
    modeloBase = tf.keras.applications.MobileNetV2(input_shape=(*TMN_IMG, 3),
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
    #compilamos el modelo que habiamos cargado anteriormente y le indicamos como se va a entrenar y el optimizador que utilizara
    modelo.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    return modelo

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

if __name__ == '__main__':
    main()
