import cv2
import numpy as np
import tensorflow as tf

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

# Abre la camara del dispositivo
cap = cv2.VideoCapture(0)  

#si la camara esta abierta o hay algun error se imprimira esto
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara :(")
    exit()



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

    
    #presionar la tecla q para salir del programa
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
