import random 
import json 
import pickle 
import numpy as np 
import nltk
from nltk.stem import WordNetLemmatizer 
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, Activation, Dropout 
from tensorflow.python.keras.optimizers import SGD 
 

#------------------Código de preparación de datos

# Importa el lematizador de NLTK
lemmatizer = WordNetLemmatizer()

# Carga datos de un archivo JSON llamado "intents.json"
intents = json.load(open("intents.json").read())


# Inicializa listas y variables
word = []  # Almacenará palabras individuales de las preguntas
classes = []  # Almacenará las etiquetas de intención
documents = []  # Almacenará las palabras y etiquetas de intención
ignorar_letras = ["?", "i", "", "!", ".", ","]  # Lista de caracteres a ignorar en las preguntas

# Itera a través de las intenciones del archivo JSON
for intent in intents["intents"]:
    # Itera a través de los patrones de preguntas en cada intención
    for pattern in intent["patterns"]:
        # Divide las palabras de cada pregunta en palabras individuales
        word_list = nltk.word_tokenize(pattern)
        word.append(word_list)  # Agrega las palabras a la lista 'word'

        # Agrega la pregunta (palabras) y la etiqueta de intención a 'documents'
        documents.append((word_list), intent["tag"])

        # Si la etiqueta de intención no está en 'classes', agrégala
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Imprime los documentos (preguntas y etiquetas de intención)
print(documents)


#----------------------Preprocesamiento de datos
words = [item for sublist in word for item in sublist]  # Combina todas las palabras de las preguntas en una sola lista
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignorar_letras]  # Lematiza palabras, las convierte a minúsculas y excluye palabras en 'ignorar_letras'
words = sorted(list(set(words)))  # Elimina duplicados y ordena las palabras en orden alfabético
classes = sorted(list(set(classes)))  # Ordena las etiquetas de intención en orden alfabético



#--------------Creación de datos de entrenamiento

training = []  # Inicializa la lista de datos de entrenamiento
output_empty = [0] * len(classes)  # Crea una lista de ceros con la longitud de 'classes'

# Itera a través de los documentos que contienen palabras y etiquetas
for doc in documents:
    pattern_words = doc[0]  # Obtiene las palabras de la pregunta
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]  # Lematiza y convierte las palabras a minúsculas
    for w in words:
        training.append(int(w in pattern_words))  # Agrega 1 si una palabra está en la pregunta, 0 si no

output_row = list(output_empty)  # Crea una fila de salida con ceros
output_row[classes.index(doc[1])] = 1  # Establece el valor correspondiente a la etiqueta de intención en 1
training.append(output_row)  # Agrega la fila de salida a los datos de entrenamiento

training = np.array(training)  # Convierte la lista de datos de entrenamiento a un arreglo NumPy
train_x = list(training[:, :len(words)])  # Separa las características (input) del entrenamiento
train_y = list(training[:, len(words):])  # Separa las etiquetas (output) del entrenamiento




# --------------------------Entrenamiento del modelo 

model = Sequential()  # Crea un modelo secuencial

model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  # Agrega una capa densa con 128 unidades de neuronas, función de activación 'relu' y especifica la forma de entrada
model.add(Dropout(0.5))  # Agrega una capa de dropout para reducir el riesgo de sobreajuste
model.add(Dense(64, activation='relu'))  # Agrega otra capa densa con 64 unidades de neuronas y función de activación 'relu'
model.add(Dropout(0.5))  # Agrega otra capa de dropout
model.add(Dense(len(train_y[0]), activation='softmax'))  # Agrega una capa densa con un número de neuronas igual al número de etiquetas y función de activación 'softmax'

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # Configura el optimizador SGD (Gradiente Descendente Estocástico) con hiperparámetros
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # Compila el modelo con la función de pérdida 'categorical_crossentropy' y el optimizador SGD

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5)  # Entrena el modelo con los datos de entrenamiento, especificando el número de épocas y el tamaño del lote
model.save('chatbot_model.h5')  # Guarda el modelo entrenado en un archivo llamado 'chatbot_model.h5'

