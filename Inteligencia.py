import random 
import json 
import pickle 
import numpy as np 
import nltk
from nltk.stem import WordNetLemmatizer 
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, Activation, Dropout 
from tensorflow.python.keras.optimizers import SGD 
 

 # Código de preparación de datos 

lemmatizer =WordNetLemmatizer 
intents=json.load (open ("intents.json").read () ) 
 
word=[] 

classes= [] 
 
documents=[] 
ignorar_letas=["?"," i","","!",".",","] 
for intent in intents ["intents" ] :
    for pattern in intent ["patterns"] : ##Por cada intento busca una pregunta 
        word_list=nltk.word_tokenize (pattern) #tokenize divide las palabras indiv word.append (word list) 
        documents. append((word_list),intent["tag"]) 
        if intent [" tag" ] not in classes : 
            classes.append (intent ["tag"]) 
print (documents) 


# Preprocesamiento de datos
words = [item for sublist in word for item in sublist]
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignorar_letas]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Creación de datos de entrenamiento
training = []
output_empty = [0] * len(classes)
for doc in documents:
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        training.append(int(w in pattern_words))
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append(output_row)

training = np.array(training)
train_x = list(training[:, :len(words)])
train_y = list(training[:, len(words):])

# Entrenamiento del modelo 
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5)
model.save('chatbot_model.h5')