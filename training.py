import json
import pickle
import random

import nltk
import numpy as np
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # mentokenisasi setiap kata
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # menambahkan documents kedalam corpus
        documents.append((w, intent['tag']))

        # menambahkan ke list classes kita
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize dan ubah menjadi huruf kecil dan membuat kata yang duplikat
words = [lemmatizer.lemmatize(w.lower())
         for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# mensortir classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)


pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# membuat training data
training = []
# membuat array kosong untuk output
output_empty = [0] * len(classes)
# training set, bag of words untuk setiap kalimat
for doc in documents:
    # menginsialisasi bag of words
    bag = []
    # list dari tokenisasi kata untuk setiap pola
    pattern_words = doc[0]
    # lemmatize setiap kata - buat kata dasar, agar menjadi "key" bagi bentuk kata ubahannya
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]
    # membuat bag of words array dengan 1, jika kecocokan kata ditemukan dalam pola saat ini
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output '0' untuk setiap 'tag' dan '1' untuk 'tag' saat ini (untuk setiap pola)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# acak semua feature dan ubah kedalam bentuk np.array
random.shuffle(training)
training = np.array(training)
# buat list train dan test. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")


# buat model - 3 layers. Layer pertama 128 neurons, 
# layer kedua 64 neurons 
# Layer ke-3 output layer mengandung jumlah neurons
# sama dengan jumlah maksud untuk memprediksi maksud keluaran dengan softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Penurunan gradien stokastik dengan gradien 
# akselerasi Nesterov memberikan hasil yang baik untuk model ini
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

# fitting dan simpan model
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=200, batch_size=5, verbose=1)
model.save('model.h5', hist)

print("model created")
