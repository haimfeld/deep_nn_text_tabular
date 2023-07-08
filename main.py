import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

train_df=pd.read_csv("../input/hotel-review/train.csv")
test_df=pd.read_csv("../input/hotel-review/test.csv")

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train_df["Is_Response"] = labelencoder.fit_transform(train_df["Is_Response"])
#1 not happy, 0 happy

train_df["Device_Used"] = labelencoder.fit_transform(train_df["Device_Used"])

train_df["Browser_Used"] = labelencoder.fit_transform(train_df["Browser_Used"])


reviews=train_df['Description'].to_numpy()
devices = train_df["Device_Used"].to_numpy()
ratings = train_df["Browser_Used"].to_numpy()
sentiments = train_df["Is_Response"].to_numpy()

reviews_train, reviews_test, ratings_train, ratings_test, devices_train, devices_test, sentiments_train, sentiments_test = train_test_split(
    reviews, ratings,devices, sentiments, test_size=0.2, random_state=42
)

max_words = 1000
max_sequence_length = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(reviews_train)
sequences_train = tokenizer.texts_to_sequences(reviews_train)
sequences_test = tokenizer.texts_to_sequences(reviews_test)
word_index = tokenizer.word_index


X_train = pad_sequences(sequences_train, maxlen=max_sequence_length)
X_test = pad_sequences(sequences_test, maxlen=max_sequence_length)


ratings_min = ratings.min()
ratings_max = ratings.max()
ratings_train_normalized = (ratings_train - ratings_min) / (ratings_max - ratings_min)
ratings_test_normalized = (ratings_test - ratings_min) / (ratings_max - ratings_min)


devices_min = devices.min()
devices_max = devices.max()
devices_train_normalized = (devices_train - devices_min) / (devices_max - devices_min)
devices_test_normalized = (devices_test - devices_min) / (devices_max - devices_min)

embedding_dim = 100
lstm_units = 128
max_sequence_length = 100
max_words = 10000

text_input = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(max_words, embedding_dim)(text_input)
lstm_layer = LSTM(lstm_units)(embedding_layer)


ratings_input = Input(shape=(1,))
devices_input = Input(shape=(1,))
numerical_dense = Dense(128, activation='relu')(ratings_input)


merged = concatenate([lstm_layer, numerical_dense, devices_input])

dense_layer = Dense(64, activation='relu')(merged)

layer = Dense(48, activation='relu')(dense_layer)

new_dense_layer = Dense(32, activation='relu')(layer)

layer_ = Dense(16, activation='relu')(new_dense_layer)

output = Dense(2, activation='softmax')(layer_)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model = Model(inputs=[text_input, ratings_input, devices_input], outputs=output)


model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([X_train, ratings_train_normalized,devices_train_normalized], sentiments_train, epochs=3, batch_size=32, verbose=1, callbacks=[early_stopping])


loss, accuracy = model.evaluate([X_test, ratings_test_normalized,devices_test_normalized], sentiments_test, verbose=0)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy*100:.2f}%")

