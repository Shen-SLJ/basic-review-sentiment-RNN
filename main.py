from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.datasets import imdb
from keras.src.layers import Embedding, SimpleRNN, Dense, InputLayer
from keras.src.utils import pad_sequences

if __name__ == '__main__':
    max_len = 500
    vocab_size = 10000
    embedding_size = 128
    batch_size = 32
    epochs = 25

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(sequences=x_train, maxlen=max_len)
    x_test = pad_sequences(sequences=x_test, maxlen=max_len)

    model = Sequential()
    model.add(InputLayer(input_shape=(max_len,)))
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size))
    model.add(SimpleRNN(units=embedding_size, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping]
    )

    model.save("model.keras")