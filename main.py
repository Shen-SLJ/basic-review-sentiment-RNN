from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.datasets import imdb
from keras.src.layers import Embedding, SimpleRNN, Dense, InputLayer
from keras.src.utils import pad_sequences

MAX_SENT_SIZE = 500
VOCAB_SIZE = 10000
EMBEDDING_SIZE = 64
RNN_UNITS = 16
BATCH_SIZE = 32
EPOCHS = 30
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'binary_crossentropy'

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    x_train = pad_sequences(sequences=x_train, maxlen=MAX_SENT_SIZE)
    x_test = pad_sequences(sequences=x_test, maxlen=MAX_SENT_SIZE)

    model = Sequential()
    model.add(InputLayer(shape=(MAX_SENT_SIZE,)))
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_SIZE))
    model.add(SimpleRNN(units=RNN_UNITS, activation='tanh'))
    model.add(Dense(units=1, activation='sigmoid'))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=['accuracy'])

    model.fit(
        x=x_train,
        y=y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping]
    )

    model.save("model.keras")
