import keras_tuner as kt
from keras.src.callbacks import EarlyStopping
from keras.src.datasets import imdb
from keras.src.utils import pad_sequences
from keras_tuner import Tuner

from tuning.ImdbSentimentHypermodel import ImdbSentimentHyperModel

MAX_EPOCHS = 15
BATCH_SIZE = 32


def __print_best_hyperparameter(tuner: Tuner, hp_name: str) -> None:
    best_hyperparam = tuner.get_best_hyperparameters()[0].get(hp_name)

    print(f"Best value for hp {hp_name} = {best_hyperparam}")


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=ImdbSentimentHyperModel.VOCAB_SIZE)
    x_train = pad_sequences(sequences=x_train, maxlen=ImdbSentimentHyperModel.MAX_SENT_SIZE)
    x_test = pad_sequences(sequences=x_test, maxlen=ImdbSentimentHyperModel.MAX_SENT_SIZE)

    hypermodel = ImdbSentimentHyperModel()
    tuner = kt.GridSearch(
        hypermodel=hypermodel,
        objective=kt.Objective(name='val_accuracy', direction='max'),
        project_name='data'
    )

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    tuner.search(
        x=x_train,
        y=y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping_callback]
    )

    __print_best_hyperparameter(tuner=tuner, hp_name=ImdbSentimentHyperModel.HP_EMBEDDING_SIZE_NAME)
    __print_best_hyperparameter(tuner=tuner, hp_name=ImdbSentimentHyperModel.HP_RNN_UNITS_NAME)
    __print_best_hyperparameter(tuner=tuner, hp_name=ImdbSentimentHyperModel.HP_LOSS_FUNCTION_NAME)
