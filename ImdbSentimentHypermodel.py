import keras_tuner as kt
from keras import Sequential, Model
from keras.src.layers import InputLayer, Embedding, SimpleRNN, Dense
from keras_tuner import HyperParameters


class ImdbSentimentHyperModel(kt.HyperModel):
    MAX_LEN = 500
    VOCAB_SIZE = 10000

    HP_EMBEDDING_SIZE_NAME = 'embedding_size'
    HP_RNN_UNITS_NAME = 'rnn_units'
    HP_LOSS_FUNCTION_NAME = 'loss_function'

    def build(self, hp: HyperParameters) -> Model:
        model = Sequential()
        embedding_size = ImdbSentimentHyperModel.__hp_embedding_size(hp)
        rnn_units = ImdbSentimentHyperModel.__hp_rnn_units(hp)
        loss_function = ImdbSentimentHyperModel.__hp_loss_function(hp)

        model.add(InputLayer(shape=(self.MAX_LEN,)))
        model.add(Embedding(input_dim=self.VOCAB_SIZE, output_dim=embedding_size))
        model.add(SimpleRNN(units=rnn_units, activation='tanh'))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])

        return model

    @staticmethod
    def __hp_embedding_size(hp: HyperParameters) -> int:
        return hp.Choice(name=ImdbSentimentHyperModel.HP_EMBEDDING_SIZE_NAME, values=[64, 128, 192, 256])

    @staticmethod
    def __hp_rnn_units(hp: HyperParameters) -> int:
        return hp.Choice(name=ImdbSentimentHyperModel.HP_RNN_UNITS_NAME, values=[64, 128, 192, 256])

    @staticmethod
    def __hp_loss_function(hp: HyperParameters) -> str:
        return hp.Choice(
            name=ImdbSentimentHyperModel.HP_LOSS_FUNCTION_NAME,
            values=[
                'binary_crossentropy',
                'binary_focal_crossentropy'
            ]
        )
