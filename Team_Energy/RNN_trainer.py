# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD

from Team_Energy.data import create_data
from Team_Energy.prepare import prepare_sequences

from sklearn.metrics import mean_absolute_percentage_error
import joblib

class Trainer():
    # Init Function
    def __init__(self, x_train, x_test, y_train, name, tariff):
        self.regressor = None
        self.name = name
        self.tariff = tariff
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train


    def train_model(self, epochs = 50, batch_size = 32):
        # The LSTM architecture
        regressor = Sequential()

        # First LSTM layer with Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1],1)))
        regressor.add(Dropout(0.2))
        # Second LSTM layer

        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        # Third LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        # Fourth LSTM layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.2))

        # The output layer
        regressor.add(Dense(units=1))

        # Compiling the RNN
        regressor.compile(optimizer='rmsprop',loss='MeanAbsolutePercentageError')

        # Fitting to the training set
        regressor.fit(self.x_train, self.y_train,epochs=epochs,batch_size=batch_size)

        # Attaching regressor to self
        self.regressor = regressor

    def save_model(self):
        joblib.dump(self.regressor, f'RNNmodel_{self.name}_{self.tariff}.joblib')

if __name__ == "__main__":

    # Group creation
    print('Select Acorn Group')
    name = input()
    print('Select tariff')
    tariff = input()

    print('Fetching data ...')

    # Get Data
    train_df, test_df, val_df = create_data(name, tariff)

    print('data imported successfully')

    print('preparing sequences')
    # Prepare sequences
    x_train,y_train,x_test,sc, test_set = prepare_sequences(train_df, test_df, val_df)
    print('sequence preparation complete')
    print('Now training model...')
    # Train
    trainer = Trainer(x_train, x_test, y_train, name = name, tariff = tariff)
    trainer.train_model()
    print('Model trained successfully')

    # Save
    print('Now saving model')
    trainer.save_model()
    print('Model saved successfully')
