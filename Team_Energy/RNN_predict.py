# Imports
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import joblib
from Team_Energy.data import create_data
from Team_Energy.prepare import prepare_sequences
import seaborn as sns
import matplotlib.pyplot as plt


print('input name')
name = input()
print('input tariff: Std or ToU')
tariff = input()

# Joblib import model
filename = f'RNNmodel_{name}_{tariff}.joblib'
m = joblib.load(filename)
print('model loaded succcessfully')


# Predict
def forecast_model(m,X_test,sc):
    predicted_consumption = m.predict(X_test)
    predicted_consumption = sc.inverse_transform(predicted_consumption)
    return predicted_consumption

# Evaluate model
def evaluate(test_set,predicted_consumption):
    mape = mean_absolute_percentage_error(test_set,predicted_consumption)
    print("The  mean absolute percenatge error is {}.".format(mape))
    return mape

# Plot
def plot_graphs(test_set,predicted_consumption):
    df_plot=test_df['2014':].copy()
    df_plot.rename(columns={'KWH/hh':'Test'},inplace=True)
    df_plot['Predicted']=predicted_consumption
    plt.figure(figsize=(18,6))
    plt.plot(df_plot)
    plt.title('Electricity Consumption Prediction')
    plt.xlabel('Time')
    plt.ylabel('Consumption (kWh/hh)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # define df's using data.py

    train_df, test_df,val_df = create_data(name,tariff)
    X_train, y_train, X_test, sc, test_set = prepare_sequences(train_df, test_df,val_df)

    print('dataframes loaded')
    # Calculate forecast and MAPE
    predicted_consumption = forecast_model(m,X_test,sc)
    print('forecast made')
    mape = evaluate(test_set,predicted_consumption)

    # Print MAPE
    print('mape is:')
    print(mape)

    # Plot the graphs
    print('now plotting')
    plot_graphs(test_df, predicted_consumption)
    print('operation complete')
