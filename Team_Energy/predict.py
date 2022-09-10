# Imports
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import joblib
from Team_Energy.data import split_data, create_data, get_holidays, get_weather
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

print('input name')
name = input()
print('input tariff: Std or ToU')
tariff = input()

# Joblib import model
filename = f'model_{name}_{tariff}.joblib'
m = joblib.load(filename)
print('model loaded succcessfully')


# Predict
def forecast_model(m,train_wd,test_wd,add_weather=False):
    future = m.make_future_dataframe(periods=48*27+1, freq='30T')
    if add_weather==True:
        wd_filt_future=future[['ds']].merge(pd.concat([future,pd.concat([train_wd,test_wd],axis=0)]),left_on='ds',right_on='DateTime',how='inner').drop(columns='DateTime')
        temp_future=wd_filt_future['temperature'].interpolate(method='linear')
        future['temp']=temp_future
        fcst = m.predict(future)
    else:
        fcst = m.predict(future)
    forecast=fcst.loc[fcst['ds']>='2014-02-01 00:00:00',['ds','yhat']]
    return forecast

# Evaluate model
def evaluate(actual,forecast):
    return np.round(mean_absolute_percentage_error(actual,forecast),4)

# Plot
def plot_graphs(test_df, forecast):
    figure(figsize=(15,6))
    sns.lineplot(x=forecast['ds'],y=forecast['yhat'],label='Forecast');
    sns.lineplot(x=test_df['DateTime'],y=test_df['KWH/hh'],label='Actual');
    figure(figsize=(15,6))
    sns.lineplot(x=test_wd['DateTime'],y=test_wd['temperature'],label='Weather');


if __name__ == "__main__":
    # define df's using data.py

    train_df, test_df = create_data(name = name, tariff = tariff)
    train_wd, test_wd = get_weather(train_df, test_df)
    print('dataframes loaded')
    # Calculate forecast and MAPE
    forecast = forecast_model(m=m, train_wd = train_wd, test_wd = test_wd, add_weather = False)
    print('forecast made')
    mape = evaluate(test_df['KWH/hh'], forecast['yhat'])
    # Print MAPE
    print('mape is:')
    print(mape)

    # Plot the graphs
    print('now plotting')
    plot_graphs(test_df = test_df, forecast= forecast)
    print('operation complete')
