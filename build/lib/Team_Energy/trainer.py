# ** Imports **
import pandas as pd
import numpy as np
import seaborn as sns
import os
from matplotlib import pyplot as plt
import joblib
from Team_Energy.data import split_data, create_data, get_holidays, get_weather
from sklearn.metrics import mean_absolute_percentage_error

#Facebook Prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

class Trainer ():

    # Init Function
    def __init__(self, train_df, name, tariff):
        self.m = None
        self.train_df = train_df
        self.name = name
        self.tariff = tariff

    # Train
    def train_model(self, train_wd, holidays,add_weather=False):
        self.train_df.rename(columns={"DateTime": "ds", "KWH/hh": "y"},inplace=True)
        if add_weather==True:
            temp = train_wd['temperature'].interpolate(method='linear')
            self.train_df['temp']=temp
            m = Prophet(holidays=holidays,changepoint_prior_scale=0.01).add_regressor('temp', prior_scale=0.5, mode='multiplicative')
            m.fit(self.train_df)
        else:
            m = Prophet(holidays=holidays,changepoint_prior_scale=0.01)
            m.fit(self.train_df)
        self.m = m

    # Predict
    def forecast_model(self,train_wd,test_wd,add_weather=False):
        future = self.m.make_future_dataframe(periods=48*27+1, freq='30T')
        if add_weather==True:
            wd_filt_future=future[['ds']].merge(pd.concat([future,pd.concat([train_wd,test_wd],axis=0)]),left_on='ds',right_on='DateTime',how='inner').drop(columns='DateTime')
            temp_future=wd_filt_future['temperature'].interpolate(method='linear')
            future['temp']=temp_future
            fcst = self.m.predict(future)
        else:
            fcst = self.m.predict(future)
        self.fcst = fcst
        forecast=fcst.loc[fcst['ds']>='2014-02-01 00:00:00',['ds','yhat']]
        return forecast

    # Evaluate model
    def evaluate(actual,forecasted):
        return np.round(mean_absolute_percentage_error(actual,forecasted),4)

    # Save the model
    def save_model(self):
        joblib.dump(self.m, f'model_{self.name}_{self.tariff}.joblib')

if __name__ == "main":

    name = 'A'
    tariff = 'Std'

    # Get Data
    train_df, test_df = create_data(name, tariff)
    holidays = get_holidays()
    train_wd, test_wd = get_weather()

    # Train
    trainer = Trainer(train_df, name = name, tariff = tariff)
    trainer.train_model()

    # Evaluate (MAPE)
    # evaluation = trainer.evaluate()
    # print(evaluation)

    # Save
    trainer.save_model()
