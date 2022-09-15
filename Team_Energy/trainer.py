# ** Imports **
import pandas as pd
import numpy as np
import seaborn as sns
import os
import itertools
from matplotlib import pyplot as plt
import joblib
from Team_Energy.data import split_data, create_data, get_holidays, get_weather
from sklearn.metrics import mean_absolute_percentage_error


#Facebook Prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics


class Trainer ():

    # Init Function
    def __init__(self, train_df, name, tariff):
        self.m = None
        self.train_df = train_df
        self.name = name
        self.tariff = tariff

    # Train
    def train_model(self, train_wd, holidays,add_weather=True):
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

    # Hyper-parameter Tuning
    def tune_model(self):

        # Set grid
        param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        mapes = []  # Store the MAPE's for each params here

        # Rename columns for accurate cv
        self.train_df = self.train_df.rename(columns = {"DateTime": "ds", "KWH/hh": "y"})
        print(self.train_df.keys())

        # Use cross validation to evaluate all parameters
        for params in all_params:
            m = Prophet(**params).fit(self.train_df)  # Fit model with given params
            df_cv = cross_validation(m, horizon='30 days', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            mapes.append(df_p['mape'].values[0])

        # Finding the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['mape'] = mapes

        # Python
        best_params = all_params[np.argmin(mapes)]
        print(best_params)


    # Predict
    def forecast_model(self,train_wd,test_wd,add_weather=True):
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

if __name__ == "__main__":
    groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']
    print('input group')
    group = input()
#for name in groups:
    print(f'Now modelling {group}')
    tariff = 'ToU'

    print('starting process')

    # Get Data
    train_df, test_df = create_data(group, tariff)
    holidays = get_holidays()
    train_wd, test_wd = get_weather(train_df, test_df)

    print('data imported successfully')

    # # Train
    # trainer = Trainer(train_df, name = group, tariff = tariff)
    # trainer.train_model(train_wd = train_wd, holidays = holidays)

    # print('model trained successfully')

    # # Evaluate (MAPE)
    # # evaluation = trainer.evaluate()
    # # print(evaluation)

    # # Save
    # trainer.save_model()
    # print('model saved successfully')

    # Grid search
    print('now tuning model')
    trainer = Trainer(train_df, name = group, tariff = tariff)
    trainer.tune_model()
    print('process complete')
