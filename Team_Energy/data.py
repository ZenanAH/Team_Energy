import pandas as pd
import numpy as np
import joblib

# Data - Paritition
def split_data(filename):
    fulldata = pd.read_csv(filename)
    fulldata['DateTime'] = pd.to_datetime(fulldata['DateTime'])
    train_data = fulldata[(fulldata['DateTime'] >= '2012-01-01') & (fulldata['DateTime'] < '2014-01-01')].reset_index(drop = True)
    validation_data = fulldata[(fulldata['DateTime'] >= '2014-01-01') & (fulldata['DateTime'] < '2014-02-01')].reset_index(drop = True)
    test_data = fulldata[(fulldata['DateTime'] >= '2014-02-01') & (fulldata['DateTime'] < '2014-03-01')].reset_index(drop = True)
    return train_data, validation_data, test_data

# Data-Processing
def create_data(name, tariff):
    tdata, vdata, testd= split_data(f'https://storage.googleapis.com/energy_usage_prediction_903/df_{name}_avg_{tariff}_v1.csv')
    # add val and train for prophet
    pdata=pd.concat([tdata,vdata],axis=0).reset_index(drop=True)

    global_average=False

    if global_average==False:
        # not for global average
        pdata.drop(columns='Unnamed: 0',inplace=True)
        testd.drop(columns='Unnamed: 0',inplace=True)
        train_df=pdata.loc[:,['DateTime','KWH/hh']]
        test_df=testd.loc[:,['DateTime','KWH/hh']]
    else:
        # group all for dumb models
        df5=pdata.loc[:,['DateTime','Acorn_Group','KWH/hh']]
        df5.set_index('DateTime',inplace=True)
        train_df=df5.groupby(by=df5.index).mean()
        train_df=train_df.reset_index()
        test_df=testd.loc[:,['DateTime','Acorn_Group','KWH/hh']].groupby(by=testd['DateTime']).mean()
        test_df.reset_index(inplace=True)
    return train_df,test_df


# ****  Other Parameters ****

# Holidays
def get_holidays():
    holidays=pd.read_csv('https://storage.googleapis.com/energy_consumption_903/uk_bank_holidays.csv')
    holidays.rename(columns={'Type':'holiday','Bank holidays':'ds'},inplace=True)
    holidays.loc[:,'ds']=pd.to_datetime(holidays['ds'],format="%d/%m/%Y")
    return holidays

# Weather

## if using prophet

def get_weather(train_df, test_df):

    twd, vwd, testwd=split_data('https://storage.googleapis.com/weather-data-processed-for-le-wagon/cleaned_weather_hourly_darksky.csv')
    wd=pd.concat([twd,vwd],axis=0).reset_index(drop=True)
    wd_filt=wd[['DateTime','temperature','windSpeed','precipType_rain']].dropna()
    wd_filt['DateTime']=pd.to_datetime(wd_filt['DateTime'])
    test_wd=testwd[['DateTime','temperature','windSpeed','precipType_rain']].dropna()
    test_wd['DateTime']=pd.to_datetime(test_wd['DateTime'])
    # # wind = wd_filt['windSpeed'].interpolate(method='linear')
    # # rain = wd_filt['precipType_rain'].interpolate(method='linear')
    train_wd=train_df[['DateTime']].merge(wd_filt,on='DateTime',how='inner')
    test_wd=test_df[['DateTime']].merge(test_wd,on='DateTime',how='inner')

    return train_wd, test_wd
