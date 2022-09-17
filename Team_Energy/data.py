import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Prepare sequences
def prepare_sequences(train_df,test_df, val_df = None):
    train_df.set_index('DateTime',inplace=True)
    test_df.set_index('DateTime',inplace=True)
    val_df.set_index('DateTime',inplace=True)
    training_set = train_df[:'2013'].loc[:,['KWH/hh']].values
    test_set = test_df['2014':].loc[:,['KWH/hh']].values

    # Scaling the training set
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)

    #Since LSTMs store long term memory state, we create a data structure with 60 timesteps and 1 output
    # So for each element of training set, we have 60 previous training set elements
    X_train = []
    y_train = []

    for i in range(60,len(training_set_scaled)):
        X_train.append(training_set_scaled[i-60:i,0])
        y_train.append(training_set_scaled[i,0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

    # Now to get the test set ready in a similar way as the training set.
    # The following has been done so first 60 entries of test set have 60 previous values which is impossible to get
    dataset_total = pd.concat((train_df["KWH/hh"][:'2013'],test_df["KWH/hh"]['2014':]),axis=0)
    inputs = dataset_total[len(dataset_total)-len(test_set) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = sc.transform(inputs)

    # Preparing X_test and predicting the prices
    X_test = []
    for i in range(60,len(inputs)):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    return  X_train, y_train, X_test, sc, test_set



# Data - Paritition
def split_data(filename, tariff):
    fulldata = pd.read_csv(filename)
    fulldata['DateTime'] = pd.to_datetime(fulldata['DateTime'])

    if tariff == 'Tou':
        start_date = '2013-01-01'
    else:
        start_date = '2012 -01-01'

    train_data = fulldata[(fulldata['DateTime'] >= start_date) & (fulldata['DateTime'] < '2014-01-01')].reset_index(drop = True)
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
        vdata.drop(columns='Unnamed: 0',inplace=True)
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

    # Group By
    df5=pdata.loc[:,['DateTime','KWH/hh']]
    df5.set_index('DateTime',inplace=True)

    train_df=df5.groupby(by=df5.index).mean()
    train_df=train_df.reset_index()
    test_df=testd.loc[:,['DateTime','KWH/hh']].groupby(by=testd['DateTime']).mean()
    test_df.reset_index(inplace=True)
    val_df=vdata.loc[:,['DateTime','KWH/hh']].groupby(by=vdata['DateTime']).mean()
    val_df.reset_index(inplace=True)

    return train_df,test_df,val_df












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
