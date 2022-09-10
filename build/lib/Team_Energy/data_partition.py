import pandas as pd
import numpy as np

def split_data():
    print('Input precise path for data including extension')
    filename = input()
    fulldata = pd.read_csv(f'{filename}')
    fulldata['DateTime'] = pd.to_datetime(fulldata['DateTime'])
    train_data = fulldata[(fulldata['DateTime'] >= '2012-01-01') & (fulldata['DateTime'] < '2014-01-01')]
    validation_data = fulldata[(fulldata['DateTime'] >= '2014-01-01') & (fulldata['DateTime'] < '2014-02-01')]
    test_data = fulldata[(fulldata['DateTime'] >= '2012-02-01') & (fulldata['DateTime'] < '2014-03-01')]

    return train_data, validation_data, test_data
