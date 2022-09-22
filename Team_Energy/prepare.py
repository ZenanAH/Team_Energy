import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

<<<<<<< Updated upstream
def prepare_sequences(train_df,test_df,val_df):
=======
def prepare_sequences(train_df,test_df, val_df = None):
>>>>>>> Stashed changes
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
