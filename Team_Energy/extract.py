import pandas as pd
import numpy as np


print('Extract ACORN data by group')
print('NB: Parameters are hardcoded')
print('Please input ACORN group for extraction')
groupletter = input()

# Check if groupletter is valid

if groupletter not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']:
    print('Unknown group')
    exit()

# Import Summary Datasheet

df_summary = pd.read_csv('../raw_data/summary_v1_selection.csv')
df_summary = df_summary[['LCLid',
                            'start_date',
                            'end_date',
                            'number_of_days',
                            'data_points',
                            'missing_hh',
                            'zeros',
                            'tariff',
                            'Block',
                            'Acorn',
                            'Group',
                            'Classification']]

# Reduce selection to chosen group which has been selected

string = f'ACORN-{groupletter}'
df_hot = df_summary[(df_summary['Acorn'] == string)
                & (df_summary['number_of_days'] >= 365)
                & (df_summary['zeros'] + df_summary['missing_hh'] <= 48)]

# Create a list of houses matching criteria

houselist = df_hot['LCLid'].tolist()

# Create a second list of the blocks where the data for each house is found

blocklist = df_hot['Block'].tolist()

# Create a third list of the tariff type

tarifflist = df_hot['tariff'].tolist()

# Iterable loop concactinating the data into lists
LCL = []
Acorn_Group = []
DateTime = []
KWH = []
tariff = []

for i in range (0,int(len(houselist))):
    print(f'Now adding house {i+1} of {len(houselist)}, {houselist[i]} to the dataset...')
    blockstr = f'../raw_data/halfhourly_dataset/{blocklist[i]}.csv'
    df_house = pd.read_csv(blockstr, dtype = {'LCLid': object, 'tstp': object , 'energy(kWh/hh)': object})
    df_house = df_house[df_house['LCLid'] == houselist[i]]
    df_house['tariff'] = tarifflist[i]
    LCL.extend(df_house['LCLid'])
    DateTime.extend(df_house['tstp'])
    KWH.extend(df_house['energy(kWh/hh)'])
    tariff.extend(df_house['tariff'])

# Creating a new Data Frame with lists

block_data = pd.DataFrame({'LCLid': LCL, 'DateTime': DateTime, 'KWH/hh': KWH, 'Tariff': tariff})

# Remove Null values

block_data = block_data[block_data['KWH/hh'] != 'Null']

# Convert KWH/hh to numeric

block_data['KWH/hh'] = pd.to_numeric(block_data['KWH/hh'])

# Convert DateTime to DateTime format

block_data['DateTime'] = pd.to_datetime(block_data['DateTime'])

# Create ACORN group label

block_data['Acorn_Group'] = groupletter
block_data['Group'] = df_hot['Group'].iloc[0]
block_data['Classification'] = df_hot['Classification'].iloc[0]

# Save and Export dataset to CSV
filename = f'df_{groupletter}_v1.csv'

block_data.to_csv(f'../raw_data/{filename}')
print('Process Complete')
quit()
