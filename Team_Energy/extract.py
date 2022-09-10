import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np

print('###########################')
print('Extract ACORN data by group')
print('###########################')
print('Please input ACORN group for extraction')
groupletter = input()
print('Please enter the minimum number of days of data for houses')
mindaydata = input()
print('Please enter maximum acceptable number of missing data')
minmissdata = input()

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
                & (df_summary['number_of_days'] >= int(mindaydata))
                & (df_summary['zeros'] + df_summary['missing_hh'] <= int(minmissdata))]

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

path = f'../raw_data/halfhourly_dataset/{blocklist[0]}.csv'

df_house = pd.read_csv(path, dtype = {'LCLid': object, 'tstp': object , 'energy(kWh/hh)': object})
for i in range (0,int(len(houselist))):
    print(f'Now adding house {i+1} of {len(houselist)}, {houselist[i]} to the dataset...')
    blockstr = f'../raw_data/halfhourly_dataset/{blocklist[i]}.csv'
    if blocklist[i] != blocklist[i-1]:
        df_house = pd.read_csv(blockstr, dtype = {'LCLid': object, 'tstp': object , 'energy(kWh/hh)': object})
    df2_house = df_house[df_house['LCLid'] == houselist[i]]
    df2_house['tariff'] = tarifflist[i]
    LCL.extend(df2_house['LCLid'])
    DateTime.extend(df2_house['tstp'])
    KWH.extend(df2_house['energy(kWh/hh)'])
    tariff.extend(df2_house['tariff'])

# Creating a new Data Frame with lists

block_data = pd.DataFrame({'LCLid': LCL, 'DateTime': DateTime, 'KWH/hh': KWH, 'tariff': tariff})

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

# Fill 0 with nans
block_data['KWH/hh'].replace(0,np.nan,inplace=True)

# Fill nan's using a back fill
date_range = pd.DataFrame(pd.date_range(block_data.index[0],block_data.index[-1], freq='30 min'),columns=['DateTime'])
block_data = date_range.merge(block_data,on='DateTime',how='outer')
if np.sum(block_data['KWH/hh'].isna())!=0:
    block_data.fillna(method='bfill',inplace=True)

# Restrict date range

block_data = block_data[block_data['DateTime'] >= '2012-01-01']

# Identify and remove outliers

u = np.mean(block_data['KWH/hh'])
q3 = np.quantile(block_data['KWH/hh'], 0.75)
q1 = np.quantile(block_data['KWH/hh'], 0.25)
IQR = q3-q1
ub = q3 + 1.5 * IQR
lb = q1 - 1.5 * IQR
block_data[(block_data['KWH/hh'] <= ub) & (block_data['KWH/hh'] >= lb)]

# Ask to separate data by tariff
print('tariff choice bypassed')
# print('Separate data by tariff? Y/N')
# tariffchoice = input()
# if tariffchoice not in ['Y', 'N']:
#     print('Invalid selection')
#     exit()

if 1 == 1:
    # Create average time
    block_data_avg=block_data.groupby(by=block_data.DateTime).mean()
    block_data_avg.sort_index(inplace=True)
    block_data_avg.reset_index(inplace = True)
    block_data_avg['tariff'] = 'Mixed'

    # Save and Export dataset to CSV
    filename = f'df_{groupletter}_avg_v1.csv'

    block_data_avg.to_csv(f'../raw_data/{filename}')
    # print('Process Complete')
    # quit()

if 1 == 1:
    tariffblock = block_data[block_data['tariff'] == 'ToU']
    stdblock = block_data[block_data['tariff'] == 'Std']

    # Create average time
    tariff_data_avg=tariffblock.groupby(by=tariffblock.DateTime).mean()
    tariff_data_avg.sort_index(inplace=True)
    tariff_data_avg.reset_index(inplace = True)
    tariff_data_avg['tariff'] = 'ToU'
    std_data_avg=stdblock.groupby(by=stdblock.DateTime).mean()
    std_data_avg.sort_index(inplace=True)
    std_data_avg.reset_index(inplace = True)
    std_data_avg['tariff'] = 'Std'

    # Save and Export datasets to CSV
    filename = f'df_{groupletter}_avg_ToU_v1.csv'
    tariff_data_avg.to_csv(f'../raw_data/{filename}')
    filename = f'df_{groupletter}_avg_Std_v1.csv'
    std_data_avg.to_csv(f'../raw_data/{filename}')

    print('Process Complete')
    quit()
