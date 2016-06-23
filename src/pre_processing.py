'''
USP - ICMC
Kaggle: Shelter animal outcomes
Authors: Rafael
         Thomio Watanabe
'''

import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import normalize
from matplotlib.ticker import MaxNLocator
#import matplotlib.pyplot as plt


# Set max line output display
# pd.set_option('display.max_rows', 2000)

def parse_age(row):
    age = row
    result = re.search(r'([0-9]+)[^a-z]*([a-z]+)', age)
    if result:
        quantity = float(result.group(1))
        time_unit = result.group(2)
        if time_unit == 'week' or time_unit == 'weeks':
            quantity = quantity * 7
        elif time_unit == 'month' or time_unit == 'months':
            quantity = quantity * 30
        elif time_unit == 'year' or time_unit == 'years':
            quantity = quantity * 365
        row = quantity
    else:
        row = 99999
    return row


df = pd.read_csv('../data/train.csv.gz', compression='gzip')

# Database has 26729 examples in 10 categorical attributes
print 'Data shape: ', df.shape
print 'First elements: '
print df.head()
# print df.describe()


# Converting categorial attributes to numeric
# 7 Categorical attributes
# Categorial without order -> generate dummy attributes
name = pd.get_dummies(df.Name, dummy_na=False)
print np.sort( np.sum(name == 1) ).size
#    6375 names
#    NaN 7691

outcometype = pd.get_dummies(df.OutcomeType, dummy_na=False)
print np.sum(outcometype == 1)
#    Adoption           10769
#    Died                 197
#    Euthanasia          1555
#    Return_to_owner     4786
#    Transfer            9422
#    NaN                    0

outcomesubtype = pd.get_dummies(df.OutcomeSubtype, dummy_na=False)
print np.sum(outcomesubtype == 1)
#    Aggressive               320
#    At Vet                     4
#    Barn                       2
#    Behavior                  86
#    Court/Investigation        6
#    Enroute                    8
#    Foster                  1800
#    In Foster                 52
#    In Kennel                114
#    In Surgery                 3
#    Medical                   66
#    Offsite                  165
#    Partner                 7816
#    Rabies Risk               74
#    SCRP                    1599
#    Suffering               1002
#    NaN                    13612

animaltype = pd.get_dummies(df.AnimalType, dummy_na=False)
print np.sum(animaltype == 1)
#    Cat    11134
#    Dog    15595
#    NaN        0

sexuponoutcome = pd.get_dummies(df.SexuponOutcome, dummy_na=False)
print np.sum(sexuponoutcome == 1)
#    Intact Female    3511
#    Intact Male      3525
#    Neutered Male    9779
#    Spayed Female    8820
#    Unknown          1093
#    NaN                 1

breed = pd.get_dummies(df.Breed, dummy_na=False)
print np.sum( breed == 1 )
#    1381 beeds

color = pd.get_dummies(df.Color, dummy_na=False)
print np.sum( color == 1 )
#    367 colors



# 2 Numerical attributes
# Age in simbolic representation -> convert to integers (days)
ageuponoutcome = pd.get_dummies(df.AgeuponOutcome, dummy_na=False)
age = np.sum( ageuponoutcome == 1 )
age.sort(ascending = False)
print age
# 45 periods
#    1 year       3969
#    2 years      3742
#    2 months     3397
#    3 years      1823
#    1 month      1281
#    3 months     1277
#    4 years      1071
#    5 years       992
#    4 months      888
#    6 years       670
#    3 weeks       659
#    5 months      652
#    6 months      588
#    8 years       536
#    7 years       531
#    2 weeks       529
#    10 months     457
#    10 years      446
#    8 months      402
#    4 weeks       334
#    9 years       288
#    7 months      288
#    12 years      234
#    9 months      224
#    1 weeks       171
#    11 months     166
#    1 week        146
#    13 years      143
#    11 years      126
#    3 days        109
#    2 days         99
#    14 years       97
#    15 years       85
#    1 day          66
#    6 days         50
#    4 days         50
#    16 years       36
#    5 days         24
#    0 years        22
#    NaN            18 -> NA
#    17 years       17
#    5 weeks        11
#    18 years       10
#    19 years        3
#    20 years        2

# Plot age.
stay_animals = np.ndarray( [len(age), 2], dtype=int )
for i in range( len(age) ):
    stay_animals[i, 0] = parse_age(age.index.values[i])
    stay_animals[i, 1] = age[i]

indices = np.argsort(stay_animals[:,0])
stay_animals[:,0] = stay_animals[indices,0]
stay_animals[:,1] = stay_animals[indices,1]

#    plt.figure( facecolor='white' )
#    plt.subplot(121)
#    plt.plot(stay_animals[:,0], stay_animals[:,1], 'bo-')
#    plt.title('Animal stay')
#    plt.ylabel('Number of animals')
#    plt.xlabel('Period of time (days)')
#    plt.grid(True)

#    plt.subplot(122)
#    plt.semilogx(stay_animals[:,0],stay_animals[:,1], 'bo-')
#    plt.title('Animal stay (semilog)')
#    plt.ylabel('Number of animals')
#    plt.xlabel('Period of time (days)')
#    plt.grid(True)
#    #plt.savefig('animal_stay.png')
#    plt.show()

stay_intervals = np.array_split( stay_animals, 8 )


bins = np.zeros(len(stay_intervals) + 1)
for i in range( 1, len(stay_intervals ) ):
    bins[i] = ( ( min( stay_intervals[i][:,0] ) + max( stay_intervals[i-1][:,0] ) ) / 2. )

bins[len(stay_intervals)] = ( max( stay_intervals[ len(stay_intervals) - 1 ][:,0] ) * 1.05 )


# Separate the age in bins and keep the original order
# Deal with NaN -> 0 days
ageuponoutcome = df.AgeuponOutcome.fillna(0)
days = []
for i in range( len(ageuponoutcome) ):
    if ageuponoutcome[i] == 0:
        days.append(0)
    else:
        days.append(parse_age(ageuponoutcome[i]) )

#ageuponoutcome = pd.DataFrame( days )
ageuponoutcome = pd.Series( days )
ageuponoutcome[ 0:len(ageuponoutcome) ] = np.digitize( ageuponoutcome, bins )

ageuponoutcome = pd.DataFrame( ageuponoutcome, columns=['AgeuponOutcome'] )




# df.DateTime
# Period: 2013 - 2016 (max interval)
date_time = np.ndarray( [2,len(df.DateTime)] )
for i in range( len( df.DateTime ) ):
    # split date and time
    dt = datetime.strptime(df.DateTime[i], '%Y-%m-%d %H:%M:%S')
    # store months
    date_time[0,i] = (dt.year - 2013) * 12 + dt.month + np.round( dt.day / 30. )
    # store hours
    date_time[1,i] = dt.hour + np.round( dt.minute / 60. )

# date mean = 24.6 months
# hour mean = 14.89 hour
# Standardize month and hour vectors
date_time[0] = ( date_time[0] - np.mean( date_time[0] ) ) / np.std( date_time[0] )
date_time[1] = ( date_time[1] - np.mean( date_time[1] ) ) / np.std( date_time[1] )
# plt.plot(date_time[0]), plt.show()
# plt.plot(date_time[1]), plt.show()

data_matrix = {'Date (months)' : pd.Series( date_time[0] ), 'Hour' : pd.Series( date_time[1] )}
DateTime = pd.DataFrame( data_matrix )


# -----------------------------------------------------------------------------------
# Attribute selection
# -----------------------------------------------------------------------------------

def select_main_samples( samples, num_samples ):
    samples_count = samples.sum()
    samples_sel = samples_count[samples_count > num_samples]
    samples_sel.sort(ascending = False)
    print samples_sel
    print 'Array size: ', samples_sel.size
    main_samples = samples_sel.index.values
    return main_samples


# Select main names (names that are more frequent/more significant)
main_names = select_main_samples( name, 40 ) # 40 -> 35
name[main_names]
# Top 10 names
#    Max           136
#    Bella         135
#    Charlie       107
#    Daisy         106
#    Lucy           94
#    Buddy          87
#    Princess       86
#    Rocky          85
#    Luna           68
#    Jack           66



# Get the most significant breeds
main_breeds = select_main_samples( breed, 25 ) # 25 -> 85
breed[main_breeds]
# Top 10 breeds from 181 (x=10)
#    Domestic Shorthair Mix                      8810
#    Pit Bull Mix                                1906
#    Chihuahua Shorthair Mix                     1766
#    Labrador Retriever Mix                      1363
#    Domestic Medium Hair Mix                     839
#    German Shepherd Mix                          575
#    Domestic Longhair Mix                        520
#    Siamese Mix                                  389
#    Australian Cattle Dog Mix                    367
#    Dachshund Mix                                318

# Agroup breeds by similarity
# breed.columns.values


# Get the most significant colors
main_colors = select_main_samples( color, 30 ) # 30 -> 79
color[main_colors]
# Top 10 colors
#    Black/White             2824.0
#    Black                   2292.0
#    Brown Tabby             1635.0
#    Brown Tabby/White        940.0
#    White                    931.0
#    Brown/White              884.0
#    Orange Tabby             841.0
#    Tan/White                773.0
#    Tricolor                 752.0
#    Blue/White               702.0


# -----------------------------------------------------------------------------------
# Concatenate attributes in a single table
# number of samples = 26728
# 25, 30, 40 -> 214 attributes

result = pd.concat([outcometype,
                    name[main_names],
                    DateTime,
                    animaltype,
                    sexuponoutcome,
                    ageuponoutcome,
                    breed[main_breeds],
                    color[main_colors]], axis = 1)

output_file = '../output/filtered_dataset.csv'
print 'Saving result table at: ', output_file
result.to_csv( output_file )


# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------


