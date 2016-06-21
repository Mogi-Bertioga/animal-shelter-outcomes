'''
USP - ICMC
Kaggle: Shelter animal outcomes
Authors: Rafael
         Thomio Watanabe
'''

import re
import pandas as pd
import numpy as np
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
name = pd.get_dummies(df.Name, dummy_na=True)
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

outcomesubtype = pd.get_dummies(df.OutcomeSubtype, dummy_na=True)
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
ageuponoutcome = pd.get_dummies(df.AgeuponOutcome, dummy_na=True)
print np.sum( ageuponoutcome == 1 )
#    0 years        22
#    1 day          66
#    1 month      1281
#    1 week        146
#    1 weeks       171
#    1 year       3969
#    10 months     457
#    10 years      446
#    11 months     166
#    11 years      126
#    12 years      234
#    13 years      143
#    14 years       97
#    15 years       85
#    16 years       36
#    17 years       17
#    18 years       10
#    19 years        3
#    2 days         99
#    2 months     3397
#    2 weeks       529
#    2 years      3742
#    20 years        2
#    3 days        109
#    3 months     1277
#    3 weeks       659
#    3 years      1823
#    4 days         50
#    4 months      888
#    4 weeks       334
#    4 years      1071
#    5 days         24
#    5 months      652
#    5 weeks        11
#    5 years       992
#    6 days         50
#    6 months      588
#    6 years       670
#    7 months      288
#    7 years       531
#    8 months      402
#    8 years       536
#    9 months      224
#    9 years       288
#    NaN            18

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



