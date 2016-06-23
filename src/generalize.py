
'''
USP - ICMC
Kaggle: Shelter animal outcomes
Authors: Thomio Watanabe
 - load test samples and classification models
 - transform dataset
 - generalize test dataset
'''

import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from datetime import datetime



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


# Load test dataset
test_df = pd.read_csv('../data/test.csv.gz', compression='gzip')

# Load models
adoption_model = joblib.load('./randomforest/Adoption.pkl')
died_model = joblib.load('./randomforest/Died.pkl')
euthanasia_model = joblib.load('./randomforest/Euthanasia.pkl')
return_to_owner_model = joblib.load('./randomforest/Return_to_owner.pkl')
transfer_model = joblib.load('./randomforest/Transfer.pkl')


# Transform categorical variables
# load main names, breeds and colors
main_names = np.loadtxt( '../output/main_names.txt', delimiter = '\n', dtype = str )
main_breeds = np.loadtxt( '../output/main_breeds.txt', delimiter = '\n', dtype = str )
main_colors = np.loadtxt( '../output/main_colors.txt', delimiter = '\n', dtype = str )


mask = np.in1d( test_df['Name'], main_names, invert = True )
test_df['Name'].iloc[mask] = 0
names = pd.get_dummies( test_df.Name, dummy_na = False)
names.drop(0, axis = 1, inplace = True)


mask = np.in1d( test_df['Breed'], main_breeds, invert = True )
test_df['Breed'].iloc[mask] = 0
breeds = pd.get_dummies( test_df.Breed, dummy_na = False)
breeds.drop(0, axis = 1, inplace = True)


mask = np.in1d( test_df['Color'], main_colors, invert = True )
test_df['Color'].iloc[mask] = 0
colors = pd.get_dummies( test_df.Color, dummy_na = False)
colors.drop(0, axis = 1, inplace = True)

animaltype = pd.get_dummies(test_df.AnimalType, dummy_na=False)

sexuponoutcome = pd.get_dummies(test_df.SexuponOutcome, dummy_na=False)


# Transform numerical attributes

# Period: 2013 - 2016 (max interval)
date_time = np.ndarray( [2,len(test_df.DateTime)] )
for i in range( len( test_df.DateTime ) ):
    # split date and time
    dt = datetime.strptime(test_df.DateTime[i], '%Y-%m-%d %H:%M:%S')
    # store months
    date_time[0,i] = (dt.year - 2013) * 12 + dt.month + np.round( dt.day / 30. )
    # store hours
    date_time[1,i] = dt.hour + np.round( dt.minute / 60. )

# Standardize month and hour vectors
date_time[0] = ( date_time[0] - np.mean( date_time[0] ) ) / np.std( date_time[0] )
date_time[1] = ( date_time[1] - np.mean( date_time[1] ) ) / np.std( date_time[1] )
# plt.plot(date_time[0]), plt.show()
# plt.plot(date_time[1]), plt.show()

data_matrix = {'Date (months)' : pd.Series( date_time[0] ), 'Hour' : pd.Series( date_time[1] )}
DateTime = pd.DataFrame( data_matrix )



bins = np.loadtxt( '../output/age_bins.txt', delimiter = '\n')

# Separate the age in bins and keep the original order
# Deal with NaN -> 0 days
ageuponoutcome = test_df.AgeuponOutcome.fillna(0)
days = []
for i in range( len(ageuponoutcome) ):
    if ageuponoutcome[i] == 0:
        days.append(0)
    else:
        days.append(parse_age(ageuponoutcome[i]) )

#ageuponoutcome = pd.DataFrame( days )
ageuponoutcome = np.digitize( pd.Series( days ), bins )

ageuponoutcome = pd.DataFrame( ageuponoutcome, columns=['AgeuponOutcome'] )


test_table = pd.concat([names,
                    DateTime,
                    animaltype,
                    sexuponoutcome,
                    ageuponoutcome,
                    breeds,
                    colors], axis = 1)


adoption_prediction = adoption_model.predict_proba( test_table )
transfer_prediction = transfer_model.predict_proba( test_table )
return_to_owner_prediction = return_to_owner_model.predict_proba( test_table )
died_prediction = died_model.predict_proba( test_table )
euthanasia_prediction = euthanasia_model.predict_proba( test_table )

#    adoption_prediction = adoption_model.predict( test_table )
#    died_prediction = died_model.predict( test_table )
#    euthanasia_prediction = euthanasia_model.predict( test_table )
#    return_to_owner_prediction = return_to_owner_model.predict( test_table )
#    transfer_prediction = transfer_model.predict( test_table )


predictions = np.array([adoption_prediction[:,0], died_prediction[:,1],
                        euthanasia_prediction[:,1], return_to_owner_prediction[:,0],
                        transfer_prediction[:,0]], dtype = float)


file_handle = open('../output/predictions.csv', 'ab')
head = ['ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer']
np.savetxt( file_handle, head, delimiter = '\n,', fmt='%s' )
for i in range( len(predictions.T) ):
    row = predictions.T[i]
    save = str(i+1) + ','
    for j in range( len(row) ):
        save = save + str(row[j]) + ','
    save = save[:-1]
    np.savetxt( file_handle, [save], delimiter = '\n', fmt='%s' )

file_handle.close()



#    for i in range( len(test_df) ):
#        if( adoption_prediction[i,0] > return_to_owner_prediction[i,0] ):
#            if( adoption_prediction[i,0] > transfer_prediction[i,0]):
#                adoption_prediction[i,0] = 1
#                return_to_owner_prediction[i,0] = 0
#                transfer_prediction[i,0] = 0
#            else:
#                adoption_prediction[i,0] = 0
#                return_to_owner_prediction[i,0] = 0
#                transfer_prediction[i,0] = 1
#        else:
#            if( return_to_owner_prediction[i,0] > transfer_prediction[i,0]):
#                adoption_prediction[i,0] = 0
#                return_to_owner_prediction[i,0] = 1
#                transfer_prediction[i,0] = 0
#            else:
#                adoption_prediction[i,0] = 0
#                return_to_owner_prediction[i,0] = 0
#                transfer_prediction[i,0] = 1



#    ID = np.arange( len(test_df) ) + 1
#    predictions = np.array([ID, adoption_prediction[:,0], died_prediction,
#                            euthanasia_prediction, return_to_owner_prediction[:,0],
#                            transfer_prediction[:,0]], dtype = int)

#    file_handle = open('../output/predictions.csv', 'ab')
#    head = ['ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer']
#    np.savetxt( file_handle, head, delimiter = '\n,', fmt='%s' )
#    for i in range( len(predictions.T) ):
#        row = predictions.T[i]
#        if( row[2] == 1):
#            row[1] = 0
#            row[3] = 0
#            row[4] = 0
#            row[5] = 0
#        if( row[3] == 1):
#            row[1] = 0
#            row[4] = 0
#            row[5] = 0
#        row = np.array_str( row )
#        row = row.translate(None, '[]')
#        row = np.fromstring(row, dtype = int, sep=' ' )
#        save = ''
#        for j in range( len(row) ):
#            save = save + str(row[j]) + ','
#    #    row = row.replace(' ',',')
#        save = save[:-1]
#        np.savetxt( file_handle, [save], delimiter = '\n', fmt='%s' )

#    file_handle.close()



