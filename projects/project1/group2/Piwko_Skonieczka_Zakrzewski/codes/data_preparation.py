import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

#____________________________________
#BANKNOTE AUTHENTICATION DATA   
banknote = pd.read_csv('data_banknote_authentication.txt', header=None)
banknote.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
banknote.to_csv('Project_data/banknote_authentication.csv', index=False)


#____________________________________
#ABALONE DATA
abalone = pd.read_csv('abalone.data', header=None)
abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']

#CONVERT TARGET VARIABLE TO BINARY
abalone['Rings'] = np.array([1 if abalone['Rings'][i] >= 10 else 0 for i in range(len(abalone))])

#ONE HOT ENCODING 'SEX' FEATURE
abalone['Female'] = pd.get_dummies(abalone['Sex'])['F']
abalone['Male'] = pd.get_dummies(abalone['Sex'])['M']
abalone = abalone.drop('Sex', axis=1)

#CORRELATED VARIABLES - DELETION WHERE |CORR| > 0.9
abalone.drop(['Diameter', 'Whole_weight', 'Viscera_weight', 'Shucked_weight'], axis=1, inplace=True)

abalone.to_csv('Abalone.csv', index=False)


#__________________
#ONLINE NEWS POPULARITY
popularity = pd.read_csv('OnlineNewsPopularity.csv')
popularity.drop(['url', ' n_non_stop_words', ' n_non_stop_unique_tokens', ' self_reference_avg_sharess', ' data_channel_is_world'], axis=1, inplace=True)

#CONVERT TARGET VARIABLE TO BINARY
popularity[' shares'] = np.array([1 if popularity[' shares'][i] >= 1500 else 0 for i in range(len(popularity))])
popularity.to_csv('NewsPopularity.csv', index=False)


#___________________
#MALWARE DETECTION 
malware = pd.read_csv('Project_data/Malware_dataset.csv')

#CONVERT TARGET VARIABLE TO BINARY
malware['classification'] = np.array([1 if malware['classification'][i] == 'malware' else 0 for i in range(len(malware))])

malware.drop(['hash', 'vm_truncate_count', 'mm_users', 'shared_vm', 'end_data', 'utime', 'nvcsw'], axis=1, inplace=True)

for col in malware.columns:
    if len(np.unique(malware[col])) == 1:
        malware.drop(col, axis=1, inplace=True)


#__________________
#STUDENTS ACADEMIC SUCCESS
students = pd.read_csv('Project_data/data.csv', delimiter=';')

#CONVERT TARGET TO VARIABLE TO BINARY
students['Target'] = np.array([1 if students.Target[i] == "Dropout" else 0 for i in range(len(students))])
students.drop(['Curricular units 1st sem (credited)', 'Curricular units 2nd sem (approved)',
               'Curricular units 1st sem (approved)', 'Curricular units 1st sem (enrolled)',
               'Curricular units 1st sem (grade)'], axis=1, inplace=True)
students.to_csv('students.csv', index=False)


#_________________
#DRUGS CONSUMPTION
drugs = pd.read_csv('Project_data/drug_consumption.data', header=None)
drugs = drugs[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]]
drugs.columns = ['age', 'gender', 'education', 'country', 'ethnicity', 
                  'nscore', 'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss', 'amphet']

#CONVERT TARGET VARIABLE TO BINARY
drugs.loc[:, 'amphet'] = [int(level[2]) for level in drugs['amphet']]
drugs.loc[:, 'amphet'] = [1 if level >= 2 else 0 for level in drugs['amphet']]
drugs.to_csv('drugs.csv', index=False)


#____________________
#BOSON HIGGS DETECTION DATA
data = arff.loadarff('Project_data/phpZLgL9q.arff')
boson = pd.DataFrame(data[0])
boson['class'].astype(str).astype(int)
boson.dropna(inplace=True)
boson.to_csv('higgs_boson.csv', index=False)