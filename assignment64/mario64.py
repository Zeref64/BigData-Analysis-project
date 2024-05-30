## Python BigData assingment [Fire Incidents]

import pandas as pd

# Κατεβάζω το dataset
url = "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/64a26694-01dc-4ec3-aa87-ad8509604f50/resource/1e824947-d73b-4f48-9bac-7f7f3731a6b9/download/Fire%20Incidents%20Data.csv"
dataset = pd.read_csv(url)

# Εμφάνιση της πρώτης γραμμής
# print(dataset.head())

# print (dataset.info())

# print (dataset.describe())

# print(dataset.isnull().sum())

# Display the column names
# print(dataset.columns)

# ξεχωριστά σε μοναδικές στήλες
print("Incident Types:", dataset['Final_Incident_Type'].unique())
neighborhoods_cleaned = dataset['Incident_Station_Area'].str.strip().unique()
print('\n')

print("Neighbourhoods:", neighborhoods_cleaned)
print('\n')
print("Incident Causes:", dataset['Possible_Cause'].unique())
print('\n')

# Ημερομηνία 
print("Date Range:", dataset['TFS_Alarm_Time'].min(), "to", dataset['TFS_Alarm_Time'].max()) 


