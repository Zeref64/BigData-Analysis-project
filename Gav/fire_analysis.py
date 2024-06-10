import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Φόρτωση των δεδομένων
file_path = 'Cleaned_Fire_Incidents_Data.csv'  # Ή 'Fire Incidents Data.csv'
data = pd.read_csv(file_path)

# Εξερεύνηση Δεδομένων
print("Columns in the dataset:")
print(data.columns)

print("\nFirst 5 rows of the dataset:")
print(data.head())

# Χρήση της στήλης 'Incident_Station_Area' για ανάλυση κατανομής
column_name = 'Incident_Station_Area'  # Προσαρμόστε αυτό εάν χρειάζεται

# Κατανομή των περιστατικών πυρκαγιάς σε διάφορες περιοχές του Τορόντο
neighborhood_distribution = data[column_name].value_counts()
print(f"\nDistribution of fire incidents across {column_name}:")
print(neighborhood_distribution)

# Μεταβολή των περιστατικών πυρκαγιάς με την πάροδο του χρόνου
# Μετατροπή της στήλης ημερομηνίας σε τύπο datetime
data['TFS_Alarm_Time'] = pd.to_datetime(data['TFS_Alarm_Time'])
data['Year'] = data['TFS_Alarm_Time'].dt.year
annual_incidents = data['Year'].value_counts().sort_index()
print("\nAnnual fire incidents over the years:")
print(annual_incidents)

# Οπτικοποίηση της κατανομής των περιστατικών πυρκαγιάς στις περιοχές
plt.figure(figsize=(14, 7))
sns.barplot(x=neighborhood_distribution.index, y=neighborhood_distribution.values)
plt.xticks(rotation=90)
plt.title(f'Distribution of Fire Incidents Across {column_name}')
plt.xlabel('Area')
plt.ylabel('Number of Incidents')
plt.show()

# Οπτικοποίηση της μεταβολής των περιστατικών πυρκαγιάς με την πάροδο του χρόνου
plt.figure(figsize=(14, 7))
sns.lineplot(x=annual_incidents.index, y=annual_incidents.values)
plt.title('Annual Fire Incidents Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Incidents')
plt.show()

