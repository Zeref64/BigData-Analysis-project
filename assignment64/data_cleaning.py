# Import the necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
file_path = 'Fire Incidents Data.csv'  # Βεβαιώσου ότι το αρχείο βρίσκεται στον ίδιο φάκελο με το script ή δώσε το πλήρες μονοπάτι
data = pd.read_csv(file_path)

# Data Exploration

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Display the structure of the dataset
print("\nDataset Info:")
print(data.info())

# Display summary statistics of the dataset
print("\nSummary statistics:")
print(data.describe(include='all'))

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Data Cleaning

# Handle missing values (example: drop rows with any missing values)
data_cleaned = data.dropna()

# Alternatively, you can fill missing values with a specific value or a method (e.g., mean, median)
# data_cleaned = data.fillna(data.mean())

# Check for duplicates
print("\nNumber of duplicate rows:")
print(data_cleaned.duplicated().sum())

# Drop duplicate rows
data_cleaned = data_cleaned.drop_duplicates()

# Verify the cleaned data
print("\nCleaned dataset info:")
print(data_cleaned.info())

# Save the cleaned dataset to a new CSV file
cleaned_file_path = 'Cleaned_Fire_Incidents_Data.csv'
data_cleaned.to_csv(cleaned_file_path, index=False)
