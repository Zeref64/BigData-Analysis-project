# Assignment for BigData 2024 - Fire Incidents Analysis and Clustering in Toronto

This repository provides a comprehensive analysis of fire incidents in Toronto, including clustering, sub-clustering, cause analysis, response time analysis, and an attempt to estimate the safety from fire incidents using a machine learning model. The analysis is performed using the Toronto Fire Incidents dataset and includes both geographical clustering and sub-clustering based on timestamps, as well as feature engineering and machine learning using Random Forest Regression.

## Project Overview

This project involves:
- Clustering fire incidents based on their geographical location (latitude and longitude).
- Sub-clustering within specific regions (Downtown, North York, and Scarborough) based on timestamp and possible causes of fires.
- Analyzing the response time of the fire department in these regions.
- Using Random Forest Regressor to estimate the safety from fire incidents for each area.

## Dataset

The dataset used for this analysis is the [Toronto Fire Incidents Dataset](https://open.toronto.ca/dataset/fire-incidents/), which contains detailed information about fire incidents, including their location, possible causes, fire department response, and more.
- **Rows**: 29,425
- **Columns**: 43 (After cleaning and preprocessing, some columns are removed)

## Steps-Involved

### 1. Data Preprocessing
Handling missing values: Rows where latitude and longitude are missing or zero are removed.
Dropping unnecessary columns: Columns such as Area_of_Origin, Building_Status, and other categorical features that are not needed for clustering are dropped.
DateTime processing: Date columns such as TFS_Alarm_Time, TFS_Arrival_Time are converted to datetime format and numeric representations (in hours) for clustering.

### 2 Clustering using K-Means
The dataset is clustered based on the geographical coordinates (latitude and longitude) using the K-Means algorithm. The optimal number of clusters (k=3) is selected using the elbow method.
The clusters are mapped to geographical areas:

- Cluster 0: Downtown Toronto
- Cluster 1: North York Toronto
- Cluster 2: Scarborough Toronto
The results are visualized in a scatter plot.

### 3. Sub-Clustering and Elbow Method
For each cluster (Downtown, North York, Scarborough), the elbow method is applied again to find the optimal number of sub-clusters based on timestamps (TFS_Arrival_Time, Fire_Under_Control_Time, etc.) and possible causes of fires.

### 4. Possible Causes Analysis
For each region, the most common causes of fires are analyzed and visualized using stacked bar charts. The Possible_Cause column is grouped by the cluster, and the count of each possible cause is displayed.

### 5. Response Time Analysis
For each region, fire department response times are calculated and analyzed. Three key metrics are examined:

Response Time: Time from alarm to arrival at the fire scene.
Fire Under Control Time: Time taken to bring the fire under control.
Total Clear Time: Time from arrival to the fire scene being fully cleared.
These metrics are visualized by cluster within each region.

### 6. Fire Department Performance Comparison
A statistical comparison of the fire department's performance in each area is conducted by calculating the mean and standard deviation of response times, fire control times, and total clear times. The results are visualized as bar plots.

### 7. Random Forest Regressor to Estimate Safety
A Random Forest Regressor is trained to estimate the overall Safety_Score for each region. The safety score is calculated as a combination of:

Response Time
Fire Under Control Time
Total Clear Time
Civilian Casualties
Estimated Dollar Loss
The model is trained on the relevant features and used to estimate safety in different areas.

## Results

- Geographical Clustering: Fire incidents are grouped into three main clusters corresponding to the major areas in Toronto: Downtown, North York, and Scarborough.
- Sub-Clustering: Further clustering based on timestamps reveals distinct sub-patterns in fire incidents within each area.
- Fire Cause Analysis: The leading causes of fire incidents vary by region and cluster.
- Response Time Analysis: The fire department's response times differ across clusters, with some clusters showing quicker response times than others.

## Acknowledgements
This project was inspired by several online resources, including:

- The Toronto Open Data Portal
- A tutorial on K-Means Clustering
- Various data preprocessing techniques from the community and online courses.
- Special thanks to the contributors of open-source libraries like pandas, scikit-learn, and matplotlib that made this project possible.
Safety Scores: The Random Forest model provides an estimation of safety based on several factors. Scarborough has the highest safety score, followed by North York and Downtown.
