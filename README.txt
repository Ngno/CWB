2DIAMOND - FOUNDATIONAL STAGE
TYPE 2 DIABETES PERSONA CREATION

Overview
This project focuses on persona creation and predictive modeling to analyze health-related data and predict the likelihood of diabetes among individuals based on various demographic and health factors. Using machine learning techniques, we aim to provide insights into factors influencing diabetes risk and create persona profiles to better understand different segments of the population.

Dataset
The dataset used in this project was downloaded from Kaggle https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data.
We've reuploaded into this GitHub repo.

Methodology
For persona creation, we explore data from specific categories of age and gender in class 1 diabetes (positive). We made it to two clusters per categories, resulting from the optimum sillouete scores on the data.
Median was used as the data is not normally distributed.  
We then employed supervised learning techniques, particularly Random Forest Classifier, for predicting diabetes. The dataset was split into training and testing sets, and we utilized GridSearchCV for hyperparameter tuning to optimize model performance. Evaluation metrics such as accuracy, precision, recall, F1 score, and ROC AUC score were used to assess model effectiveness.

Model Training
The Random Forest Classifier was trained on the dataset to predict the binary outcome of diabetes presence. We utilized the following hyperparameters, determined through GridSearchCV:

Based on the analysis, we derived persona profiles to characterize different segments of the population with respect to their health risks. These personas include demographic features like age, sex, education, income, as well as health-related indicators such as BMI category, physical activity level, and mental health status.

General Male Persona:
Middle-aged man with BMI category 3
Physically active, non-smoker, with high blood pressure and cholesterol
Average income and education level
General Female Persona:
Middle-aged woman with BMI category 3
Physically active, non-smoker, without high blood pressure or cholesterol issues
Moderate income and education level

Personas DataFrame:
The personas dataframe summarizes the characteristics of different personas based on demographic and health factors. It includes five distinct personas representing various combinations of age, sex, BMI category, and other relevant attributes.

Files Included
Notebooks: Jupyter notebooks used for data exploration, model training, and persona analysis.

Data: Original dataset and preprocessed data files.

Trained Model: The trained Random Forest Classifier model can be downloaded from OneDrive: https://1drv.ms/u/s!AoU5C8MiW8ELgiW_FR4zyIwGawK0?e=yRv3Aa

Instructions to Use the Trained Model
To use the trained model:

Click on the download link provided above to download the model file.

Save the downloaded file to your local machine.

Load the model in your Python environment using the appropriate library (e.g., joblib):

python
Copy code
import joblib

# Load the trained model
model = joblib.load('path_to_downloaded_model_file')

# Example: Make predictions using the loaded model
predictions = model.predict(X_test)
Requirements
Ensure you have the libraries listed on the requeirements.txt installed.
Preprocessed data.


Conclusion
This project highlights the persona creation in some categories, application of machine learning in healthcare analytics, demonstrating how predictive models can aid in understanding health risks and guiding preventive strategies. The persona profiles provide actionable insights for personalized healthcare interventions based on individual characteristics.

Contact Information
For any questions or feedback, please contact:

Anggi Novitasari
angginovitasari.id@gmail.com