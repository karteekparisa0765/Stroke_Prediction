# Stroke_Prediction
This project predicts stroke occurrences using machine learning on a healthcare dataset. It includes data preprocessing (label encoding, KNN imputation, SMOTE for balancing), and trains models like Naive Bayes, Decision Tree, SVM, and Logistic Regression. The goal is to develop a robust model for stroke prediction.

This project involves predicting stroke occurrences using machine learning models applied to a healthcare dataset. The dataset contains demographic and medical data about individuals, and the objective is to build an effective model that can accurately predict whether an individual will experience a stroke based on these features.

1. Data Loading & Exploration:
The process starts with loading the stroke dataset into a Pandas DataFrame using pd.read_csv(). Initial data inspection is performed using methods like .head(), .info(), and .describe() to gain a better understanding of the dataset. The .info() method gives details about the columns, their data types, and non-null counts. The .describe() method provides a summary of the statistical properties of the numerical columns, such as mean, standard deviation, and percentiles. Missing values are checked using .isnull().sum(), and duplicated rows are identified using .duplicated().sum() to ensure data integrity.

2. Data Visualization:
Several visualizations are generated to better understand the dataset:

Count plots: These are used to visualize the distribution of categorical features such as gender, marital status, work type, residence type, smoking status, and stroke occurrence.
Heatmap: A heatmap is created to visualize the presence of missing values in the dataset. These visualizations help identify data imbalances, trends, and patterns in the features.
3. Data Preprocessing:
Data preprocessing involves multiple steps to prepare the data for machine learning:

Label Encoding: Categorical variables such as gender, marital status, work type, residence type, and smoking status are encoded into numerical values using LabelEncoder. This makes them suitable for input into machine learning models, which typically require numerical data.
Handling Missing Values: The bmi column contains missing values, which are imputed using KNN imputation with the KNNImputer class. This method estimates the missing values based on the average values of the nearest neighbors.
Outlier Detection & Treatment: Outliers in the avg_glucose_level feature are detected using the Interquartile Range (IQR) method. Values outside the range defined by 1.5 times the IQR are considered outliers and are replaced with the mean value of the column.
4. Data Balancing:
The dataset is imbalanced with a larger number of non-stroke instances. To address this, SMOTE (Synthetic Minority Over-sampling Technique) is applied. SMOTE generates synthetic samples for the minority class (stroke occurrences) by creating new samples that are similar to existing ones. This ensures that both classes are represented equally during model training and reduces bias towards the majority class.

5. Model Training & Evaluation:
After data preprocessing and balancing, the dataset is split into training and testing sets using train_test_split(), with 60% of the data used for training and 40% for testing. Several machine learning models are trained on the data:

Naive Bayes: A probabilistic classifier that assumes independence between features.
Decision Tree: A tree-based model that splits the data based on feature values to make predictions.
Support Vector Machine (SVM): A model that finds the optimal hyperplane to separate classes in high-dimensional space.
Logistic Regression: A linear model used for binary classification tasks.
Each model is trained on the training data and evaluated on the test set. The accuracy score is calculated to assess the performance of each model, which measures the percentage of correct predictions.

6. Model Comparison:
Finally, the performance of all trained models is compared visually. A bar plot is created to show the accuracy of Naive Bayes, Decision Tree, SVM, and Logistic Regression, allowing for an easy comparison of which model performed best in predicting stroke occurrences.

In conclusion, this project demonstrates the entire pipeline of data preprocessing, model training, and evaluation, ultimately leading to a comparison of multiple machine learning models for stroke prediction. The key steps include data exploration, handling missing data, encoding categorical variables, balancing the dataset, training models, and evaluating their performance. This approach provides a robust framework for predicting health outcomes using machine learning techniques.
