# CS 422 Course Project: Employment Status Prediction in Hubei Province

## Project Overview

This project, developed for the CS 422 Data Mining course (Spring 2025, due May 23, 2025), analyzes labor market dynamics in Hubei Province, China, using a dataset of 4,980 individuals (`data.xlsx`). The dataset includes demographic, educational, employment, and socio-economic features to predict employment status (0 = unemployed, 1 = employed). The project employs three machine learning models—Random Forest (RF), Long Short-Term Memory (LSTM), and Support Vector Machine (SVM)—to classify employment status and identify key factors influencing unemployment. The results provide actionable insights for policymakers to design targeted interventions, such as training programs and job placement services, to enhance employment rates and reduce unemployment.

### Objectives
1. Identify demographic and socio-economic factors correlated with employment status.
2. Develop predictive models to classify individuals as employed or unemployed.
3. Provide policy recommendations to improve employability and address labor market challenges in Hubei Province.

### Dataset
- **Source**: `data.xlsx` (4,980 records, 54 features).
- **Key Features**:
  - Demographic: `age`, `sex`, `marriage`, `edu_level`.
  - Employment: `b_acc033` (labor contract), `b_aab022` (industry code).
  - Unemployment: `c_acc02e` (unemployment review date), `c_acc03b` (unemployment registration date), `c_ajc093` (unemployment reason).
  - Target: `c_acc028` (employment status: 0 = unemployed, 1 = employed).
- **Preprocessing**:
  - Removed columns with >80% missing values (e.g., `live_status`, `c_aca111`).
  - Dropped irrelevant columns (e.g., `people_id`, `name`).
  - Converted date columns (`c_acc02e`, `c_acc03b`) to days since January 1, 2020.
  - Transformed `c_acc028` to binary (0 = unemployed, 1 = employed).
  - Handled missing values differently per model (e.g., RF/LSTM: fill with 0; SVM: median for numeric, 'missing' for categorical).
- **Final Dataset**: `cleaned_data_modified.csv` (4,652 records, 26 features).

## Methodology

### Data Preprocessing
- **Cleaning**: Standardized missing values (`\N` to NaN), removed high-missing columns, and dropped irrelevant features.
- **Feature Engineering**: Converted date fields to numeric (days since 2020-01-01) and binarized employment status.
- **Exploratory Analysis**: Visualized feature distributions (e.g., `c_acc02e`, `c_acc03b`) and correlations (e.g., age vs. education level).

### Models
Three models were implemented to predict employment status:
1. **Random Forest (RF)**:
   - Handles non-linear patterns and mixed feature types.
   - Tuned: `n_estimators` (50, 100, 200), `max_depth` (5, 10, 20), `min_samples_split` (2, 5, 10).
2. **Long Short-Term Memory (LSTM)**:
   - Captures potential temporal patterns in sequential data.
   - Tuned: `units1` (30, 50, 70), `units2` (15, 25, 35), `learning_rate` (0.0001, 0.001, 0.01), `batch_size` (16, 32, 64).
3. **Support Vector Machine (SVM)**:
   - Uses RBF kernel for non-linear classification.
   - Tuned: `C` (0.1, 1.0, 10.0), `kernel` (linear, rbf, sigmoid), `gamma` (scale, auto, 0.1).

### Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1 Score.
- **Train-Test Split**: 80-20 (3,722 training, ~926 testing records).
- **Hyperparameter Tuning**: One-at-a-time tuning with fixed base parameters.
- **Feature Importance**: Analyzed using Gini (RF), permutation importance (RF, SVM, LSTM), and SVM coefficients (linear kernel).

## Results

### Model Performance
The table below summarizes the best performance metrics for each model based on the highest F1 Score:

| Model         | Tuned Parameters                              | Accuracy | Precision | Recall | F1 Score |
|---------------|-----------------------------------------------|----------|-----------|--------|----------|
| Random Forest | n_estimators=100, max_depth=20, min_split=5  | 0.8528   | 0.8665    | 0.9668 | 0.9139   |
| LSTM          | units1=8, units2=15, lr=0.001, batch=32      | 0.8067   | 0.8310    | 0.9548 | 0.8886   |
| SVM           | C=1.0, kernel='rbf', gamma=0.1               | 0.7132   | 0.8855    | 0.7407 | 0.8067   |

- **Random Forest**: Best overall (F1=0.9139), with high Recall (0.9668) and balanced Precision (0.8665), ideal for minimizing missed employed cases.
- **LSTM**: Strong Recall (0.9548) but lower Precision (0.8310), suitable for high-Recall scenarios but with more false positives.
- **SVM**: Highest Precision (0.8855) but lowest Recall (0.7407), limiting its use for broad screening.

### Key Findings
- **Top Features**: `c_acc02e` (unemployment review date), `c_acc03b` (unemployment registration date), and `c_ajc093` (unemployment reason) consistently ranked high across models, indicating their importance in predicting employment status.
- **Policy Insights**:
  - High Recall of RF ensures comprehensive identification of employed individuals, optimizing intervention allocation.
  - Precision-focused SVM can confirm employment status, reducing false positives.
  - Features like `edu_level` and `age` suggest targeted training for younger or less-educated individuals.



