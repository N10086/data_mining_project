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

### Visualizations
- **Data Completeness Heatmap**: Shows data availability across features (`completeness_matrix.csv`).
  ![Data Completeness](completeness_matrix.png)
- **Scatter Plot**: Age vs. Education Level by employment status (`scatter_age_edu_level.png`).
  ![Scatter Plot](scatter_age_edu_level.png)
- **3D Scatter Plot**: Age, Education Level, and Unemployment Registration Date vs. Employment Status (`employment_3d.png`).
  ![3D Scatter Plot](employment_3d.png)
- **Correlation Heatmap**: Numeric feature correlations (`correlation_heatmap.png`).
  ![Correlation Heatmap](correlation_heatmap.png)
- **Radar Chart**: Compares RF, LSTM, and SVM performance across Accuracy, Precision, Recall, and F1 Score (`model_performance_radar.png`).
  ![Radar Chart](model_performance_radar.png)
- **Hyperparameter Tuning Plots**: Metrics for RF (`rf_n_estimators_metrics.png`, etc.), LSTM (`lstm_units1_metrics.png`, etc.), and SVM (`svm_C_metrics.png`, etc.).
- **Feature Importance Plots**: Gini and permutation importance for RF (`rf_feature_importance.png`), permutation for LSTM (`lstm_feature_importance.png`), and SVM (`svm_feature_importance.png`).

## Installation and Usage

### Prerequisites
- Python 3.8+
- Libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn tensorflow openpyxl

File Structure

├── data.xlsx                        # Raw dataset
├── cleaned_data_modified.csv        # Preprocessed dataset
├── completeness_matrix.csv          # Data completeness matrix
├── project_report.ipynb             # Main Jupyter Notebook
├── model_performance_radar.png      # Radar chart for model comparison
├── scatter_age_edu_level.png        # Age vs. Education scatter plot
├── employment_3d.png                # 3D scatter plot
├── correlation_heatmap.png          # Correlation heatmap
├── rf_feature_importance.png        # RF feature importance
├── lstm_feature_importance.png      # LSTM feature importance
├── svm_feature_importance.png       # SVM feature importance
├── rf_results.csv                   # RF experiment results
├── lstm_results.csv                 # LSTM experiment results
├── svm_results.csv                  # SVM experiment results
├── rf_n_estimators_metrics.png      # RF tuning plot
├── rf_max_depth_metrics.png         # RF tuning plot
├── rf_min_samples_split_metrics.png # RF tuning plot
├── lstm_units1_metrics.png          # LSTM tuning plot
├── lstm_units2_metrics.png          # LSTM tuning plot
├── lstm_learning_rate_metrics.png   # LSTM tuning plot
├── lstm_batch_size_metrics.png      # LSTM tuning plot
├── svm_C_metrics.png                # SVM tuning plot
├── svm_kernel_metrics.png           # SVM tuning plot
├── svm_gamma_metrics.png            # SVM tuning plot
├── README.md                        # Project documentation

Running the Project





Clone the repository:

git clone <repository-url>
cd <repository-directory>



Install dependencies:

pip install -r requirements.txt

Or manually install required libraries (see Prerequisites).



Open project_report.ipynb in Jupyter Notebook:

jupyter notebook project_report.ipynb



Run all cells to preprocess data, train models, and generate visualizations.





Outputs: cleaned_data_modified.csv, completeness_matrix.csv, model results (rf_results.csv, etc.), and visualizations (model_performance_radar.png, etc.).



Expected runtime: ~10-15 minutes (depending on hardware).

Troubleshooting





Missing Fonts: If SimHei font fails for Chinese characters, replace with Arial Unicode MS:

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']



Plot Not Displaying: Add %matplotlib inline in the notebook:

%matplotlib inline



Dependency Issues: Update libraries:

pip install --upgrade pandas matplotlib seaborn scikit-learn tensorflow

Policy Recommendations





Targeted Training Programs:





Focus on individuals with lower edu_level and younger age, as these features correlate with unemployment.



Use RF’s high Recall to identify at-risk groups for skill development.



Job Placement Services:





Leverage c_acc03b (unemployment registration date) to prioritize recent registrants.



SVM’s high Precision can confirm employment status for targeted placements.



Resource Allocation:





RF’s feature importance highlights c_acc02e and c_ajc093, suggesting policies addressing unemployment duration and reasons.



Minimize false negatives with RF to ensure comprehensive coverage.



Data Collection:





Enhance dataset with economic indicators to improve model generalization.



Collect more samples to address potential class imbalance.

Future Work





Feature Engineering: Add interaction terms (e.g., age × education level) and apply PCA for dimensionality reduction.



Model Enhancement: Implement ensemble methods (e.g., stacking RF and LSTM) and broader hyperparameter tuning.



Evaluation: Use 5-fold cross-validation and confusion matrix analysis to assess model stability and business impact.



Data Augmentation: Incorporate external data (e.g., regional unemployment rates) for better generalization.

Contributors





[Your Name] (CS 422 Student, Spring 2025)



Contact: [Your Email or GitHub Profile]

Acknowledgments





CS 422 Instructor and TAs for guidance.



Hubei Province dataset providers for enabling labor market analysis.



Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow.
