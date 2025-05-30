### Project Title

Employment Status Prediction in Hubei, China

**Author**

Zibo Nie

#### Abstract

This project predicts employment status (employed vs. unemployed) in Hubei Province, China, using a dataset of 4,980 individuals. Machine learning models—Random Forest (RF), Long Short-Term Memory (LSTM), and Support Vector Machine (SVM)—were applied, with RF achieving the highest F1 Score (0.97). Key predictors include unemployment registration date and reason. Findings support targeted policy interventions like training programs. Next steps include ensemble modeling and data augmentation.

#### Rationale

Unemployment in Hubei Province impacts economic stability and social welfare, necessitating data-driven policies to enhance employability. Predictive models can identify at-risk individuals, enabling efficient resource allocation for job placement and training, aligning with regional goals of reducing unemployment and promoting economic growth.

#### Research Question

Which demographic and socio-economic factors predict employment status in Hubei Province, and which machine learning model best classifies individuals as employed or unemployed to inform policy interventions?

#### Data Sources

- **Dataset**: `data.xlsx` (4,980 records, 54 features), anonymized labor market data from Hubei Province.
- **Features**:
  - Demographic: `age`, `sex`, `marriage`, `edu_level`.
  - Employment: `b_acc033` (labor contract), `b_aab022` (industry code).
  - Unemployment: `c_acc02e` (unemployment review date), `c_acc03b` (unemployment registration date), `c_ajc093` (unemployment reason).
  - Target: `c_acc028` (0 = unemployed, 1 = employed).
- **Preprocessing**: Removed columns with >80% missing values (e.g., `live_status`), dropped irrelevant features (e.g., `name`), converted dates to days since 2020-01-01, resulting in `cleaned_data_modified.csv` (4,652 records, 26 features).

#### Methodology

- **Preprocessing**: Standardized missing values (`\N` to NaN), handled missing data (RF/LSTM: fill with 0; SVM: median for numeric, 'missing' for categorical), and binarized `c_acc028`.
- **Models**:
  - **Random Forest**: Tuned `n_estimators` (50, 100, 200), `max_depth` (5, 10, 20), `min_samples_split` (2, 5, 10) using Scikit-learn.
  - **LSTM**: Tuned `units1` (30, 50, 70), `units2` (15, 25, 35), `learning_rate` (0.0001, 0.001, 0.01), `batch_size` (16, 32, 64) using TensorFlow/Keras.
  - **SVM**: Tuned `C` (0.1, 1.0, 10.0), `kernel` (linear, rbf, sigmoid), `gamma` (scale, auto, 0.1) using Scikit-learn.
- **Evaluation**: 80-20 train-test split (~926 test records), metrics (Accuracy, Precision, Recall, F1 Score), one-at-a-time hyperparameter tuning, and feature importance via Gini (RF) and permutation (all models).

#### Results

- **Performance**:
  | Model         | Tuned Parameters                              | Accuracy | Precision | Recall | F1 Score |
  |---------------|-----------------------------------------------|----------|-----------|--------|----------|
  | Random Forest | n_estimators=100, max_depth=20, min_split=5  | 0.8528   | 0.8665    | 0.9668 | 0.9139   |
  | LSTM          | units1=8, units2=15, lr=0.001, batch=32      | 0.8067   | 0.8310    | 0.9548 | 0.8886   |
  | SVM           | C=1.0, kernel='rbf', gamma=0.1               | 0.7132   | 0.8855    | 0.7407 | 0.8067   |
- **Key Findings**: RF outperformed others (F1=0.9139), with `c_acc03b`, `c_acc02e`, and `c_ajc093` as top predictors. RF’s high Recall (0.9668) minimizes missed employed cases, ideal for policy screening.
- **Visualizations**:
  - Radar chart comparing model performance (`model_performance_radar.png`).
  - 3D scatter plot of age, education, and unemployment registration vs. employment status (`employment_3d.png`).
  - Correlation heatmap of numeric features (`correlation_heatmap.png`).

#### Next steps

- Engineer interaction features (e.g., age × education).
- Augment data with economic indicators and more samples.
- Explore ensemble models (e.g., stacking RF and LSTM).
- Implement 5-fold cross-validation and confusion matrix analysis.
- Develop a real-time prediction tool for policymakers.

#### Conclusion

RF’s high F1 Score (0.9139) and Recall (0.9668) make it the best model for identifying employed individuals, supporting efficient policy interventions. SVM’s high Precision (0.8855) suits targeted tasks, but low Recall (0.7407) limits its scope. Recommendations include using RF for screening and SVM for confirmation, focusing on education and unemployment duration. Caveats: Models rely on static data; dynamic labor market shifts may require retraining, and missing values may introduce bias.

### Bibliography

Breiman, Leo. 2001. “Random Forests.” *Machine Learning* 45 (1): 5–32. https://doi.org/10.1023/A:1010933404324.

Cortes, Corinna, and Vladimir Vapnik. 1995. “Support-Vector Networks.” *Machine Learning* 20 (3): 273–97. https://doi.org/10.1007/BF00994018.

Hochreiter, Sepp, and Jürgen Schmidhuber. 1997. “Long Short-Term Memory.” *Neural Computation* 9 (8): 1735–80. https://doi.org/10.1162/neco.1997.9.8.1735.

##### Contact and Further Information

Zibo Nie, niezibo0814@hotmail.com

