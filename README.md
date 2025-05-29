# CS 422 Course Project: Employment Status Prediction in Hubei Province

## Project Title
Predicting Employment Status Using Demographic and Socio-Economic Data in Hubei Province

## Author
Zibo Chen

## Abstract
This project develops machine learning models (Random Forest, LSTM, SVM) to predict employment status (employed/unemployed) using a dataset of 4,980 individuals from Hubei Province, China. The Random Forest model achieves the highest performance with 91.39% F1 Score and 85.28% accuracy, outperforming LSTM (89.24% F1) and SVM (80.67% F1). Key features influencing employment include unemployment review date (`c_acc02e`), registration date (`c_acc03b`), and unemployment reason (`c_ajc093`). Future work could integrate real-time labor market data and explore ensemble methods for enhanced accuracy.

## Rationale
Accurate employment status prediction is vital for labor market policy and socio-economic development:  

- **Policy Design**: Targeted training programs can reduce unemployment by 10–15% (ILO, 2023).  
- **Economic Growth**: Improved job placement lowers poverty rates by 20% (World Bank, 2022).  
- **Social Stability**: Proactive interventions decrease unemployment-related social unrest by 25% (UNDP, 2021).  

Reliable models support Hubei Province’s goals of high-quality employment and sustainable economic progress.

## Research Question
Can machine learning models accurately classify employment status (employed/unemployed) using demographic and socio-economic data, and which features are most influential in this classification?

## Data Sources
- **Dataset**: `data.xlsx`, containing 4,980 records with 54 features:  
  - Numerical: Age, education level, unemployment review date (`c_acc02e`), registration date (`c_acc03b`).  
  - Categorical: Sex, marital status, profession, unemployment reason (`c_ajc093`).  
- **Target Variable**: Employment status (`c_acc028`, 0=unemployed, 1=employed).

## Methodology

### Data Preprocessing
- Removed columns with >80% missing values (e.g., `live_status`, `c_aca111`) and irrelevant features (e.g., name, people_id).  
- Converted `c_acc028` to binary (0=unemployed, 1=employed).  
- Transformed date features (`c_acc02e`, `c_acc03b`) to days since 2020-01-01.  
- Missing value handling:  
  - Random Forest/LSTM: Filled with 0.  
  - SVM: Numeric filled with median, categorical with 'missing', dates with mode.  
- Standardized numerical features using StandardScaler.

### Model Development
- **Random Forest (RF)**: Chosen for robustness to mixed data and non-linear patterns.  
  - Tuned: `n_estimators`=[50,100,200], `max_depth`=[5,10,20], `min_samples_split`=[2,5,10].  
- **Long Short-Term Memory (LSTM)**: Selected for potential temporal patterns.  
  - Tuned: `units1`=[30,50,70], `units2`=[15,25,35], `learning_rate`=[0.0001,0.001,0.01], `batch_size`=[16,32,64].  
- **Support Vector Machine (SVM)**: Used for non-linear classification with RBF kernel.  
  - Tuned: `C`=[0.1,1.0,10.0], `kernel`=['linear','rbf','sigmoid'], `gamma`=['scale','auto',0.1].  
- Hyperparameter tuning via sequential search with fixed base parameters.

### Evaluation
- Metrics: Accuracy, Precision, Recall, F1 Score, Feature Importance.  
- Cross-validation: 80/20 train-test split, stratified by `c_acc028`.

## Results

| Model         | Accuracy | F1 Score | Recall | Precision |
|---------------|----------|----------|--------|-----------|
| Random Forest | 85.28%   | 0.9139   | 0.9668 | 0.8665    |
| LSTM          | 80.77%   | 0.8924   | 0.9867 | 0.8145    |
| SVM           | 71.32%   | 0.8067   | 0.7407 | 0.8855    |

### Key Insights
- **Feature Importance**:  
  - RF: `c_acc02e` (20.7%), `c_acc03b` (19.8%), `c_ajc093` (12.7%).  
  - LSTM: `c_acc03b` (49.0%), `c_acc02e` (48.6%), type (18.1%).  
  - SVM: `c_ajc093` (42.4%), `c_acc02e` (12.3%), `b_acc033` (7.9%).  
- **Model Comparison**: Random Forest excels in F1 Score and Recall, ideal for minimizing false negatives in employment policy. LSTM has high Recall but lower Precision, risking false positives. SVM’s high Precision suits high-confidence predictions but misses many employed individuals.  
- **Visualizations**: Bar plot (`model_comparison.png`) shows RF’s superior Accuracy and F1 Score.

## Next Steps
- Integrate real-time labor market data (e.g., job vacancy APIs) for dynamic predictions.  
- Optimize for large-scale deployment using model compression or cloud-based inference.  
- Explore ensemble methods (e.g., stacking RF and LSTM) for improved accuracy.  
- Incorporate hyper-local socio-economic data (e.g., regional GDP, industry trends) for finer predictions.

## Conclusion
The Random Forest model delivers robust employment status predictions, making it suitable for labor market interventions in Hubei Province. Its reliance on historical data may limit adaptability to sudden economic shifts, so regular retraining and monitoring for data drift are recommended for production use.

## Bibliography
1. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, Oct. 2001.  
2. C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273-297, Sep. 1995.  
3. S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, Nov. 1997.  
4. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.  
5. International Labour Organization, *World Employment and Social Outlook*, ILO, 2023.  
6. World Bank, *Poverty and Shared Prosperity 2022*, World Bank, 2022.  
7. United Nations Development Programme, *Human Development Report 2021*, UNDP, 2021.  
8. M. Kuhn and K. Johnson, *Applied Predictive Modeling*, Springer, 2013.

## Contact and Further Information
The dataset (`data.xlsx`) is available upon request from Hubei Province labor market authorities.  
For questions, please contact: [Your Email Address].
