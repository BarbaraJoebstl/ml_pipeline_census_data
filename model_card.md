# Model Card

This model was trained on the U.S. Census dataset, which contains demographic and employment information for individuals. The goal is to predict whether an individual’s income exceeds 50K/year based on their features.


## Model Details
- **Model type:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)  
- **Implementation:** Python 3.13, scikit-learn  
- **Hyperparameters:**  
  - `n_estimators=100`  
  - `max_depth=None`  
  - `random_state=42`  
  - `n_jobs=-1`  
- **Encoders & Model Storage:** Saved as `random_forest_model.joblib`, `encoder.joblib`, `lb.joblib`  


## Intended Use
- **Purpose:** Predict whether an individual’s income exceeds 50K/year based on census features.  
- **Intended Users:** Data scientists, analysts, or researchers studying census data.  
- **Use Cases:**  
  - Socioeconomic analysis  
  - Income distribution studies  


## Training Data
- Source: U.S. Census dataset (`census.csv`)  
- Features included:  
  - **Categorical:** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`  
  - **Continuous:** all remaining numeric features  
- Label: `salary` (binary: >50K vs ≤50K)  
- Preprocessing: One-hot encoding, label binarization  
- Training/test split: 80/20  


## Evaluation Data
- Held-out test set (20% of dataset)  
- Evaluated on overall metrics and per-slice metrics for categorical features  


## Metrics

### Overall Model Performance
- Precision: 0.727  
- Recall: 0.612  
- F1-score: 0.665  

### Column-Level Aggregated Metrics

| Feature           | n       | Precision | Recall | F1    |
|------------------|--------|-----------|--------|-------|
| workclass         | 6501   | 0.722     | 0.601  | 0.655 |
| education         | 6505   | 0.673     | 0.505  | 0.561 |
| marital-status    | 6511   | 0.770     | 0.518  | 0.605 |
| occupation        | 6511   | 0.685     | 0.504  | 0.564 |
| race              | 6513   | 0.731     | 0.606  | 0.661 |
| relationship      | 6513   | 0.802     | 0.483  | 0.571 |
| sex               | 6513   | 0.733     | 0.598  | 0.658 |
| native-country    | 6357   | 0.732     | 0.611  | 0.664 |

> Metrics calculated with `precision_score`, `recall_score`, and `fbeta_score` (β=1) from scikit-learn.  

### Slice-Level Highlights
- Small categories may produce extreme metrics (e.g., `education=1st-4th`, `native-country=Vietnam`) — interpret with caution  
- Categories with missing values (`?`) included; performance may vary  

## Ethical Considerations
- Dataset contains sensitive demographic features: race, sex, education 
- Model may **amplify historical biases** present in the census data  

## Caveats and Recommendations
- Small sample sizes in some slices may lead to unreliable metrics  
- Model performance varies across categories
- Only trained on historical U.S. census data — may **not generalize** to other populations or years  
- Preprocessing requires categorical features to match the training schema  
- Recommended: use slice-level metrics to check fairness when analyzing predictions  


For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf