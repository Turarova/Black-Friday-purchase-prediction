# Black Friday Purchase Prediction

This repository contains my solution for a **practice Kaggle competition** focused on predicting purchase amounts for Black Friday shoppers.

## Task
Predict the total purchase amount (`Purchase`) for each customer based on available customer and product features.

- **Metric**: Root Mean Squared Error (RMSE)
- **Current Score**: 2998.43883  
- **Public Leaderboard Rank**: 21 / 61

## Modeling Approach
Used an ensemble strategy combining **stacking** and **blending** with the following models:
- Random Forest
- Gradient Boosting Regressor
- XGBoost
- LightGBM

### Final Prediction
```python
def blend_models_predict(X_in):
    return (
        (0.1 * rf.predict(X_in)) +
        (0.2 * gbr.predict(X_in)) +
        (0.1 * lightgbm.predict(X_in)) +
        (0.2 * xgb.predict(X_in)) +
        (0.4 * stack_gen.predict(X_in.values))
    )
