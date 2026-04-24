import plotly.express as px
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.miscmodels.ordinal_model import OrderedModel

def calculate_correlation(data_set, col_a, col_b, alpha):
    df = pd.read_csv(data_set)
    df = pd.DataFrame(df).dropna(subset=[col_a, col_b]) # stats.pearsonr does not handle NaN values, so we drop rows with NaN in the specified columns

    print(f"Calculating correlation between '{col_a}' and '{col_b}' with alpha = {alpha}")

    columns_to_check = [col_a, col_b]
    for col in columns_to_check:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col], unique_strings = pd.factorize(df[col])

            mapping_str = ", ".join([f"{val} = {i}" for i, val in enumerate(unique_strings)])
            
            print(f"Found strings in {col} with {len(unique_strings)} unique strings and {mapping_str}")

    group_a = df[col_a]
    group_b = df[col_b]

    corr_coef, p_val = stats.pearsonr(group_a, group_b)

    
    print(f"Correlation coefficient: {corr_coef}")
    print(f"P-value: {p_val}")
    print(f"Significant correlation: {p_val < alpha}")

    print("-" * 40)
    
    return corr_coef, p_val, df

# # Calculate the correlation between GDP and Health Factors
# calculate_correlation("DataSet/GroupD_DataSet.csv", "GDP (USD)", "Life Expectancy (in age)", 0.05)
# calculate_correlation("DataSet/GroupD_DataSet.csv", "GDP (USD)", "Adult Mortality (%)", 0.05)
# calculate_correlation("DataSet/GroupD_DataSet.csv", "GDP (USD)", "Hepatitis B (%)", 0.05)
# calculate_correlation("DataSet/GroupD_DataSet.csv", "GDP (USD)", "Food Safety IHR (%)", 0.05)

# # Calculate the correlation between Schooling and Health Factors
# calculate_correlation("DataSet/GroupD_DataSet.csv", "Schooling (Yrs)", "Life Expectancy (in age)", 0.05)
# calculate_correlation("DataSet/GroupD_DataSet.csv", "Schooling (Yrs)", "Adult Mortality (%)", 0.05)
# calculate_correlation("DataSet/GroupD_DataSet.csv", "Schooling (Yrs)", "Hepatitis B (%)", 0.05)
# calculate_correlation("DataSet/GroupD_DataSet.csv", "Schooling (Yrs)", "Food Safety IHR (%)", 0.05)

# # Calculate the correlation between Status and Health Factors
# calculate_correlation("DataSet/GroupD_DataSet.csv", "Status", "Life Expectancy (in age)", 0.05)
# calculate_correlation("DataSet/GroupD_DataSet.csv", "Status", "Adult Mortality (%)", 0.05)
# calculate_correlation("DataSet/GroupD_DataSet.csv", "Status", "Hepatitis B (%)", 0.05)
# calculate_correlation("DataSet/GroupD_DataSet.csv", "Status", "Food Safety IHR (%)", 0.05)

"""
QUESTION 1:
Is there a statistically significant relationship between economic factors (GDP, status, schooling) or health factors(life expectancy, adult mortality, hepatitis B) and food safety IHR % across countries?  
Null Hypothesis: There is no relationship between economic/health factors and food safety IHR % across countries.
Alternative Hypothesis: There is a relationship between economic/health factors and food safety IHR % across countries.
"""

def multiple_regression_stepwise(data_set, response_col, predictor_cols, alpha, vif_threshold=5.0):
    df = pd.read_csv(data_set)
    
    # Drop rows with NaN for the columns we care about
    df = pd.DataFrame(df).dropna(subset=[response_col] + predictor_cols)
    
    print(f"Starting multiple ordinal regression for response: '{response_col}'")
    print(f"Initial predictors: {predictor_cols}\n")

    columns_to_check = predictor_cols
    for col in columns_to_check:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col], unique_strings = pd.factorize(df[col])
            mapping_str = ", ".join([f"{val} = {i}" for i, val in enumerate(unique_strings)])
            print(f"Found strings in {col} with {len(unique_strings)} unique strings and {mapping_str}")
            
    # Convert response to ordered categorical
    df[response_col] = pd.Categorical(df[response_col], ordered=True)
    cats = df[response_col].cat.categories
    y = df[response_col]
    X = df[predictor_cols]
    
    predictors = list(X.columns)
    
    # Step 1: Remove based on VIF (Multicollinearity)
    while len(predictors) > 1:
        X_current = sm.add_constant(X[predictors])
        
        vifs = [variance_inflation_factor(X_current.values, i) for i in range(X_current.shape[1])]
        vif_series = pd.Series(vifs, index=X_current.columns).drop('const', errors='ignore')
        
        max_vif = vif_series.max()
        max_vif_var = vif_series.idxmax()
        
        if max_vif > vif_threshold:
            print(f"Removing overlap: '{max_vif_var}' (VIF: {max_vif:.2f} > {vif_threshold})")
            predictors.remove(max_vif_var)
        else:
            break
            
    # Step 2: Remove based on P-values
    while len(predictors) > 0:
        X_current = X[predictors]
        model = OrderedModel(y, X_current, distr='logit').fit(method='bfgs', disp=False)
        p_values = model.pvalues[predictors]
        
        if len(p_values) == 0:
            break
            
        max_p_value = p_values.max()
        max_p_var = p_values.idxmax()
        
        if max_p_value > alpha:
            print(f"Removing insignificant predictor: '{max_p_var}' (p-value: {max_p_value:.4f} > {alpha})")
            predictors.remove(max_p_var)
        else:
            break
            
    if len(predictors) > 0:
        X_final = X[predictors]
        final_model = OrderedModel(y, X_final, distr='logit').fit(method='bfgs', disp=False)
        
        f_pvalue = final_model.llr_pvalue
        status = "REJECT Null Hyp" if f_pvalue <= alpha else "FAIL to REJECT Null Hyp"
        
        print("\n--- Final Model ---")
        print(f"Result: {status} (Overall Model LLR P-value: {f_pvalue:.4f})")
        print(f"Predictors kept: {predictors}")
        print(final_model.summary())
        
        import plotly.graph_objects as go
        import plotly.express as px
        import numpy as np

        # Calculate Expected Value as predicted continuous value for ordinal labels
        predicted_probs = final_model.predict(X_final)
        expected = (predicted_probs.values * cats.values.astype(float)).sum(axis=1)
        y_true = y.astype(float).values
        residuals = y_true - expected
        
        df['Predicted_Expected'] = expected
        df['Residuals'] = residuals

        # 1. Actual vs Predicted Plot
        fig1 = px.scatter(df, x=response_col, y='Predicted_Expected', 
                         title=f"Ordinal Regression Result: {status} (p={f_pvalue:.4f})<br>Predictors: {', '.join(predictors)}",
                         labels={response_col: f"Actual {response_col}", 'Predicted_Expected': f"Predicted Expected {response_col}"})
        
        min_val = min(y_true.min(), expected.min())
        max_val = max(y_true.max(), expected.max())
        fig1.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Ideal (y=x)', line=dict(dash='dash', color='red')))
        fig1.show()
        
        # 2. Residual Plot
        fig2 = px.scatter(df, x='Predicted_Expected', y='Residuals', 
                          title='Residual Plot (Residuals vs Predicted Expected Value)',
                          labels={'Predicted_Expected': f"Predicted Expected {response_col}", 'Residuals': 'Residuals'})
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        fig2.show()

        # 3. QQ Plot
        res_sorted = np.sort(residuals)
        n = len(res_sorted)
        quantiles = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = stats.norm.ppf(quantiles)
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=theoretical_quantiles, y=res_sorted, mode='markers', name='Data'))
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(theoretical_quantiles, res_sorted)
        line_x = np.array([min(theoretical_quantiles), max(theoretical_quantiles)])
        line_y = intercept + slope * line_x
        fig3.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='Fit Line', line=dict(dash='dash', color='red')))
        fig3.update_layout(title="QQ Plot of Residuals", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        fig3.show()
        
        return final_model
    else:
        print("No predictors found significant after simplification. FAIL to REJECT Null Hyp.")
        return None

economic_and_health_factors = [
    "GDP (USD)", "Status", "Schooling (Yrs)", 
    "Life Expectancy (in age)", "Adult Mortality (%)", "Hepatitis B (%)"
]
multiple_regression_stepwise("DataSet/GroupD_DataSet.csv", "Food Safety IHR (%)", economic_and_health_factors, 0.05)