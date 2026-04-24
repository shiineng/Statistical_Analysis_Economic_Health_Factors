import plotly.express as px
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

"""
QUESTION 1:
Is there a statistically significant relationship between economic factors (GDP, status, schooling)
and health factors(life expectancy, adult mortality, hepatitis B,food safety IHR %) across countries?
"""

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

# Calculate the correlation between GDP and Health Factors
calculate_correlation("DataSet/GroupD_DataSet.csv", "GDP (USD)", "Life Expectancy (in age)", 0.05)
calculate_correlation("DataSet/GroupD_DataSet.csv", "GDP (USD)", "Adult Mortality (%)", 0.05)
calculate_correlation("DataSet/GroupD_DataSet.csv", "GDP (USD)", "Hepatitis B (%)", 0.05)
calculate_correlation("DataSet/GroupD_DataSet.csv", "GDP (USD)", "Food Safety IHR (%)", 0.05)

# Calculate the correlation between Schooling and Health Factors
calculate_correlation("DataSet/GroupD_DataSet.csv", "Schooling (Yrs)", "Life Expectancy (in age)", 0.05)
calculate_correlation("DataSet/GroupD_DataSet.csv", "Schooling (Yrs)", "Adult Mortality (%)", 0.05)
calculate_correlation("DataSet/GroupD_DataSet.csv", "Schooling (Yrs)", "Hepatitis B (%)", 0.05)
calculate_correlation("DataSet/GroupD_DataSet.csv", "Schooling (Yrs)", "Food Safety IHR (%)", 0.05)

# Calculate the correlation between Status and Health Factors
calculate_correlation("DataSet/GroupD_DataSet.csv", "Status", "Life Expectancy (in age)", 0.05)
calculate_correlation("DataSet/GroupD_DataSet.csv", "Status", "Adult Mortality (%)", 0.05)
calculate_correlation("DataSet/GroupD_DataSet.csv", "Status", "Hepatitis B (%)", 0.05)
calculate_correlation("DataSet/GroupD_DataSet.csv", "Status", "Food Safety IHR (%)", 0.05)

"""
QUESTION 2:
Is there a statistically significant difference in food safety IHR between “Developed” and “Developing” countries?
Null Hypothesis: There is no difference between “Developed” and “Developing” countries.
Alternative Hypothesis: “Developed” countries have a greater food safety IHR percentage compared to “Developing” countries.
"""

# THIS DOESNT WORK IT SHOULD BE LOGISTIC REGRESSION
def simple_regression(data_set, col_a, col_b, alpha):
    df = pd.read_csv(data_set)
    df = pd.DataFrame(df).dropna(subset=[col_a, col_b])

    columns_to_check = [col_a, col_b]
    for col in columns_to_check:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col], unique_strings = pd.factorize(df[col])

            mapping_str = ", ".join([f"{val} = {i}" for i, val in enumerate(unique_strings)])
            
            print(f"Found strings in {col} with {len(unique_strings)} unique strings and {mapping_str}")

    independent_var = sm.add_constant(df[col_a])  # Add a constant term for the intercept
    dependent_var = df[col_b]

    model = sm.OLS(dependent_var, independent_var).fit()

    p_val = model.pvalues[col_a]
    one_tailed_p = p_val / 2 if model.params[col_a] > 0 else 1 - (p_val / 2)

    status = "REJECT Null Hyp" if one_tailed_p <= alpha else "FAIL to REJECT Null Hyp"
   
    
    fig = px.scatter(df, x=col_a, y=col_b, trendline="ols",
                     title=f"Regression Result: {status} (p={one_tailed_p:.4f})")
    fig.show()
    
    print(f"\nResult: {status}")
    print(f"One-tailed P-value: {one_tailed_p:.4f}")

    return model.summary()

simple_regression("DataSet/GroupD_DataSet.csv", "Status", "Food Safety IHR (%)", 0.05)