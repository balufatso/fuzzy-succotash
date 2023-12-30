import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

# Function Definitions
def load_and_preprocess_data(file_path):
    """
    Loads the CSV data, preprocesses it by adding month, date, and birthday columns,
    and filters data for customers with a specific birthday.
    """
    df = pd.read_csv(file_path)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%b')
    df['date'] = pd.to_datetime(df['date'])
    df['birthday'] = pd.to_datetime(df['birthday'])

    # Filter DataFrame for customers with birthday '1/1/2018'
    birthday_filter = pd.Timestamp('2018-01-01')
    df_filtered = df[df['birthday'] == birthday_filter]
    df_filtered.sort_values(by='date', inplace=True)
    return df_filtered

def calculate_distinct_customer_count(df):
    """
    Groups data by date and counts distinct customer IDs.
    """
    grouped_df = df.groupby('date').agg({'cust_id': pd.Series.nunique}).reset_index()
    grouped_df = grouped_df.rename(columns={'cust_id': 'distinct_cust_count'})
    grouped_df['date'] = pd.to_datetime(grouped_df['date'])
    grouped_df.sort_values(by='date', inplace=True)
    return grouped_df

def calculate_alive_percentage(df):
    """
    Calculates '%Alive' column for the DataFrame.
    """
    base_value = df.loc[df['cohort'] == 0, 'distinct_cust_count'].iloc[0]
    df['%Alive'] = df['distinct_cust_count'] / base_value
    return df

def e_alive(cohort, gamma, delta):
    """
    E(% Alive) function.
    """
    return np.exp(gammaln(delta + cohort) + gammaln(gamma + delta) - gammaln(delta) - gammaln(gamma + delta + cohort))

def calculate_values(df, gamma, delta):
    """
    Calculates E(% Alive), P(ChurnTime = t), and E(# of Cust) for the DataFrame.
    """
    df['E(% Alive)'] = e_alive(df['cohort'], gamma, delta)
    df['P(ChurnTime = t)'] = df['E(% Alive)'].diff().fillna(df['E(% Alive)'].iloc[0])
    df['E(# of Cust)'] = df['distinct_cust_count'].iloc[0] * df['E(% Alive)']
    return df

def sse(params, df):
    """
    SSE function for optimization.
    """
    gamma, delta = params
    df_temp = calculate_values(df.copy(), gamma, delta)
    return np.sum((df_temp['E(# of Cust)'] - df_temp['distinct_cust_count']) ** 2)

def optimize_gamma_delta(df, initial_guess):
    """
    Optimizes gamma and delta values using the SSE function.
    """
    result = minimize(sse, initial_guess, args=(df,), bounds=[(0.00001, None), (0.00001, None)])
    return result.x