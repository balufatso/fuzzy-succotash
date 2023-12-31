import streamlit as st
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
    optimized_params = result.x

    return optimized_params


def main():
    st.title('Customer Churn Analysis')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df_filtered = load_and_preprocess_data(uploaded_file)
        
        grouped_df = calculate_distinct_customer_count(df_filtered)
        grouped_df['cohort'] = range(len(grouped_df))
        grouped_df = calculate_alive_percentage(grouped_df)

        # Optimization
        initial_guess = [st.sidebar.number_input("Enter initial guess for gamma", value=1.0, step=0.1),
                        st.sidebar.number_input("Enter initial guess for delta", value=1.0, step=0.1)]
        
        optimized_gamma, optimized_delta = optimize_gamma_delta(grouped_df, initial_guess)

        grouped_df = calculate_values(grouped_df, optimized_gamma, optimized_delta)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Gamma", value=f"{optimized_gamma:.4f}")
        with col2:
            st.metric(label="Delta", value=f"{optimized_delta:.4f}")
        with col3:
            st.metric(label="SSE", value=f"{sse}")


        # Extend the cohort values up to 24 using vectorized operations
        max_cohort = grouped_df['cohort'].max()
        extended_cohorts = pd.DataFrame({'cohort': range(max_cohort + 1, 25), '%Alive': np.nan})
        grouped_df = pd.concat([grouped_df, extended_cohorts], ignore_index=True)

        # Calculate E(% Alive) for the entire range of cohorts
        grouped_df['E(% Alive)'] = e_alive(grouped_df['cohort'], optimized_gamma, optimized_delta)

        # Calculate P(ChurnTime = t) and E(# of Cust)
        initial_cust_count = grouped_df['distinct_cust_count'].iloc[0]
        grouped_df['P(ChurnTime = t)'] = grouped_df['E(% Alive)'].diff().fillna(grouped_df['E(% Alive)'].iloc[0])
        grouped_df['E(# of Cust)'] = initial_cust_count * grouped_df['E(% Alive)']

        # Adding cohort month
        from pandas.tseries.offsets import MonthBegin
        start_month = grouped_df['date'].min()
        grouped_df['cohort_month'] = grouped_df.apply(lambda row: start_month + MonthBegin(n=int(row['cohort'])), axis=1)

        # Final DataFrame
        df_final = grouped_df[['cohort_month', 'cohort', 'distinct_cust_count', '%Alive', 'E(% Alive)', 'P(ChurnTime = t)', 'E(# of Cust)']]

        st.write("Final Data:")
        st.dataframe(df_final)

        # Example Plot
        st.line_chart(grouped_df[['%Alive', 'E(% Alive)']])
        st.write(sse)
        st.write(optimized_params)

if __name__ == '__main__':
    main()