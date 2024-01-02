import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

def load_and_preprocess_data(file_path):
    """
    Loads the CSV data, preprocesses it by adding month, date, and birthday columns,
    and filters data for customers with a specific birthday.
    """
    df = pd.read_csv(file_path)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%b')
    df['date'] = pd.to_datetime(df['date'])
    df['birthday'] = pd.to_datetime(df['birthday'])
    df['spend_rand'] = np.random.uniform(20, 40, size=len(df))


    # Filter DataFrame for customers with birthday '1/1/2018'
    unique_birthdays = sorted(df['birthday'].dt.date.unique())


    birthday_filter = pd.Timestamp('2018-01-01')
    selected_birthday = st.sidebar.selectbox("Select a Birthday", unique_birthdays)
    birthday_filter = pd.Timestamp(selected_birthday)

    df_filtered = df[df['birthday'] == birthday_filter]
    df_filtered.sort_values(by='date', inplace=True)
    return df_filtered

def calculate_distinct_customer_count(df):
    """
    Groups data by date and counts distinct customer IDs.
    """
    grouped_df = df.groupby('date').agg({
        'cust_id': pd.Series.nunique,  # Count distinct customer IDs
        'spend_rand': 'sum'  # Sum the spend for each group
    }).reset_index()
    grouped_df = grouped_df.rename(columns={
        'cust_id': 'distinct_cust_count',
        'spend_rand': 'total_spend'
    })
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
    # Convert cohort to numpy array if it's not already
    cohort_array = np.array(cohort) if not isinstance(cohort, np.ndarray) else cohort

    # Debugging print statements
    #print("Shapes and types:")
    #print("Cohort:", cohort_array.shape, type(cohort_array))
    #print("Gamma:", gamma, type(gamma))
    #print("Delta:", delta, type(delta))

    # Perform element-wise operations
    gamma_delta_sum = gammaln(gamma + delta)
    return np.exp(gammaln(delta + cohort_array) + gamma_delta_sum - gammaln(delta) - gammaln(gamma + delta + cohort_array))


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

def calculate_e_lifetime_years(df, column='t * P(ChurnTime = t)'):
    """
    Calculates the expected lifetime (E(Lifetime)) in years.
    """
    total = df[column].sum()
    e_lifetime_years = total / 12
    return  e_lifetime_years

def calculate_e_lifetime_years_3mo(df, column='t * P(churnTime = t | Alive @ 3)'):
    """
    Calculates the expected lifetime (E(Lifetime)) in years.
    """
    total = df[column].sum()
    e_lifetime_years_3mo = total / 12
    return  e_lifetime_years_3mo

def calculate_new_column(row):
    """
    Calculates the new column 't * P(ChurnTime = t)' for a given row.
    """
    # Use 'cohort' directly since it's an integer
    return row['cohort'] * row['P(ChurnTime = t)']

def append_new_row(df):
    """
    Appends a new row to the dataframe with specified calculations.

    :param df: The dataframe to which the row will be appended.
    :param initial_cust_count: The initial customer count, used for 'E(# of Cust)' calculation.
    :return: The dataframe with the new row appended.
    """
    new_row = {
        'cohort_month': '> 120',  # as specified
        'cohort': 121,  # as specified
        't * P(ChurnTime = t)': df['t * P(ChurnTime = t)'].sum(),
        # Sum all values for 'P(ChurnTime = t)' to get the new value
        'P(ChurnTime = t)': 2 - df['P(ChurnTime = t)'].sum(),
        # Calculate the new 'E(% Alive)'
        'E(% Alive)': 2 - df['P(ChurnTime = t)'].sum() - df['E(% Alive)'].iloc[-1],
        'E(# of Cust)': df['distinct_cust_count'].iloc[0] * (2 - df['P(ChurnTime = t)'].sum() - df['E(% Alive)'].iloc[-1])
    }
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)
    return df


def calculate_conditional_probability(df):
    """
    Calculates the conditional probability 'P(churnTime = t | Alive @ 3)'.
    """
    # Get the 'E(% Alive)' value for cohort 3
    alive_at_3 = df.loc[df['cohort'] == 3, 'E(% Alive)'].values[0]

    # Define a function to apply to each row
    def calculate_probability(row):
        if row['cohort'] <= 3:
            return 0
        else:
            return   row['P(ChurnTime = t)'] / alive_at_3

    # Apply the function to each row to create the new column
    df['P(churnTime = t | Alive @ 3)'] = df.apply(calculate_probability, axis=1)

    return df

def calculate_filtered_sse(df, actual_column, predicted_column):
    """
    Filters the DataFrame for non-NaN values in 'distinct_cust_count' and calculates SSE.
    """
    # Filter DataFrame based on non-NaN values in 'distinct_cust_count'
    filtered_df = df[df[actual_column].notna()]

    # Extract actual and predicted values
    actual = filtered_df[actual_column].values
    predicted = filtered_df[predicted_column].values

    # Calculate SSE
    sse = np.sum((actual - predicted) ** 2)

    return sse

def calculate_t_times_conditional_probability(df, conditional_prob_column='P(churnTime = t | Alive @ 3)', cohort_column='cohort'):
    """
    Calculates the new column 't * P(churnTime = t | Alive @ 3)'.
    
    :param df: The dataframe containing the data.
    :param conditional_prob_column: The name of the column containing 'P(churnTime = t | Alive @ 3)' values.
    :param cohort_column: The name of the column containing cohort values.
    :return: The dataframe with the new column added.
    """
    # Calculate the new column by multiplying 'P(churnTime = t | Alive @ 3)' by 'cohort'
    new_column_name = f"t * {conditional_prob_column}"
    df[new_column_name] = df[cohort_column] * df[conditional_prob_column]
    
    return df

def calculate_financial_metrics(data, margin=0.20, WACC_monthly=0.0095):
    """
    Calculate various financial metrics based on the provided dataset.

    Parameters:
    data (DataFrame): Pandas DataFrame containing the data.
    margin (float): Margin for calculating profit (default: 20%).
    WACC_monthly (float): Monthly Weighted Average Cost of Capital (default: 0.95%).

    Returns:
    DataFrame: Updated DataFrame with new financial metrics.
    """
    data['ARPU'] = data['total_spend'] / data['distinct_cust_count']

    # Define the model
    def model(params, X):
        return params[0] + params[1] * X

    # Define the SSE function
    def sse(params, X, Y):
        return np.sum((Y - model(params, X)) ** 2)

    # Extract variables
    X = data['cohort']
    Y = data['ARPU']

    # Initial guess for parameters
    initial_guess = [0, 0]

    # Optimize parameters
    result = minimize(sse, initial_guess, args=(X, Y))
    if not result.success:
        raise ValueError(result.message)
    
    fitted_params = result.x

    b0 = {fitted_params[0]}
    b1 = {fitted_params[1]}

    # Calculate additional metrics
    
    data['Calculated_ARPU'] = model(fitted_params, X)
    data['E(Total Rev)'] = data['Calculated_ARPU'] * data['distinct_cust_count']
    data['E(Rev. per Acq. Cust)'] = data['Calculated_ARPU'] * data['E(% Alive)']
    data['E(Profit per Acq. Cust)'] = data['E(Rev. per Acq. Cust)'] * margin
    data['PV( E(Profit per Acq. Cust) )'] = data['E(Profit per Acq. Cust)'] / ((1 + WACC_monthly) ** data['cohort'])

    return data,b0, b1
# Use in the main function as before

def main_calc(df_filtered):

    grouped_df = calculate_distinct_customer_count(df_filtered)
    grouped_df['cohort'] = range(len(grouped_df))
    grouped_df = calculate_alive_percentage(grouped_df)

    # Optimization
    initial_guess = [1,1]
    #[st.sidebar.number_input("Enter initial guess for gamma", value=1.0, step=0.1),st.sidebar.number_input("Enter initial guess for delta", value=1.0, step=0.1)]
    
    optimized_gamma, optimized_delta = optimize_gamma_delta(grouped_df, initial_guess)

    grouped_df = calculate_values(grouped_df, optimized_gamma, optimized_delta)

    # Extend the cohort values up to 24 using vectorized operations
    max_cohort = grouped_df['cohort'].max()
    extended_cohorts = pd.DataFrame({'cohort': range(max_cohort + 1, 120), '%Alive': np.nan})
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
    updated_sample_data, b0,b1 = calculate_financial_metrics(grouped_df)
    print(b0)

    # Final DataFrame
    df_final = grouped_df[['cohort_month', 'cohort', 'total_spend', 'ARPU','distinct_cust_count', '%Alive', 'E(% Alive)', 
                        'P(ChurnTime = t)', 'E(# of Cust)','Calculated_ARPU', 'E(Total Rev)', 'E(Rev. per Acq. Cust)',
        'E(Profit per Acq. Cust)', 'PV( E(Profit per Acq. Cust) )']]
    df_final['P(ChurnTime = t)'] = np.abs(df_final['P(ChurnTime = t)'])
    df_final['t * P(ChurnTime = t)'] = df_final.apply(calculate_new_column, axis=1)
    df_final = append_new_row(df_final)
    # Apply the function to df_final
    df_final = calculate_conditional_probability(df_final)
    df_final = calculate_t_times_conditional_probability(df_final)
    df_final = df_final[['cohort_month', 'cohort','distinct_cust_count','%Alive', 
                        'E(% Alive)', 'P(ChurnTime = t)', 'E(# of Cust)','t * P(ChurnTime = t)',	'P(churnTime = t | Alive @ 3)','t * P(churnTime = t | Alive @ 3)',
                        'total_spend', 'ARPU', 'Calculated_ARPU', 'E(Total Rev)', 'E(Rev. per Acq. Cust)','E(Profit per Acq. Cust)', 'PV( E(Profit per Acq. Cust) )']]
    
    # Getting some important values: Lifetime, Lifetime_3mo, future_Lifetime
    e_lifetime_years =calculate_e_lifetime_years(df_final)
    e_lifetime_years_3mo_v = calculate_e_lifetime_years_3mo(df_final)

    cohort_nr =  3  ### USER SET

    e_future_lifetime_years_3mo = calculate_e_lifetime_years_3mo(df_final) - (cohort_nr/12)
    e_future_lifetime_years_3mo = e_future_lifetime_years_3mo



    sse_value_1 = calculate_filtered_sse(df_final, actual_column='distinct_cust_count', predicted_column='E(# of Cust)') # Customer

    sse_value_2 = calculate_filtered_sse(df_final, actual_column='total_spend', predicted_column='E(Total Rev)') # Revenue

# e_lifetime_years, e_lifetime_years_3mo_v, e_future_lifetime_years_3mo, sse_value_1, sse_value_2
    df_final_1 = df_final[['cohort_month', 'cohort','distinct_cust_count','%Alive', 
                        'E(% Alive)', 'P(ChurnTime = t)', 'E(# of Cust)','t * P(ChurnTime = t)',	'P(churnTime = t | Alive @ 3)','t * P(churnTime = t | Alive @ 3)']]

    df_final_2 = df_final[['cohort_month', 'cohort','distinct_cust_count','%Alive', 
                        'total_spend', 'ARPU', 'Calculated_ARPU', 'E(Total Rev)', 'E(Rev. per Acq. Cust)','E(Profit per Acq. Cust)', 'PV( E(Profit per Acq. Cust) )']]
    return df_final_1, df_final_2, optimized_gamma,optimized_delta,sse_value_1,e_lifetime_years,e_lifetime_years_3mo_v,e_future_lifetime_years_3mo,b0,b1,sse_value_2,grouped_df

def main():
    st.title('Customer Churn Analysis')
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/PS_Logo_RGB.svg/1200px-PS_Logo_RGB.svg.png")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_and_preprocess_data(uploaded_file)
        column_names = df.columns.tolist()
        # User input for column mapping
        birthday_column = st.sidebar.selectbox("Select the column for 'Birthday'", column_names)
        date_column = st.sidebar.selectbox("Select the column for 'Date'", column_names)
        cust_id_column = st.sidebar.selectbox("Select the column for 'Customer ID'", column_names)
        spend_column = st.sidebar.selectbox("Select the column for 'Spend'", column_names)
        df_ml = df[[birthday_column, date_column, cust_id_column, spend_column]]
        #df_filtered = load_and_preprocess_data(df, birthday_column, date_column, cust_id_column, spend_column)
        #load_and_preprocess_data(uploaded_file)

         # Check if all required columns are selected
        if birthday_column and date_column and cust_id_column and spend_column:
            # Now, pass these column names to your processing functions
            #df_filtered = load_and_preprocess_data(df)
            if st.sidebar.button('Run Analysis', type="primary"):
                df_final_1, df_final_2, optimized_gamma,optimized_delta,sse_value_1,e_lifetime_years,e_lifetime_years_3mo_v,e_future_lifetime_years_3mo,b0,b1,sse_value_2,grouped_df = main_calc(df_ml)
                st.write("")
                st.write("")
                st.write("Final Data:")    
                st.line_chart(grouped_df[['%Alive', 'E(% Alive)']])
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric(label="Gamma", value=f"{optimized_gamma:.2f}")
                with col2:
                    st.metric(label="Delta", value=f"{optimized_delta:.2f}")
                with col3:
                    st.metric(label="SSE for %Alive", value=f"{sse_value_1:.0f}")
                with col4:
                    st.metric(label="SSE for %Alive", value=f"{e_lifetime_years:.2f}")
                with col5:
                    st.metric(label="SSE for %Alive", value=f"{e_lifetime_years_3mo_v:.2f}")
                with col6:
                    st.metric(label="SSE for %Alive", value=f"{e_future_lifetime_years_3mo:.2f}")

                st.write("Final Data:")

                st.dataframe(df_final_1)
                st.write("")
                st.write("")
                st.write("")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="b0",value=f"{b0.pop():.2f}" )
                with col2:
                    st.metric(label="b1", value=f"{b1.pop():.2f}" )
                with col3:
                    st.metric(label="SSE for Revenue", value=f"{sse_value_2:.0f}")
                

                st.write("")
                st.write("")
                st.write("Final Data:")
                st.dataframe(df_final_2)

                

            else:
                # Display a message to select all columns
                st.warning("Please select all required columns to proceed.")
        else:
            st.write('Goodbye')