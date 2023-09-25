# database connection defined function
# module imports

import os
import pandas as pd
from env import get_connection
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# custom modules

import summarize

# Acquire data.
# ----------------------ACQUIRE FUNCTION---------------------------------
def acquire_mall():

    filename_ = 'mall_data.csv'
    
    if os.path.isfile(filename_):
        
        return pd.read_csv(filename_)
        
    else: 

        query = '''
                SELECT *
                FROM customers
                '''
        
        url = get_connection('mall_customers')
                
        df = pd.read_sql(query, url)

        # save to csv
        df.to_csv(filename_,index=False)

    return df




def missing_values(df):
    # calculate number of missing value for each attribute
    missing_counts = df.isna().sum()

    # calculate the percent of missing vals in each attribute
    total_rows = len(df)
    missing_percentages = (missing_counts / total_rows) * 100

    # create a summary df
    summary_df = pd.DataFrame({'Missing Values' : missing_counts, 'Percentage Missing (%)': missing_percentages})

    return summary_df





def handle_missing_values(df, prop_required_column, prop_required_row):
    # Calculate the threshold for columns and rows
    
    total_rows = df.shape[0]
    total_columns = df.shape[1]
    col_threshold = int(total_rows * prop_required_column)
    row_threshold = int(total_columns * prop_required_row)
    
    # Drop columns with missing values exceeding the threshold
    df = df.dropna(axis=1, thresh=col_threshold)
    
    # Drop rows with missing values exceeding the threshold
    df = df.dropna(axis=0, thresh=row_threshold)
    
    return df



def train_val_test(df, target=None, seed = 42):

    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed,
                                       stratify = target)
    
    val, test = train_test_split(val_test, train_size = 0.5,
                                 random_state = seed,
                                 stratify = target)
    
    return train, val, test


def scale_data(train, val, test, scaler):

    # make copies for scaling
    train_scaled = train.copy()
    validate_scaled = val.copy()
    test_scaled = test.copy()

    columns_to_scale = ['customer_id', 'age', 'annual_income', 'spending_score', 'female', 'male']
    
    # Fit the scaler on the training data for all of the columns
    scaler.fit(train[columns_to_scale])
    
    # Transform the data for each split
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(val[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])

    scaled_col = [train_scaled, validate_scaled, test_scaled]
    
    return train_scaled, validate_scaled, test_scaled




def wrangle_mall():

    df = acquire_mall()
    
    # Summarize the data
    summarize.summarize(df)

    # Identify and create outlier columns
    for col in df.select_dtypes(include=['number']):
        df[f'{col}_outliers'] = summarize.identify_outliers(df[col])

    # Split the data into train, validation, and test sets
    train, val, test = train_val_test(df)

    # Define the categories for 'gender'
    gender_categories = ['Male', 'Female']  # Adjust as needed based on your data

    # Create dummy variables for 'gender' in all sets
    for data_set in [train, val, test]:
        for category in gender_categories:
            data_set[f'gender_{category}'] = (data_set['gender'] == category).astype(int)

        # Handle missing values
        data_set = handle_missing_values(data_set, prop_required_column=0.20, prop_required_row=0.75)

    # Select only the numeric columns for scaling (excluding 'gender' columns)
    numeric_columns = train.select_dtypes(include=['number']).columns

    # Scale the numeric data using Min-Max scaling
    mms = MinMaxScaler()
    for data_set in [train, val, test]:
        data_set[numeric_columns] = mms.fit_transform(data_set[numeric_columns])

    return train, val, test
