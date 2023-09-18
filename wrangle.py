# database connection defined function
# module imports

import os
import pandas as pd
from env import get_connection

# Acquire data.
# ----------------------ACQUIRE FUNCTION---------------------------------
def acquire_zillow():

    filename = 'zillow_data.csv'
    
    if os.path.isfile(filename):
        
        return pd.read_csv(filename)
        
    else: 

        query = '''
                -- Create a Common Table Expression (CTE) named RankedTransactions
                
                WITH RankedTransactions AS (
                
                    -- Select the relevant columns and assign a row number to each row within each parcelid group
                    
                    SELECT
                        p17.*,
                        pr17.logerror,
                        pr17.transactiondate,
                        ROW_NUMBER() OVER (PARTITION BY p17.parcelid ORDER BY pr17.transactiondate DESC) AS rn
                    FROM (
                    
                        -- Subquery to find the maximum transaction date for each parcelid in 2017
                        
                        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
                        FROM predictions_2017
                        WHERE YEAR(transactiondate) = 2017
                        GROUP BY parcelid
                    ) AS latest_transactions
                    
                    -- Join the CTE with other tables
                    
                    LEFT JOIN properties_2017 AS p17 ON latest_transactions.parcelid = p17.parcelid
                    LEFT JOIN predictions_2017 AS pr17 ON p17.parcelid = pr17.parcelid
                    LEFT JOIN propertylandusetype AS plu ON p17.propertylandusetypeid = plu.propertylandusetypeid
                    -- Filter out rows with null latitude or longitude
                    WHERE p17.latitude IS NOT NULL AND p17.longitude IS NOT NULL
                )
                
                -- Select all columns from the CTE where the row number is 1 (latest transaction for each parcelid)
                
                SELECT *
                FROM RankedTransactions
                WHERE rn = 1;
                '''

                                # SELECT p17.*, pr17.logerror, pr17.transactiondate
                # FROM (
                #     SELECT parcelid, MAX(transactiondate) AS max_transactiondate
                #     FROM predictions_2017
                #     WHERE YEAR(transactiondate) = 2017
                #     GROUP BY parcelid
                # ) AS latest_transactions
                # LEFT JOIN properties_2017 AS p17 ON latest_transactions.parcelid = p17.parcelid
                # LEFT JOIN predictions_2017 AS pr17 ON p17.parcelid = pr17.parcelid
                # LEFT JOIN propertylandusetype AS plu ON p17.propertylandusetypeid = plu.propertylandusetypeid
                # WHERE p17.latitude IS NOT NULL AND p17.longitude IS NOT NULL;
        
        url = get_connection('zillow')
                
        df = pd.read_sql(query, url)

        # save to csv
        df.to_csv(filename,index=False)

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

    

