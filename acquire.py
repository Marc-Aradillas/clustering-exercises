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
                SELECT p17.*, pr17.logerror, pr17.transactiondate
                FROM (
                    SELECT parcelid, MAX(transactiondate) AS max_transactiondate
                    FROM predictions_2017
                    WHERE YEAR(transactiondate) = 2017
                    GROUP BY parcelid
                ) AS latest_transaction_date
                JOIN properties_2017 AS p17
                    ON latest_transaction_date.parcelid = p17.parcelid
                LEFT JOIN predictions_2017 AS pr17
                    ON p17.parcelid = pr17.parcelid
                LEFT JOIN propertylandusetype AS plu
                    ON p17.propertylandusetypeid = plu.propertylandusetypeid
                WHERE p17.latitude IS NOT NULL
                    AND p17.longitude IS NOT NULL;
                '''

        url = get_connection('zillow')
                
        df = pd.read_sql(query, url)

        # save to csv
        df.to_csv(filename,index=False)

    return df 
