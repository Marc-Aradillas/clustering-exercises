# database connection defined function
# module imports

import os
import pandas as pd
from env import get_connection

# Acquire data.
# ----------------------ACQUIRE FUNCTION---------------------------------
def acquire_zillow():

    filename = 'zillow_data'
    
    if os.path.isfile(filename):
        
        return pd.read_csv(filename)
        
    else: 

        query = '''
                SELECT *
                FROM properties_2017 AS p17
                LEFT JOIN predictions_2017 AS pr17 ON p17.parcelid = pr17.parcelid
                LEFT JOIN propertylandusetype AS plu ON p17.propertylandusetypeid = plu.propertylandusetypeid
                WHERE YEAR(pr17.transactiondate) = 2017;
                '''

        url = get_connection('zillow')
                
        df = pd.read_sql(query, url)

        # save to csv
        df.to_csv(filename,index=False)

        return df 
