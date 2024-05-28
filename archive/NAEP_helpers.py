## Helper functions to scrape and clean NAEP data from www.nationsreportcard.gov

import requests
from bs4 import BeautifulSoup
import pandas as pd

def url_constructor():
    return None


def NAEP_df(url, write_csv=False, csv_name='NAEP_data.csv'):
    '''
    Gather NAEP data into a dataframe through www.nationsreportcard.gov's API

    Arguments: 
        url - string containing the API path
        write_csv - Boolean. If True, writes the data into a CSV file in the current directory. Defaults to False
        csv_name - String. Only needed if you're writing a CSV file. Good idea to give it a descriptive name
    Returns:
        pandas Dataframe containing all information requested from the API
    '''
    html = requests.get(url)
    assert html.json()['status']==200, 'Problem getting url: error code ' + str(html.json()['status'])

    df = pd.DataFrame.from_dict(html.json()['result'])
    if write_csv:
        df.to_csv(csv_name)

    return df

def make_df_nice(df, year_state=True):
    '''
    Converts the df returned by NAEP_df() into a multi-indexed dataframe (pivot table) with statistics organized by state and year

    Arguments:
        df - dataframe output by NAEP_df() or created using pd.read_csv() using the csv written by NAEP_df()
        year_state - Boolean. If True, index first by year then state. If False, index by state then year. Defaults to True.
    Returns:
        df_nice - a multi-indexed dataframe containing the NAEP data organized by state and year
    '''
    rename_dict = {'ALD:AD': 'percent_advanced', 'ALD:BA': 'percent_basic', 'ALD:PR': 'percent_proficient', 'MN:MN': 'mean_score', 'SD:SD': 'std_dev'}

    if year_state:
        ind = ['year', 'jurisdiction']
    else:
        ind = ['jurisdiction', 'year']
    df_nice = df[['year','jurisdiction','stattype','value']].pivot(columns=['stattype'],index=ind)['value']

    df_nice.rename(columns=rename_dict,inplace=True)

    return df_nice.loc[:,['mean_score','std_dev','percent_basic','percent_proficient','percent_advanced']]
#df_test = NAEP_df(url2)

