import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def clean_columns(df, remove_sparse=False, remove_unexpected=False):
    '''
    Arguments:
    df -- Full NSCH data frame
    remove_sparse -- Boolean -- whether to include columns with sparse entries
    remove_unexpected -- Boolean -- whether to include only the columns expected to be relevent

    Returns:
    clean_df -- data frame with fewer and more comprehensive columns
    '''

    nsch_doc = pd.read_csv('data/NSCH_dictionary.csv')
    if remove_sparse:
        nsch_doc = nsch_doc.loc[nsch_doc['sparse']==0]
    if remove_unexpected:
        nsch_doc = nsch_doc.loc[nsch_doc['expected_feature']==1]
    features = nsch_doc['col_name_2019']

    nsch_dict = nsch_doc[['col_name_2019','new_name']].set_index('col_name_2019').transpose().to_dict('records')[0]

    clean_df = df[features].rename(columns=nsch_dict)
    return clean_df.dropna(subset='days_missed')




def FIPS_to_State(data, state='both'):

    '''
    Arguments:
    data -- data frame with 'FIPSST' column of binary state codes
    state -- 'abbr', 'full', or 'both' -- determines whether to label states by abbreviation or by full name

    Returns:
    original `data` with extra columns 'STATE' or 'ABBR'
    '''

    #Also note: this function changes the FIP code to an integer from a byte string.


    FIPS_state = pd.read_csv('data/FIPS_State.csv')
    FIPS_state['FIPSST'] = FIPS_state.FIPS

    data.FIPSST = data.FIPSST.apply(int)

    if state == 'abbr':
        data = data.merge(FIPS_state[['FIPSST', 'ABBR']], on='FIPSST')

    if state == 'full':
        data = data.merge(FIPS_state[['FIPSST', 'STATE']], on='FIPSST')

    if state == 'both':
        data = data.merge(FIPS_state[['FIPSST', 'STATE', 'ABBR']], on='FIPSST')

    return data


def cond_nan_NSCH(df, features, replace_with = 0):
    '''
    This function replaces nan entries which are conditional on the value of a different
    feature.

    Arguments:
    df -- full NSCH dataframe
    features -- list of str -- list of the features of interest
    replace_with -- any -- what we replace the nan value with

    Returns:
    df with conditional nan entries replaced with replace_with
    '''

    ## Note: this is all coded manually by looking at dependencies in each feature.
    ## Any new anticipated features will need to be added manually too.

    for feat in ['AVAILABLE', 'APPOINTMENT', 'ISSUECOST', 
                'NOTELIG', 'NOTOPEN', 'TRANSPORTCC']:
        if feat in features.values: df.loc[df['K4Q27'] == 2, feat] = replace_with

    for feat in ['K12Q12', 'K3Q20', 'K3Q22', 'K11Q03R']:
        if feat in features.values: df.loc[df['CURRCOV'] == 2, feat] = replace_with

    if 'ISSUECOST' and ' K4Q27' in features.values: df.loc[df['K4Q27'] == 2, 'ISSUECOST'] = replace_with


    if 'K3Q21B' and 'HOWMUCH' in features.values: df.loc[df['HOWMUCH'] == 1, 'K3Q21B'] = replace_with

    if 'K4Q26' and 'K4Q24_R' in features.values: df.loc[df['K4Q24_R'] == 3, 'K4Q26'] = replace_with
    if 'K4Q02_R' and 'K4Q01' in features.values: df.loc[df['K4Q01'] == 2, 'K4Q02_R'] = replace_with

    if 'K4Q20R' and 'S4Q01' in features.values: df.loc[df['S4Q01'] == 2, 'K4Q20R'] = replace_with
    if 'K5Q31_R'and 'S4Q01' in features.values: df.loc[df['S4Q01'] == 2, 'K5Q31_R'] = replace_with
    if 'K5Q32' and 'S4Q01' in features.values: df.loc[df['S4Q01'] == 2, 'K5Q32'] = replace_with

    if 'K5Q32'and 'K5Q31_R' in features.values: df.loc[df['K5Q31_R'] == 2, 'K5Q32'] = replace_with
    if 'K5Q32'and 'K5Q31_R' in features.values: df.loc[df['K5Q31_R'] == 3, 'K5Q32'] = replace_with 

    return df



def impute_NSCH(df, response = 'K7Q02R_R', 
                imputer = 'mode', 
                test = False,
                test_data = [],
                state = 'both'):
    '''
    This function imputes nan entries.

    Arguments:
    df --data frame of categorical features
    imputer -- str -- select imputation method
        options: mode (replace by mode)
                 rf (use RandomForestClassifier)

    Returns:
    df with imputed columns
    '''
    nan_cols = [col for col in df.columns if df[col].isnull().sum() != 0]

    # This imputes nan entries by mode, column by column
    if imputer == 'mode':
        imp_mode = SimpleImputer(missing_values = np.nan, strategy='most_frequent')

        for col in nan_cols:
            imp_col = imp_mode.fit(df[col].values.reshape(-1,1))
            if test:
                test_imp_col = imp_mode.transform(test_data[col].values.reshape(-1,1))
                test_data[col] = test_imp_col
                return test_data
            else:
                imp_col = imp_mode.transform(df[col].values.reshape(-1,1))
                df[col] = imp_col


    # This imputes nan entries via RandomForestClassifier, column by column    
    if imputer == 'rf':
        # Manually dropping the STATE and ABBR columns since they are not numerical
        # Note: Should probably just modify clean_NSCH and move FIPS_to_State after imputation.
        non_num_cols = ['STATE','ABBR']
        df = df.drop([col for col in non_num_cols if col in df.columns], axis = 1)
        if test: test_data = test_data.drop(non_num_cols, axis = 1)
        #if test: test_data = test_data.drop([col for col in non_num_cols if col in df.columns], axis = 1)

        for col in nan_cols:
            # This is df consisting of all rows where col is null.  This will be used after we fit the
            # imputer when creating the predictor.
            df_null = df.loc[df[col].isnull()]
            if test: test_null = test_data.loc[test_data[col].isnull()]
            # This is df consisting of all rows where col is not null.  This will be used to fit the imputer.
            df_notnull = df.loc[df[col].notnull()]

            df_train_X = df_notnull.drop(col, axis = 1)
            df_train_y = df_notnull[col]

            rf = RandomForestClassifier(n_estimators = 100, random_state=415)
            rf.fit(df_train_X, df_train_y)
            if test:
                X_test_pred = test_null.drop(col, axis = 1)
                y_test_pred = rf.predict(X_test_pred)
                y_test_pred = test_data.loc[test_data[col].isnull(), col]
                test_data = FIPS_to_State(df, state = state)
                test_data = test_data.drop(labels = 'FIPSST', axis = 1)
                return test_data

            X_pred = df_null.drop(col, axis = 1)
            y_pred = rf.predict(X_pred)

            df.loc[df[col].isnull(), col] = y_pred

    # Inserting the state columns and dropping the FIPSST column
    df = FIPS_to_State(df, state = state)
    df = df.drop(labels = 'FIPSST', axis = 1)

    return df
        


def clean_NSCH(df, response = 'K7Q02R_R',
               dropna_response = True,
               drop_notenrolled = True, 
               remove_sparse = False, 
               remove_unexpected = False,
               replace_with = 0,               
               state = 'both'):

    '''
    This function combines other cleaning functions to clean NSCH data.

    Arguments:
    df -- NSCH dataframe
    features -- list of str -- list of features to keep
    response -- str -- response feature
    dropna_response -- bool -- drops nan entries from the response feature
    drop_notenrolled -- bool -- drops children not enrolled in school
    remove_sprase/ remove_unexpected -- see claen_columns function
    replace_with -- any -- see cond_nan_NSCH function
    state -- str -- include state name ('full'), abbreviation ('abbr') or both ('both').
    '''

    # Getting features from NSCH_dictionary.csv
    nsch_doc = pd.read_csv('data/NSCH_dictionary.csv') 
    features = nsch_doc['col_name_2019']
    df = df[features]

    # Drops nan rows from the response variable
    if dropna_response: df = df[df[response].notna()]
    # Drops children not enrolled in school
    if drop_notenrolled: df = df[df['K7Q02R_R'] != 6]
    # Replacing conditional nan values with replace_with
    df = cond_nan_NSCH(df, features, replace_with = replace_with)
    # Column renamer
    df = clean_columns(df, remove_sparse=remove_sparse, remove_unexpected=remove_unexpected)
    # Adds state name and state abbreviation column
    df = FIPS_to_State(df, state = state)

    return df

