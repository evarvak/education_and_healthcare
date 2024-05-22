
def cond_nan_NSCH(df, features, rep_cond_nans = 0):
    '''
    This function replaces nan entries which are conditional on the value of a different
    feature.

    Arguments:
    df -- full NSCH dataframe
    features -- list of str -- list of the features of interest
    rep_cond_nans -- any -- what we replace the nan value with

    Returns:
    df with conditional nan entries replaced with rep_cond_nans
    '''

    ## Note: this is all coded manually by looking at dependencies in each feature.
    ## Any new anticipated features will need to be added manually too.

    for feat in ['AVAILABLE', 'APPOINTMENT', 'ISSUECOST', 
                'NOTELIG', 'NOTOPEN', 'TRANSPORTCC']:
        df.loc[df['K4Q27'] == 2, feat] = rep_cond_nans

    if 'K12Q12' in features: df.loc[df['CURRCOV'] == 2, 'K12Q12'] = rep_cond_nans
    if 'K3Q21B' in features: df.loc[df['HOWMUCH'] == 1, 'K3Q21B'] = rep_cond_nans        
    if 'K3Q20' in features: df.loc[df['CURRCOV'] == 2, 'K3Q20'] = rep_cond_nans        
    if 'K3Q22' in features: df.loc[df['CURRCOV'] == 2, 'K3Q22'] = rep_cond_nans  

    if 'K4Q26'and 'K4Q24_R' in features: df.loc[df['K4Q24_R'] == 3, 'K4Q26'] = rep_cond_nans
    if 'K5Q31_R'and 'S4Q01' in features: df.loc[df['S4Q01'] == 2, 'K5Q31_R'] = rep_cond_nans

    if 'K5Q32'and 'S4Q01' in features: df.loc[df['S4Q01'] == 2, 'K5Q32'] = rep_cond_nans
    if 'K5Q32'and 'K5Q31_R' in features: df.loc[df['K5Q31_R'] == 2, 'K5Q32'] = rep_cond_nans
    if 'K5Q32'and 'K5Q31_R' in features: df.loc[df['K5Q31_R'] == 3, 'K5Q32'] = rep_cond_nans 
 



def impute_NSCH(df, features, imputer = 'mode'):
    
    nan_cols = [col for col in df.columns if df[col].isnull().sum() != 0]

    # This imputes nan entries by mode, column by column
    if imputer == 'mode':
        imp_mode = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
        for col in nan_cols:
            imp_col = imp_mode.fit_transform(df[col].values.reshape(-1,1))
            df[col] = imp_col

