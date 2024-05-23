
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
        if feat in features: df.loc[df['K4Q27'] == 2, feat] = replace_with

    for feat in ['K12Q12', 'K3Q20', 'K3Q22', 'K11Q03R']:
        if feat in features: df.loc[df['CURRCOV'] == 2, feat] = replace_with

    if 'ISSUECOST' and ' K4Q27' in features: df.loc[df['K4Q27'] == 2, 'ISSUECOST'] = replace_with


    if 'K3Q21B' and 'HOWMUCH' in features: df.loc[df['HOWMUCH'] == 1, 'K3Q21B'] = replace_with

    if 'K4Q26'and 'K4Q24_R' in features: df.loc[df['K4Q24_R'] == 3, 'K4Q26'] = replace_with
    if 'K4Q02_R' and 'K4Q01' in features: df.loc[df['K4Q01'] == 2, 'K4Q02_R'] = replace_with

    if 'K4Q20R' and 'S4Q01' in features: df.loc[df['S4QO1'] == 2, 'K4Q20R'] = replace_with
    if 'K5Q31_R'and 'S4Q01' in features: df.loc[df['S4Q01'] == 2, 'K5Q31_R'] = replace_with
    if 'K5Q32'and 'S4Q01' in features: df.loc[df['S4Q01'] == 2, 'K5Q32'] = replace_with

    if 'K5Q32'and 'K5Q31_R' in features: df.loc[df['K5Q31_R'] == 2, 'K5Q32'] = replace_with
    if 'K5Q32'and 'K5Q31_R' in features: df.loc[df['K5Q31_R'] == 3, 'K5Q32'] = replace_with 


 
import numpy as np
from sklearn.impute import SimpleImputer

def impute_NSCH(df, features, imputer = 'mode'):
    
    nan_cols = [col for col in df.columns if df[col].isnull().sum() != 0]

    # This imputes nan entries by mode, column by column
    if imputer == 'mode':
        imp_mode = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
        for col in nan_cols:
            imp_col = imp_mode.fit_transform(df[col].values.reshape(-1,1))
            df[col] = imp_col



