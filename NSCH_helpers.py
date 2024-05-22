


def clean_NSCH(df, remove_sparse=True, remove_unexpected=False):
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
    return FIPS_to_State(clean_df.dropna(subset='days_missed'))




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
