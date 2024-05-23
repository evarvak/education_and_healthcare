## Feature selection functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_hists(col, df, renamed=False):
    '''
    Make two (normalized) histrograms from (1) the data in column col with df['K7Q02R_R'] = 1, 2, or 3, and 
    (2) the data in column col with df['K7Q02R_R'] = 4 or 5.


    Arguments:
        col: string, the name of the column from which the histograms will be constructed
        df: pandas DataFrame from the NSCH dataset containing the column K7Q02R_R (that is, not a renamed column)
        renamed: Boolean that tells the function whether the dataset has renamed columns (True) or not (False). Defaults to False.
    Returns:
        a tuple of numpy arrays containing histogram data for use in hist_overlap() and plot_hists())
    '''

    #set the column containing days-missed data's name
    if renamed:
        ref_col = 'days_missed'
    else:
        ref_col = 'K7Q02R_R'
    
    n_bins = list(set(df[col].dropna())) + [max(list(set(df[col].dropna())))+1]
    n_bins.sort()

    x1 = df[col].dropna().loc[df[ref_col]<4]
    hist1, _ = np.histogram(x1, bins = n_bins)

    x2 = df[col].dropna().loc[df[ref_col]>=4]
    hist2, _ = np.histogram(x2, bins = n_bins)

    return (hist1.astype(np.float32) / hist1.sum(), hist2.astype(np.float32) / hist2.sum())



def hist_overlap(col, df):
    '''
    Find the overlap between (1) the normalized histogram made from data in column col with df['K7Q02R_R'] = 1, 2, or 3, and 
    (2) the normalized historgram made from the data in column col with df['K7Q02R_R'] = 4 or 5.

    Overlap values closer to 0 mean the histrograms are very different; values closer to 1 mean they are similar.

    Features with low overlap might be important for classifying days missed

    Arguments:
        col: string, the name of the column from which the histograms will be constructed
        df: pandas DataFrame from the NSCH dataset containing the column K7Q02R_R (that is, not a renamed column)
    Returns:
        overlap: the summed total overlap of the two histograms
    '''

    overlap = 0

    h1, h2 = make_hists(col, df)

    assert len(h1)==len(h2), 'Something went wrong, histrograms should be of the same length'

    for i in range(len(h1)):
        overlap += min(h1[i],h2[i])

    return overlap


def plot_hists(col,df,renamed=False):
    '''
    Plot the histograms created by make_hists()

    Arguments:
        col: string, the name of the column from which the histograms will be constructed
        df: pandas DataFrame from the NSCH dataset containing the column K7Q02R_R (that is, not a renamed column)
        renamed: Boolean that tells the function whether the dataset has renamed columns (True) or not (False). Defaults to False.
    Returns:
        None
    '''
    
    #set the column containing days-missed data's name
    if renamed:
        ref_col = 'days_missed'
    else:
        ref_col = 'K7Q02R_R'

    n_bins = list(set(df[col].dropna())) + [max(list(set(df[col].dropna())))+1]
    n_bins.sort()

    x1 = df[col].dropna().loc[df[ref_col]<4]
    hist, bins = np.histogram(x1, bins = n_bins)
    plt.bar(bins[:-1]-(bins[1]-bins[0])/6, hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0])/3, label = 'missed 0-6 days')
    #plt.title('current insurance usually meets child\'s needs')
    #plt.show()

    x2 = df[col].dropna().loc[df[ref_col]>=4]
    hist, bins = np.histogram(x2, bins = n_bins)
    plt.bar(bins[:-1]+(bins[1]-bins[0])/6, hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0])/3, label='missed 6+ days')
    plt.xlabel(col)
    plt.legend() 
    #plt.title('current insurance usually does not child\'s needs')
    plt.show()

    return None

