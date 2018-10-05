import pandas as pd
from nistats.design_matrix import make_design_matrix
import numpy as np

def genData(n_subs, tr, n_scans, betas, n_events):
    """
        Parameters:
            n_subs:     int
            tr:         float
            n_scans:    int
            betas:      list of tuples, each tuple contains the beta
                        for each participant per condition
    """
    n_conds = len(betas[0])
    frame_times = np.arange(n_scans) * tr
    X = pd.DataFrame()
    y = pd.DataFrame()
    for sub in range(n_subs):
        tmpbs = betas[sub]
        # create design matrix
        events = pd.DataFrame()
        events['duration'] = [1.0]*n_events
        events['onset'] = np.sort(np.random.choice(frame_times[2:-2], n_events, replace=False))
        conds = [f'cond_{x}' for x in range(n_conds)]
        events['trial_type'] = np.random.choice(conds, n_events)

        Xtmp = make_design_matrix(
            frame_times, events, hrf_model='glover')

        # create data with noise
        ytmp = pd.DataFrame()
        e = np.random.random(n_scans)*4
        y_t = np.array([tmpbs[x] * Xtmp[f'cond_{0}'].values for x in range(n_conds)])
        ytmp['y'] = y_t.sum(0) + e


        Xtmp['subject'] = [sub]*len(Xtmp)
        ytmp['subject'] = [sub]*len(ytmp)

        X = pd.concat([X, Xtmp])
        y = pd.concat([y, ytmp])

    return y, X
