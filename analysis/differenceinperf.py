#outdated, do not use this anymore, included in decodedoutputdefs.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

def plot_diffinperf(file,savefile):
    df = pd.read_csv(file)
    #make dataframe for just recall
    df_include= []
    for i in df['$TrialName']:
        if 'Recall' in i:
            df_include.append('True')
        else:
            df_include.append('False')
    df.index = df_include
    df_recall = df.loc['True']
    
    #pull out target and actual decoded output for recall and all trials
    decodeout = df['#OutDecode']
    target = df['#OutTarget']
    decodeout_recall = df_recall['#OutDecode']
    target_recall = df_recall['#OutTarget']
    
    plt.subplot(1,2,1)
    diff =np.array(decodeout)-np.array(target)
    plt.hist(diff)
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.title('All Test Trials')
    plt.subplot(1,2,2)
    diff =np.array(decodeout_recall)-np.array(target_recall)
    plt.hist(diff)
    plt.xlabel('Difference')
    plt.title('Recall Test Trials')
    plt.suptitle('Difference in Precision on Test Trials (Target-Output)')
    plt.savefig(savefile)