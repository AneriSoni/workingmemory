import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
sns.set()

class Precision():
    def __init__(self,file):
        #df = pd.read_csv(file,'\s+')
        df = pd.read_csv(file)
        self.df = df
        
    def plot_sigma(self,files,savefile):
        dat = {}
        print('lower case sigma, if file is uppercase sigma, switch')
        for f in files:
            sig = f.split('.csv')[0].split('sigma')[1].split('_Base')[0]
            thres = f.split('_Threhsold')[1].split('_')[0]
            if sig in dat.keys():
                dat[sig]=dat[sig].append(pd.read_csv(f,sep='\s+'))
            else:
                dat[sig] = pd.read_csv(f,sep='\s+')
        
        plots = len(files)
        i = 1
        for key in dat.keys():
            _,diffrecall = self.get_diffs(dat[key])
            plt.subplot(2,3,i)
            #plt.hist(diffrecall,range = (-0.01,0.01))
            plt.hist(diffrecall)
            plt.title('Sigma:{}'.format(float(key)))
            i+=1
        #plt.suptitle(('Difference in Precision on Test Trials (Target-Output), Threshold: {}',float(thres)))
        plt.suptitle(('Threshold: {}'.format(float(thres))))
        plt.tight_layout()
        #plt.show()
        plt.savefig(savefile)
            
    def get_diffs(self,df):
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

        
        diff =np.array(decodeout)-np.array(target)
        self.alldiff = diff
        diffrecall =np.array(decodeout_recall)-np.array(target_recall)
        self.recalldiff = diffrecall
        return diff,diffrecall
        

    def plot_diffinperf(self,savefile):
        df = self.df
        diff,diffrecall= self.get_diffs(df)
        #make dataframe for just recall
        
        plt.subplot(1,2,1)
        plt.hist(diff)
        plt.xlabel('Difference')
        plt.ylabel('Frequency')
        plt.title('All Test Trials')
        plt.subplot(1,2,2)
        
        #plt.hist(diffrecall, range = (-0.01,0.01))
        plt.hist(diffrecall)
        plt.xlabel('Difference')
        plt.title('Recall Test Trials')
        plt.suptitle('Difference in Precision on Test Trials (Target-Output)')
        plt.savefig(savefile)
    
    def plot_var_lesions(self,files,lesion_names,savefile):
        dat = {}
        for l in lesion_names:
            for f in files:
                if "Lesion"+l+"_" in f:
                    if l in dat.keys():
                        dat[l]=dat[l].append(pd.read_csv(f,sep='\s+'))
                    else:
                        dat[l] = pd.read_csv(f,sep='\s+') 
        self.lesiondat = dat
        plots = len(lesion_names)
        i = 1
        for key in dat.keys():
            _,diffrecall = self.get_diffs(dat[key])
            plt.subplot(2,2,i)
            #plt.hist(diffrecall,range = (-0.01,0.01))
            plt.hist(diffrecall)
            plt.title('Lesion:{}'.format(key))
            i+=1
        #plt.suptitle('Difference in Precision on Test Trials (Target-Output)')
        plt.tight_layout()
        plt.savefig(savefile)
    
    def plot_var_lesionprop(self,files,lesion_name,lesion_props,savefile):
        dat = {}
        for l in lesion_props:
            for f in files:
                if "LesionProp"+l+'_' in f:
                    if l in dat.keys():
                        dat[l]=dat[l].append(pd.read_csv(f,sep='\s+'))
                    else:
                        dat[l] = pd.read_csv(f,sep='\s+') 
        self.lesiondat = dat
        plots = len(lesion_props)
        i = 1
        for key in dat.keys():
            _,diffrecall = self.get_diffs(dat[key])
            plt.subplot(3,2,i)
            #plt.hist(diffrecall,range = (-0.01,0.01))
            #plt.hist(diffrecall,bins = 30, range = (-3,3))
            plt.hist(diffrecall,bins = 30)
            plt.title('LesionProp:{}'.format(key))
            i+=1
        #plt.suptitle('Difference in Precision on Test Trials (Target-Output)')
        plt.suptitle("{} Lesions".format(lesion_name))
        plt.tight_layout()
        plt.savefig(savefile)
    def plot_random_minus_fixed(self,files_random,files_fixed,lesion_name,lesion_props,savefile):
        print('this is not complete')
        #do what is in plot_var_lesionprop for random and fixed and then subtract before plotting   
        dat_random = {}
        dat_fixed = {}
        for l in lesion_props:
            for f in files_random:
                if "LesionProp"+l+'_' in f:
                    if l in dat_random.keys():
                        dat_random[l]=dat_random[l].append(pd.read_csv(f,sep='\s+'))
                    else:
                        dat_random[l] = pd.read_csv(f,sep='\s+') 
            for f in files_fixed:
                if "LesionProp"+l+'_' in f:
                    if l in dat_fixed.keys():
                        dat_fixed[l]=dat_fixed[l].append(pd.read_csv(f,sep='\s+'))
                    else:
                        dat_fixed[l] = pd.read_csv(f,sep='\s+') 
        self.dat_random = dat_random
        self.dat_fixed = dat_fixed
        plots = len(lesion_props)
        i = 1
        for key in dat_random.keys():
            _,diffrecall_random = self.get_diffs(dat_random[key])
            _,diffrecall_fixed = self.get_diffs(dat_fixed[key])
            self.diffrecall_random = diffrecall_random
            self.diffrecall_fixed = diffrecall_fixed
            plt.subplot(3,2,i)
            #plt.hist(diffrecall,range = (-0.01,0.01))
            #diff = diffrecall_random-diffrecall_fixed
            #self.diff = diff
            plt.hist(diffrecall_fixed,bins = 30, range = (-3,3))
            plt.title('LesionProp:{}'.format(key))
            i+=1
        #plt.suptitle('Difference in Precision on Test Trials (Target-Output)')
        plt.suptitle("{} Lesions".format(lesion_name))
        plt.tight_layout()
        plt.savefig(savefile)
        
            
                
    def plot_num_stripe(self,files, stripes,savefile):
        dat = {}
        for f in files:
            for s in stripes:
                if s+'layers' in f:
                    if s in dat.keys():
                        #dat[s]=dat[s].append(pd.read_csv(f,sep='\s+'))
                        dat[s]=dat[s].append(pd.read_csv(f))
                    else:
                        #dat[s] = pd.read_csv(f,sep='\s+')
                        dat[s] = pd.read_csv(f)
        i = 1
        for key in dat.keys():
            _,diffrecall = self.get_diffs(dat[key])
            plt.subplot(2,2,i)
            plt.hist(diffrecall,bins = 30, range = (-3,3))
            plt.title('Lesion Number of Stripes:{}'.format(key))
            plt.suptitle("Stripe Lesions".format(key))
            i +=1
        plt.tight_layout()
        plt.savefig(savefile)
                    
                    
                
        
def silentsynapses(trialnames):
    same = []
    diff = []
    rew_same = []
    rew_diff = []
    for i in range(len(trialnames)):
        if 'Recall' in trialnames[i]:
            trial = trialnames[i]
            x = trial.split('Recall')[1].split('_')[1] #letter/stimulus
            xprev = trialnames[i-1].split('_')[1] #letter/stimulus from previous trial
            rew = int(trial.split('rew_')[1]) #reward

            if x == xprev:
                same.append(i)
                rew_same.append(rew)
            else:
                diff.append(i)
                rew_diff.append(rew)
    corr_same = np.mean(rew_same)
    print(rew_diff)
    corr_diff = np.mean(rew_diff)
    
    return(rew_same,rew_diff,corr_same,corr_diff)
        
    