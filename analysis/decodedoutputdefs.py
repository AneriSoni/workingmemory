import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random




class Chunking():
    def __init__(self, file):
        df = pd.read_csv(file,'\s+')
        #df = pd.read_csv(file_loc+file)
        self.df = df
        
    def chunk(self):
        df = self.df
        df_include= [] #want all trials that are not recall
        for i in df['$TrialName']:
            if 'Recall' in i:
                df_include.append('False')
            else:
                df_include.append('True')
        df.index=df_include
        df_norecall = df.loc['True']
        
        inp = [float(i.split('_')[1]) for i in df_norecall['$TrialName']]
        inp_act = df_norecall['#InDecode']
        chunk = df_norecall['#ChunkDecode']
        
        mnt1 = [float(i.split('_')[3]) for i in df_norecall['$TrialName']]
        mnt2 = [float(i.split('_')[5]) for i in df_norecall['$TrialName']]
        inp_act = inp_act.values
        chunk = chunk.values
        
        num_trials = len(mnt1)
        DInpMnt = []  #differnce between input and what is being maintained
        DInpChunk = [] #difference between input and chunk layer
        for i in range(num_trials):
            Tmnt1 = mnt1[i]
            Tmnt2 = mnt2[i]
            Tinp = inp_act[i]
            Tchunk = chunk[i]

            DChunk = Tinp - Tchunk

            if Tmnt1!=-1 or Tmnt2!=-1:
                if Tmnt1!=-1 and Tmnt2!=-1:
                    D1 = Tinp-Tmnt1
                    D2 = Tinp-Tmnt2

                    DInpMnt.append(np.min([D1,D2]))
                    DInpChunk.append(Tinp-Tchunk)

                elif Tmnt1!=-1:
                    D1 = Tinp-Tmnt1

                    DInpMnt.append(D1)
                    DInpChunk.append(Tinp-Tchunk)

                elif Tmnt2!=-1:
                    D2 = Tinp-Tmnt2
                    DInpMnt.append(D2)
                    DInpChunk.append(Tinp-Tchunk)
        self.DInpMnt = DInpMnt
        self.DInpChunk = DInpChunk
        
    
        
    def plot(self):
        self.chunk()
        plt.scatter(self.DInpMnt,self.DInpChunk)
        plt.xlabel('Diff between Input and Maintained')
        plt.ylabel('Diff between Input and Chunk')
        plt.title('Chunk Layer')
 
    
#other defs not yet in compiled defs
def training_curve_indv(base, specific_file = 'None', full_files = []):
    #input is the base file with all the model files saved into it
    if len(full_files) == 0:  
        files = os.listdir(base)
        if specific_file == 'None':
            full_files = [base+f for f in files if 'EpcLog' in f]
        else:
            full_files = [base+f for f in files if 'EpcLog' in f]
            full_files = [f for f in full_files if specific_file in f]
            
    for f in full_files:
        if 'model0_' in f:
            plt.figure()
            df = pd.read_csv(f, sep = '\t')
            #SSE = df['#SSE']
            SSE = df['#EpcDecodedDiff']
            Epoch = df['|Epoch']
            plt.plot(Epoch,SSE)
            plt.show()
#         elif 'model10_' in f: #this should be fixed now because i fixed the max number of runs. 
#             print('skipping model 10')
        else:
            #print(f)
            plt.figure()
            df = pd.read_csv(f,sep = '\t', header = None)
            SSE  = df[2]
            Epoch = df[1]
            plt.plot(Epoch[1:],SSE[1:])
            plt.show()
def training_curve_average(base, specific_file = 'None', additional_title = ''):
    files = os.listdir(base)
    if specific_file == 'None':
        full_files = [base+f for f in files if 'EpcLog' in f]
    else:
        full_files = [base+f for f in files if 'EpcLog' in f]
        full_files = [f for f in full_files if specific_file in f]
    SSE=[]
    Epoch = []
    for f in full_files:
        if 'model0_' in f:
            df = pd.read_csv(f, sep = '\t')
            SSE.append(df['#SSE'][:500])
            Epoch.append(df['|Epoch'][:500])
            
#         elif 'model10_' in f:
#             print('skipping model 10')
        else:
            df = pd.read_csv(f,sep = '\t', header = None)
            SSE.append(df[2][:500])
            #Epoch = df[1]
    SSE_np = np.array(SSE)
    SSE_avg = np.mean(SSE_np, axis = 0)
    plt.plot(Epoch[0],SSE_avg)
    if "StimRange0to45" in f:
        plt.title(f.split('/')[1]+' StimDist Max 45'+additional_title)
    else:
        plt.title(f.split('/')[1]+additional_title)
    plt.show()
    return SSE_avg,Epoch

def getfiles(base):
    files = os.listdir(base)
    files = [f for f in files if '.csv' in f]
    full_files = [base+f for f in files if 'EpcLog' not in f]
    full_files = [f for f in full_files if 'TrialLog' not in f]
    return full_files

def geterror(base):
    full_files = getfiles(base)
    num_models = len(full_files)
    p = Precision()
    L1_means = np.zeros(num_models)
    L1_vars = np.zeros(num_models)
    for f in range(len(full_files)):
        #import pdb; pdb.set_trace()
        err_model = p.mult_models([full_files[f]])
        L1 = np.abs(err_model)
        L1_means[f]=np.mean(L1)
        L1_vars[f] = np.var(L1)
    #mse = np.sqrt(np.mean(err**2)) 
    total_mean = np.mean(L1_means)
    total_var = np.mean(L1_vars)/num_models
    return err_model, total_mean, np.sqrt(total_var)
    

def geterror_array(base):
    full_files = getfiles(base)
    p = Precision()
    err = p.mult_models(full_files)
    #mse = np.sqrt(np.mean(err**2)) 
    meanL2 = (np.mean(err**2))
    var = np.var(err**2)/20
    std = np.sqrt(var)
    return err, np.sqrt(meanL2), np.sqrt(std)

def plot_title(base):
    
    if 'contexp' in base:
        rewardtype = 'Continuous Exponential'
    elif 'RewThresh' in base:
        rewardtype = 'Binary'
    else:
        rewardtype = 'Continuous Linear'
        
    if 'decodedrew' in base:
        rewardfun = 'Decoded'
    else: 
        rewardfun = 'Unit Difference'
        
    if 'NoStimLoc' in base:
        stimloc = ', No Stim Loc'
    else: 
        stimloc = ", Stim Loc Used"
   
    if 'NoIgnore' in base:
        ignore = ', No Ignore Trials'
    else:
        ignore = ''

    if "StimRange0to45" in base:
        stim_dist = ', StimDist Max 45'
    else:
        stim_dist = ""


    title = 'Reward Function: ' + rewardtype +', ' +rewardfun +stimloc + ignore +stim_dist#+' '+additional_title
    return title
    

def mse_allmodels(base, base2, experiment, plot_errorbars = True):
    sir2_file = base + "sir2_new/" +base2+experiment
    sir2_chunk_file = base + "sir2_chunk/" +base2+"hybrid/"+experiment
    sir3_file = base + "sir3/" +base2+experiment
    sir3_chunk_file = base + "sir3_chunk/" +base2+"hybrid/"+experiment
    
    err_sir2, mse_sir2, std_sir2 = geterror(sir2_file)
    err_sir2_chunk, mse_sir2_chunk, std_sir2_chunk = geterror(sir2_chunk_file)
    err_sir3, mse_sir3, std_sir3 = geterror(sir3_file)
    err_sir3_chunk, mse_sir3_chunk, std_sir3_chunk = geterror(sir3_chunk_file)
    
    title = plot_title(sir2_file)
    if plot_errorbars == True:
        plt.errorbar([2,3],[mse_sir2,mse_sir3], [std_sir2,std_sir3])
        plt.errorbar([2,3],[mse_sir2_chunk,mse_sir3_chunk], [std_sir2_chunk,std_sir3_chunk])
    else:
        plt.plot([2,3],[mse_sir2,mse_sir3])
        plt.plot([2,3],[mse_sir2_chunk,mse_sir3_chunk])
    plt.xticks([2,3])
    plt.xlim([1.8,3.2])
    plt.ylabel('Absolute Error Averaged over Models')
    #plt.ylabel(r'$\sqrt{MSE}$')
    plt.xlabel('Number of Items')
    plt.legend(['No Chunk', 'Chunk'])
    plt.title(title)
    
    



def modelerr(base, title = '', additional_title = ''): 
    files = os.listdir(base)
    files = [f for f in files if '.csv' in f]
    full_files = [base+f for f in files if 'EpcLog' not in f]
    full_files = [f for f in full_files if 'TrialLog' not in f]
    p = Precision()
    err = p.mult_models(full_files)
    #print(len(err))
    plt.figure()
    plt.hist(err, bins = 50)

    if "StimRange0to45" in base:
        model_name = base.split('workingmemory')[1].split('/')[1]+' StimDist Max 45'
    else:
        model_name = base.split('workingmemory')[1].split('/')[1] 
    if 'decodedrew' in base:
        rewardfun = 'Decoded'
    else: 
        rewardfun = 'Unit Difference'
    if 'contexp' in base:
        rewardtype = 'Continuous Exponential'
    elif 'RewThresh' in base:
        rewardtype = 'Binary'
    else:
        rewardtype = 'Continuous Linear'
    if 'NoIgnore' in base:
        ignore = ', No Ignore Trials'
    else:
        ignore = ''
    if title == '':
        title = model_name + ', Reward Function: ' + rewardtype +', ' +rewardfun +' '+ ignore +' '+additional_title
    plt.title(title)
    mse = np.sqrt(np.mean(err**2))
    legend = 'mse = {:.2f}'.format(mse)
    plt.legend([legend])
    plt.show()

        

class Precision():
    def __init__(self, stim = (0, 360)):
        #stim = (0.4, 3.4)
        #df = pd.read_csv(file,'\s+')
        ##df = pd.read_csv(file)
        #self.df = df
        self.stim = stim
    def get_diffs(self,df, nonresponse):
        df_include= []
        for i in df['$TrialName']:
            if 'Recall' in i:
                df_include.append('True')
            else:
                df_include.append('False')
        df.index = df_include
        df_recall = df.loc['True']
        self.df_recall = df_recall
        #pull out target and actual decoded output for recall and all trials
        decodeout = df['#OutDecode']
        target = df['#OutTarget']
        decodeout_recall = df_recall['#OutDecode']
        target_recall = df_recall['#OutTarget']
        
        
        if nonresponse == 'guess':
            num_nonresponse = 0
            use_decodeout = []
            use_decodeout_recall = []
            
            for i in range(len(decodeout)):
                if decodeout.values[i] == 0:
                    use_decodeout.append(np.random.uniform(0,360))
                else:
                    use_decodeout.append(decodeout.values[i])
                    
            for i in range(len(decodeout_recall)):
                if decodeout_recall.values[i] == 0:
                    num_nonresponse+=1
                    use_decodeout_recall.append(np.random.uniform(0,360))
                else:
                    use_decodeout_recall.append(decodeout_recall.values[i])
            self.num_nonresponse = num_nonresponse
            self.total_length = len(use_decodeout_recall)
        if nonresponse == 'swap':
            num_nonresponse = 0
            stripes = [i for i in df.keys() if 'stripe' in i] #find the stripe names

            use_decodeout = [] #the new output for all trials
            use_decodeout_recall = [] #the new output for recall 

            for i in range(len(decodeout)):
                if decodeout.values[i] == 0:
                    stripe_value = []
                    for s in stripes: #need the original df
                        if df[s].values[i] != 0:
                            stripe_value.append(df[s].values[i])
                    if len(stripe_value)== 0:
                        use_decodeout.append(np.random.uniform(0,360))
                    else:
                        use_decodeout.append(random.choice(stripe_value))
                else:
                    use_decodeout.append(decodeout.values[i])
            
            for i in range(len(decodeout_recall)):
                if decodeout_recall.values[i] == 0:
                    num_nonresponse+=1
                    stripe_value = []
                    for s in stripes: #need the original df
                        if df_recall[s].values[i] != 0:
                            stripe_value.append(df_recall[s].values[i])
                    if len(stripe_value)== 0:
                        use_decodeout_recall.append(np.random.uniform(0,360))
                    else:
                        use_decodeout_recall.append(random.choice(stripe_value)) #random guess from what is in memory. 
                else:
                    use_decodeout_recall.append(decodeout_recall.values[i])
            self.num_nonresponse = num_nonresponse
            self.total_length = len(use_decodeout_recall)
            
        else: 
            use_decodeout = decodeout
            use_decodeout_recall = decodeout_recall
            

        stim_diff = (self.stim[1]-self.stim[0])/2
        diff =np.array(use_decodeout)-np.array(target)
        self.alldiff = diff
        
        diffrecall =np.array(use_decodeout_recall)-np.array(target_recall)
        #diffrecall = np.abs(diffrecall) #this is wrong wrong wrong....uncomment this (comment out the below line) and plot for reason why. 
        #diffrecall = (np.mod(diffrecall+stim_diff/2, stim_diff)-stim_diff/2) #correct version, wrong mod factor
        diffrecall = (np.mod(diffrecall+stim_diff, self.stim[1])-stim_diff)
        #diffrecall = np.mod(diffrecall, stim_diff)  #this is wrong because the errors are from 0 to 360 instead of centered at 0.
        
        #diffrecall = (np.mod(diffrecall+stim_diff/2, stim_diff)-stim_diff/2)/stim_diff*360 #degrees
        self.recalldiff = diffrecall
        return diff,diffrecall
    
    def mult_models(self,files,sep = '\s+', nonresponse = 'guess'):
        #this is for plotting error histogram when you have multiple models and want them plotted in one graph 
        #nonresponse is to deal with all the times the model guessing nothing (i.e. it has 0 value)
        #if none - leave as is, if 'guess' then replace with random uniform guess, if 'swap' will replace with the other value
        dat = pd.DataFrame()
        for f in files:
            dat = dat.append(pd.read_csv(f,sep=sep))
        self.dat = dat

        _,diffrecall = self.get_diffs(dat, nonresponse = nonresponse)
        self.diffrecall = diffrecall
        return diffrecall
        
#     def get_diffs(self,df):
#         df_include= []
#         for i in df['$TrialName']:
#             if 'Recall' in i:
#                 df_include.append('True')
#             else:
#                 df_include.append('False')
#         df.index = df_include
#         df_recall = df.loc['True']
#         self.df_recall = df_recall
#         #pull out target and actual decoded output for recall and all trials
#         decodeout = df['#OutDecode']
#         target = df['#OutTarget']
#         decodeout_recall = df_recall['#OutDecode']
#         target_recall = df_recall['#OutTarget']

#         stim_diff = (self.stim[1]-self.stim[0])/2
#         diff =np.array(decodeout)-np.array(target)
#         self.alldiff = diff
        
#         diffrecall =np.array(decodeout_recall)-np.array(target_recall)
#         #diffrecall = np.abs(diffrecall) #this is wrong wrong wrong....uncomment this (comment out the below line) and plot for reason why. 
#         #diffrecall = (np.mod(diffrecall+stim_diff/2, stim_diff)-stim_diff/2) #correct version, wrong mod factor
#         diffrecall = (np.mod(diffrecall+stim_diff, self.stim[1])-stim_diff)
#         #diffrecall = np.mod(diffrecall, stim_diff)  #this is wrong because the errors are from 0 to 360 instead of centered at 0.
        
#         #diffrecall = (np.mod(diffrecall+stim_diff/2, stim_diff)-stim_diff/2)/stim_diff*360 #degrees
#         self.recalldiff = diffrecall
#         return diff,diffrecall
    
#     def mult_models(self,files,sep = '\s+'):
#         #this is for plotting error histogram when you have multiple models and want them plotted in one graph 
#         dat = pd.DataFrame()
#         for f in files:
#             dat = dat.append(pd.read_csv(f,sep=sep))
#         self.dat = dat

#         _,diffrecall = self.get_diffs(dat)
#         self.diffrecall = diffrecall
#         return diffrecall
    
    def err_recall_type(self, file, sep = '\t'):
        #this only makes sense on a model by model basis - only one model at a time - because one model could learn for recall 1 and another
        #for recall2 and then they would cancel each other out. 
        df = pd.read_csv(file,sep)
        self.dat = df
        _,diffrecall = self.get_diffs(df)
        
        recall_type = []
        df_recall = self.df_recall
        for i in range(len(df_recall)):
            trial = df_recall['$TrialName'].iloc[i]
            if 'Recall1' in trial:
                recall_type.append('Recall1')
            elif 'Recall2' in trial:
                recall_type.append('Recall2')
        recall1_err = []
        recall2_err = []
        
        for i in range(len(diffrecall)):
            if recall_type[i] == 'Recall1':
                recall1_err.append(diffrecall[i])
            elif recall_type[i] == 'Recall2':
                recall2_err.append(diffrecall[i])
            
        return recall1_err,recall2_err
    
    def err_recall_type_sir3(self, file, sep = '\t'):
        #this only makes sense on a model by model basis - only one model at a time - because one model could learn for recall 1 and another
        #for recall2 and then they would cancel each other out. 
        df = pd.read_csv(file,sep)
        self.dat = df
        _,diffrecall = self.get_diffs(df)
        
        recall_type = []
        df_recall = self.df_recall
        for i in range(len(df_recall)):
            trial = df_recall['$TrialName'].iloc[i]
            if 'Recall1' in trial:
                recall_type.append('Recall1')
            elif 'Recall2' in trial:
                recall_type.append('Recall2')
            elif 'Recall3' in trial:
                recall_type.append('Recall3')
        recall1_err = []
        recall2_err = []
        recall3_err = []
        
        for i in range(len(diffrecall)):
            if recall_type[i] == 'Recall1':
                recall1_err.append(diffrecall[i])
            elif recall_type[i] == 'Recall2':
                recall2_err.append(diffrecall[i])
            elif recall_type[i] == 'Recall3':
                recall3_err.append(diffrecall[i])
            
        return recall1_err,recall2_err,recall3_err
    
        
    def create_dat(files, var, var_values): 
        #i don't think I will use this.
        #files to go through
        #var - ex. gain/stries/lesion
        #var_values - which variables to create separate keys in data table
        
        dat = {}
        for f in files:
            self.var_extraction()
            
    def var_extraction():
        #i don't think i will use this.
        print('def not done')    
            
    
    def gain(self,files,gains):
        dat = {}
        error_dat ={}
        for f in files:
            gain = f.split('Gain')[1].split('_')[1]
            if gain in gains:
                if gain in dat.keys():
                    dat[gain]=dat[gain].append(pd.read_csv(f,sep='\s+'))
                else:
                    dat[gain] = pd.read_csv(f,sep='\s+')
        for key in gains:
            _,diffrecall = self.get_diffs(dat[key])
            error_dat[key]=diffrecall
        
        self.error_dat = error_dat
        self.gains = gains
        
    def plot_gain(self,savefile, bins = 10, data = None):
        if data == None: 
            error_dat = self.error_dat
        else:
            error_dat = data
        gains = self.gains
        plots = len(gains)
        i = 1

        for key in gains:
            plt.subplot(1,plots,i)
            i+=1
            plt.hist(error_dat[key], bins = 10)
            plt.title('Gain: {}'.format(key))
        #plt.suptitle('Errors With Different Precision')
        plt.savefig(savefile)
            
            
    def stripes(self,files,stripes):
        dat ={}
        error_dat = {}
        for f in files:
            st = f.split('_')[2].split('Stripes')[1]
            if int(st) in stripes:
                if st in dat.keys():
                    dat[st]=dat[st].append(pd.read_csv(f,sep='\s+'))
                else:
                    dat[st] = pd.read_csv(f,sep='\s+')
        for key in dat.keys():
            _,diffrecall = self.get_diffs(dat[key])
            error_dat[key] = diffrecall
            
        self.error_dat = error_dat
        self.stries = stripes
        
    def plot_stripes(self,savefile):
        error_dat = self.error_dat
        stripes = self.stripes
        plots = len(stripes)
        i = 1
        for key in error_dat.keys():
            plt.subplot(2,4,i)
            i+=1
            plt.hist(error_dat[key])
            plt.title('{} Stripes'.format(float(key)))
            plt.savefig(savefile)
            
                
    def sigma(self,files):
        dat = {}
        error_dat={}
        print('lower case sigma, if file is uppercase sigma, switch')
        for f in files:
            sig = f.split('.csv')[0].split('sigma')[1].split('_Base')[0]
            thres = f.split('_Threhsold')[1].split('_')[0]
            if sig in dat.keys():
                dat[sig]=dat[sig].append(pd.read_csv(f,sep='\s+'))
            else:
                dat[sig] = pd.read_csv(f,sep='\s+')
        for key in dat.keys():
            _,diffrecall = self.get_diffs(dat[key])
            error_dat[key] = diffrecall
        self.dat = dat
        self.plots = len(files)
        
        
    def plot_sigma(self,savefile):
        error_dat = self.error_dat
        plots = self.plots
        #plots = len(files)
        i = 1
        for key in error_dat.keys():
            plt.subplot(1,6,i)
            #plt.hist(diffrecall,range = (-0.01,0.01))
            plt.hist(error_dat[key])
            plt.title('Sigma:{}'.format(float(key)))
            i+=1
        #plt.suptitle(('Difference in Precision on Test Trials (Target-Output), Threshold: {}',float(thres)))
        plt.suptitle(('Threshold: {}'.format(float(thres))))
        plt.tight_layout()
        #plt.show()
        plt.savefig(savefile)
    
    def sigma2(self,files):
        print('give files of only 1 sigma and multiple threshold')
        print('will plot both the histograms and 1 graph with rew. threshold and variance of histograms')
        dat = {}
        
        for f in files: 
            sig = f.split('.csv')[0].split('sigma')[1].split('_Base')[0]
            thres = f.split('_Threhsold')[1].split('_')[0]
            if thres in dat.keys():
                dat[thres]=dat[thres].append(pd.read_csv(f,sep='\s+'))
            else:
                dat[thres] = pd.read_csv(f,sep='\s+')
                
        stds = []
        means = []
        i = 1
        keys = list(dat.keys())
        sorted_keys = []
        for k in np.argsort(keys):
            key = keys[k]
            _,diffrecall = self.get_diffs(dat[key])
            stds.append(np.std(diffrecall))
            means.append(np.mean(diffrecall))
            sorted_keys.append(key)
            #plt.subplot(2,4,i)
            ##plt.hist(diffrecall,range = (-0.01,0.01))
            #plt.hist(diffrecall)
            #plt.title('Threshold:{}'.format(float(key)))
            #i+=1
        self.stds = stds
        self.means = means
        self.sorted_keys = sorted_keys
        
    def plot_sigma2(self,savefile):
        stds = self.stds
        means = self.means
        sorted_keys = self.sorted_keys
        #plt.subplot(2,4,8)
        plt.subplot(2,1,1)
        plt.scatter(sorted_keys,stds)
        #plt.scatter(list(dat.keys()),stds)
        plt.title('Sigma: {}, Std'.format(sig))
        plt.subplot(2,1,2)
        plt.scatter(sorted_keys,means)
        #plt.scatter(list(dat.keys()),means)
        plt.title('Sigma: {}, Mean'.format(sig))
        plt.tight_layout()
        
    
    def sigma_best(self,files,sigma):
        dat = {}
        for f in files:
            thres = f.split('Threhsold')[1].split('_')[0]
            if thres in dat.keys():
                dat[thres]=dat[thres].append(pd.read_csv(f,sep='\s+'))
            else:
                dat[thres] = pd.read_csv(f,sep='\s+')
        each_rew = []
        for key in dat.keys():
            _,diffrecall = self.get_diffs(dat[key])
            each_rew.append(np.mean(diffrecall))
        self.each_rew = each_rew
        
        
    def plot_sigma_best(self,savefile):
        
        #plots = len(files) #if I want to plot each rew-threshold histogram separately.
        #i = 1 

        plt.scatter(list(dat.keys()),self.each_rew)
        plt.xlabel('Reward Threshold')
        plt.ylabel('Mean Error on Recall Trials')
        plt.title('Sigma:{}'.format(float(sigma)))
        
            
        

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
        
    def var_lesions(self,files,lesion_names):
        dat = {}
        error_dat = {}
        for l in lesion_names:
            for f in files:
                if "Lesion"+l+"_" in f:
                    if l in dat.keys():
                        dat[l]=dat[l].append(pd.read_csv(f,sep='\s+'))
                    else:
                        dat[l] = pd.read_csv(f,sep='\s+') 
        self.lesiondat = dat
        plots = len(lesion_names)
        for key in dat.keys():
            _,diffrecall = self.get_diffs(dat[key])
            error_dat[key]=diffrecall
        self.error_dat = error_dat
        
    
    def plot_var_lesions(self,savefile):
        error_dat = self.error_dat
        
        i = 1
        for key in error_dat.keys():

            plt.subplot(2,2,i)
            #plt.hist(diffrecall,range = (-0.01,0.01))
            plt.hist(error_dat[key])
            plt.title('Lesion:{}'.format(key))
            i+=1
        plt.xlabel('Error Degrees')
        plt.ylabel('Frequency')
        #plt.suptitle('Difference in Precision on Test Trials (Target-Output)')
        plt.tight_layout()
        plt.savefig(savefile)
    
    def var_lesionprop(self,files,lesion_name,lesion_props):
        dat = {}
        error_dat = {}
        for l in lesion_props:
            for f in files:
                if "LesionProp"+l+'_' in f:
                    if l in dat.keys():
                        dat[l]=dat[l].append(pd.read_csv(f,sep='\s+'))
                    else:
                        dat[l] = pd.read_csv(f,sep='\s+') 
        self.lesiondat = dat
    
        for key in dat.keys():
            _,diffrecall = self.get_diffs(dat[key])
            error_dat[key] = diffrecall
        self.error_dat = error_dat
        self.lesion_props = lesion_props
        
    
    def plot_var_lesionprop(self,savefile, bins = 45, data = None):
        if data == None: 
            error_dat = self.error_dat
        else: 
            error_dat = data
        plots = len(self.lesion_props)
        i = 1
        for key in error_dat.keys():
            plt.subplot(1,3,i)
            #plt.hist(diffrecall,range = (-0.01,0.01))
            #plt.hist(diffrecall,bins = 30, range = (-3,3))
            plt.hist(error_dat[key],bins = bins)
            plt.ylim([0,175])
            plt.title('LesionProp:{}'.format(key))
            i+=1
            
        #plt.suptitle('Difference in Precision on Test Trials (Target-Output)')
        #plt.suptitle("{} Lesions".format(lesion_name))
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
        
    