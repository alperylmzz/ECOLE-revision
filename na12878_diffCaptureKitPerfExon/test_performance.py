import numpy as np
from sklearn.metrics import confusion_matrix as cm
import os
from tqdm import tqdm
import pandas as pd

cur_dirname = os.path.dirname(__file__)

preds_ = np.array([])
wgs_preds_ = np.array([])

    
ecole_calls_data = pd.read_csv("NA12878_HISEQ4000.csv" , sep=",",header=None)
wgs_calls_data = pd.read_csv("Groundtruth_NA12878.csv", sep=",",header=None)
new_df = pd.merge(wgs_calls_data, ecole_calls_data,  how='left', left_on=[0,1,2], right_on = [0,1,2]).values
wgs_preds_n = np.array(new_df[:,3]) 
wgs_preds_ = np.append(wgs_preds_,wgs_preds_n)
preds_n= np.array(new_df[:,4]) 
preds_ = np.append(preds_,preds_n)

        
wgs_preds_[wgs_preds_ == "<DEL>"] = 2
wgs_preds_[wgs_preds_ == "<DUP>"] = 1
wgs_preds_[wgs_preds_ == "<NO-CALL>"] = 0
wgs_preds_ = wgs_preds_.astype(int)


delcalls = wgs_preds_ == 2
dupcalls = wgs_preds_ == 1
nocallcalls = wgs_preds_ == 0

preds_ = preds_.astype(int)

delpreds_ = preds_ == 2
duppreds_  = preds_ == 1
nocallpreds_  = preds_ == 0

delrecall_  = np.sum(delpreds_ * delcalls) / np.sum(delcalls)
duprecall_  = np.sum(duppreds_  * dupcalls) / np.sum(dupcalls)
print("deletion recall:", delrecall_)
print("duplication recall:", duprecall_)

delprec_  = np.sum(delpreds_  * delcalls) / np.sum(delpreds_ )
dupprec_  = np.sum(duppreds_  * dupcalls) / np.sum(duppreds_ )
print("deletion precision:", delprec_)
print("duplication precision: ",dupprec_ )

print("confusion matrix: \n",cm(wgs_preds_, preds_))


