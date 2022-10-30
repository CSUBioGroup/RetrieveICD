import torch
from utils import *
from DL_ClassifierModel import *
from DL_ClassifierModel_stage2_2input import *
import numpy as np
import os
import argparse
from torch.backends import cudnn

device=torch.device("cuda:0")
dataClass = torch.load('Replace with your dataClass Path...')

Candidate_num = 1000
model_cnn = DeepLabeler_Contrast(dataClass.classNum, dataClass.vector['noteEmbedding'], docEmbedding, labDescVec,cnnHiddenSize=256, contextSizeList=[3,4,5],
                           docHiddenSize=256,dropout=0.75, device=device,temp_para=0.05)
model_cnn.load(path="Replace with saved model weight path in 1st stage",map_location="cpu",dataClass = dataClass)
model_cnn.to_eval_mode()
Y_pre_test_cnn,Y_test_cnn = model_cnn.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(16, type='test', device=device))

testIdx_cnn = np.argsort(-Y_pre_test_cnn,axis=-1)[:,:Candidate_num] 
print(testIdx_cnn.shape)
candi_t={}
for index, item in enumerate(dataClass.testIdList):
    candi_t[item] = testIdx_cnn[index]
    
model_FLASH_Stage2= FLASH_ICD_Candidates_2Inputs(dataClass.classNum, dataClass.vector['noteEmbedding'],labDescVec,seqMaxLen=4000,chunk_length=400, trans_s = 300,attnList=[512],embDropout=0.2, hdnDropout=0.2,Dropout=0.0, fcDropout=0.0,numLayers=2,device=device)
model_FLASH_Stage2.load(path="Replace with saved model weight path in 2st stage",map_location="cpu",dataClass = dataClass)
model_FLASH_Stage2.to_eval_mode()
Y_pre_test_2stage,Y_test_2stage = model_FLASH_Stage2.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(16, type='test', device=device,candidate=candi_t))

model_FLASH_base = FLASH_ICD_FULL(dataClass.classNum, dataClass.vector['noteEmbedding'],labDescVec,seqMaxLen=4000,chunk_length=400, trans_s=300,attnList=[512],embDropout=0.2, hdnDropout=0.2, fcDropout=0.0,numLayers=2,device=device)
model_FLASH_base.load(path="Replace with saved FLASH model weight path without candidates",map_location="cpu",dataClass = dataClass)
model_FLASH_base.to_eval_mode()
Y_pre_test_base,Y_test_base = model_FLASH_base.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(16, type='test', device=torch.device("cuda:1")))

actue_matrix = torch.gather(torch.tensor(Y_pre_test_2stage),1,torch.tensor(testIdx_cnn)).to(device)
base_matrix = torch.tensor(Y_pre_test_base).to(device)
final_pre = base_matrix.scatter_(dim=1,index=torch.tensor(testIdx_cnn,device=device),src=actue_matrix)

metrictor = Metrictor(dataClass.classNum)
metrictor.set_data(final_pre.cpu().data.numpy().astype(np.float32), Y_test_2stage, 0.45)
report=[ "MiAUC","MiF","MiP","MiR","P@5","P@8","P@15"]
res = metrictor(report)
