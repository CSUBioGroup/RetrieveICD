# Retrieve and Rank 

Retrieve and Rerank for Automated ICD Coding via Contrastive Learning

## Preprocess data

follow [LD-PLAM](https://github.com/CSUBioGroup/LD-PLAM) to preprocess the dataset.

import the package:

```python
from utils import *
from DL_ClassifierModel import *
from DL_ClassifierModel_stage2_2input import *
```

## Retrieval: Generate Candidates in 1st Stage

```python
labDescVec = -1+2*np.random.random((dataClass.icdNum,1024))
model = DeepLabeler_Contrast(dataClass.classNum, dataClass.vector['noteEmbedding'], docEmbedding, labDescVec,cnnHiddenSize=256, contextSizeList=[3,4,5],
                           docHiddenSize=256,dropout=0.75, device=torch.device('cuda:0'),temp_para=0.05)
savePath = 'Replace with your path'
model.train(dataClass, batchSize=128, epoch=128,
            lr=0.01, stopRounds=-1, threshold=0.5, earlyStop=64, optimType='Lamb',schedulerType='cosine',warmup_ratio=0.03,
            savePath=savePath,metrics="MiAUC", report=[ "MiAUC","MiF@50","MiF@100","MiF@500","MiF@1000"])
```

#### Retrieve the closest ICD codes as candidates and forward them to the next stage：

```python
Candidate_num = 1000
model_load = DeepLabeler_Contrast(dataClass.classNum, dataClass.vector['noteEmbedding'], docEmbedding, labDescVec,cnnHiddenSize=256, contextSizeList=[3,4,5],
                           docHiddenSize=256,dropout=0.75, device=torch.device('cuda:0'),temp_para=0.05)
model_load.load(path="Replace with your path",map_location="cpu",dataClass = dataClass)
model_load.to_eval_mode()
Y_pre_train_stage1,Y_train_stage1 = model_load.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(64, type='train', device=torch.device("cuda:0")))
Y_pre_valid_stage1,Y_valid_stage1 = model_load.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(64, type='valid', device=torch.device("cuda:0")))
Y_pre_cnn = np.vstack((Y_pre_train_stage1, Y_pre_valid_stage1))
sortIdx_t = np.argsort(-Y_pre_cnn,axis=-1)[:,:Candidate_num]
IdList=np.concatenate((dataClass.trainIdList,dataClass.validIdList))
candi_t={}
for index, item in enumerate(IdList):
    candi_t[item] = sortIdx_t[index]
```

## Rekanking: Refine and rerank the candidate set in 2nd Stage

Use a BERT-based pretrained language model to encode ICD descriptions and obtain label vectors

> the shape will be (M x 1024, where M represents the number of Codes) 

```python
labDescVec = get_ICD_vectors(dataClass=dataClass, mimicPath="path to mimic3")
```

##### train the Transformer variant with linear complexity over the context size to extract latent semantic features of EMRs

semantic features of EMRs.

`without candidates:`

```python
model_FLASH= FLASH_ICD_FULL(dataClass.classNum, dataClass.vector['noteEmbedding'],labDescVec,seqMaxLen=4000,chunk_length=400, trans_s=300,attnList=[512],embDropout=0.2, hdnDropout=0.2, fcDropout=0.0,numLayers=2,device=torch.device("cuda:0"))
savePath = 'Replace with your path'
res = model_FLASH.train(dataClass, batchSize=64, epoch=128,
            lr= 0.003,stopRounds=-1, threshold=0.5, earlyStop=64, optimType='Lamb',schedulerType='cosine_Anneal',eta_min=0.0,
            savePath=savePath,dataEnhance=True, dataEnhanceRatio=0.2, attackTrain=True, metrics="MiF", report=[ "MiAUC","MiF","P@5","P@8","P@15"])
```

`with candidates:`

```python
model_FLASH_Stage2= FLASH_ICD_Candidates_2Inputs(dataClass.classNum, dataClass.vector['noteEmbedding'],labDescVec,seqMaxLen=4000,chunk_length=400, trans_s = 300,attnList=[512],embDropout=0.2, hdnDropout=0.2,Dropout=0.0, fcDropout=0.0,numLayers=2,device=torch.device("cuda:0"))

savePath = 'Replace with your path'
res = model_FLASH_Stage2.train(dataClass, batchSize=64, epoch=128,
            lr= 0.003,stopRounds=-1, threshold=0.5, earlyStop=64, optimType='Lamb',schedulerType='cosine_Anneal',eta_min=0.0,
            savePath=savePath,dataEnhance=True, dataEnhanceRatio=0.2, attackTrain=True, metrics="MiF", report=[ "MiAUC","MiF","P@5","P@8","P@15"], candidate_para=candi_t)
```

## Make final prediction

Change `dataClass` load path to where your preprocessed dataClass located.

Change model weight path to where your model saved.

```python
python inference.py
```

