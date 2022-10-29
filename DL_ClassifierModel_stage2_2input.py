import torch,time,os,pickle,sys
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import *
from nnLayer import  *
from metrics import *
from torch.nn.init import xavier_uniform_ as xavier_uniform
from math import floor
from tqdm import tqdm
from pytorch_lamb import lamb
from sklearn.metrics import roc_auc_score


class FGM():
    def __init__(self, model, emb_name='emb'):
        self.model = model
        self.emb_name = emb_name
        self.backup = {}
 
    def attack(self, epsilon=1.):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
 
    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class BaseModel:
    def __init__(self):
        pass
    def calculate_y_logit(self):
        pass
    def train(self, dataClass, batchSize, epoch, 
              lr=0.001, momentum=0.9, weightDecay=0.0, stopRounds=10, threshold=0.2, earlyStop=10, 
              savePath='model/KAICD', saveRounds=1, isHigherBetter=True, metrics="MiF", report=["ACC", "MiF"], 
              optimType='Adam',schedulerType='cosine',eta_min=0,warmup_ratio=0.1,dataEnhance=False, dataEnhanceRatio=0.0, attackTrain=False, attackLayerName='emb',ema_para = -1,candidate_para=False):
        dataClass.dataEnhance = dataEnhance
        dataClass.dataEnhanceRatio = dataEnhanceRatio

        if attackTrain:
            self.fgm = FGM(self.moduleList, emb_name=attackLayerName)

        if ema_para>0:
            self.ema = EMA(self.moduleList, ema_para)
        isBeginEMA = False

        metrictor = Metrictor(dataClass.classNum)

        trainStream = dataClass.random_batch_data_stream(batchSize=batchSize, type='train', device=self.device,candidate=candidate_para)

        itersPerEpoch = (dataClass.trainSampleNum+batchSize-1)//batchSize

        num_training_steps = itersPerEpoch*epoch
        num_warmup_steps = int(warmup_ratio*itersPerEpoch*epoch)
  
        optimizer,schedulerRLR = self.get_optimizer(optimType, schedulerType, lr, weightDecay, momentum, num_training_steps,num_warmup_steps,eta_min)
        
        mtc,bestMtc,stopSteps = 0.0,-1,0

        if dataClass.validSampleNum>0:
            validStream = dataClass.random_batch_data_stream(batchSize=batchSize, type='valid', device=self.device,candidate=candidate_para)
 
        st = time.time()
        
        for e in range(epoch):
            print(f"Epoch {e+1} with learning rate {optimizer.state_dict()['param_groups'][0]['lr']:.6f}...")
            print('========== Epoch:%5d =========='%(e+1))
            if (ema_para>0) and (e>30) and (not isBeginEMA): # 
                self.ema.register()
                isBeginEMA = True            

            pbar = tqdm(range(itersPerEpoch))
            for i in pbar:
                self.to_train_mode()
                X, Y, candidate= next(trainStream)
                loss = self._train_step(X, Y,candidate,optimizer, attackTrain, isBeginEMA)
                if schedulerRLR !=None:
                    schedulerRLR.step()
                pbar.set_description(f"Epoch {e} - Training Loss: {loss.data:.3f}")
                if stopRounds>0 and (e*itersPerEpoch+i+1)%stopRounds==0:
                    self.to_eval_mode()
                    print("After iters %d: [train] loss= %.3f;"%(e*itersPerEpoch+i+1,loss), end='')
                    if dataClass.validSampleNum>0:
                        X, Y = next(validStream)
                        loss = self.calculate_loss(X,Y)
                        print(' [valid] loss= %.3f;'%loss, end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*batchSize
                    speed = (e*itersPerEpoch+i+1)*batchSize/(time.time()-st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;"%(speed, restNum/speed))
            
            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
                if isBeginEMA:
                    self.ema.apply_shadow()
                self.to_eval_mode()
                print('[Total Valid]', end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(batchSize, type='valid', device=self.device,candidate=candidate_para))
                
                metrictor.set_data(Y_pre, Y, threshold)
                res = metrictor(report)
                mtc = res[metrics]
                print('=================================')
                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                    print('Bingo!!! Get a better Model with val %s: %.3f!!!'%(metrics,mtc))
                    bestMtc = mtc
                    self.save("%s.pkl"%savePath, e+1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps>=earlyStop:
                        print('The val %s has not improved for more than %d steps in epoch %d, stop training.'%(metrics,earlyStop,e+1))
                        break
            if isBeginEMA:
                self.ema.restore()
        self.load("%s.pkl"%savePath, dataClass=dataClass)
        os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))
        
        with torch.no_grad():
            print(f'============ Result ============')
            print(f'[Total Train]',end='')
            Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(batchSize, type='train', device=self.device,candidate=candidate_para))
            metrictor.set_data(Y_pre, Y, threshold)
            metrictor(report)
            print(f'[Total Valid]',end='')
            Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(batchSize, type='valid', device=self.device,candidate=candidate_para))
            metrictor.set_data(Y_pre, Y, threshold)
            res = metrictor(report)
            print(f'================================')
        return res
    
    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()

    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            stateDict['nword2id'],stateDict['tword2id'] = dataClass.nword2id,dataClass.tword2id
            stateDict['id2nword'],stateDict['id2tword'] = dataClass.id2nword,dataClass.id2tword
            stateDict['icd2id'],stateDict['id2icd'] = dataClass.icd2id,dataClass.id2icd
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            dataClass.trainIdList = parameters['trainIdList']
            dataClass.validIdList = parameters['validIdList']
            dataClass.testIdList = parameters['testIdList']

            dataClass.nword2id,dataClass.tword2id = parameters['nword2id'],parameters['tword2id']
            dataClass.id2nword,dataClass.id2tword = parameters['id2nword'],parameters['id2tword']
            dataClass.id2icd,dataClass.icd2id = parameters['icd2id'],parameters['id2icd']     
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))
    
    def calculate_y_prob(self, X, candidate):
        Y_pre = self.calculate_y_logit(X,candidate)['y_logit']
        return torch.sigmoid(Y_pre)
    def calculate_y(self, X, threshold=0.2):
        Y_pre = self.calculate_y_prob(X)
        isONE = Y_pre>threshold
        Y_pre[isONE],Y_pre[~isONE] = 1,0
        return Y_pre
    def calculate_loss(self, X, Y, candidate):
        out = self.calculate_y_logit(X,candidate)
        Y_logit = out['y_logit']
        
        addLoss = 0.0
        if 'loss' in out: addLoss += out['loss']
        return self.crition(Y_logit, Y) + addLoss
    def calculate_indicator_by_iterator(self, dataStream, classNum, report, threshold):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        Metrictor.set_data(Y_prob_pre, Y, threshold)
        return metrictor(report)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y,candidate = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X,candidate).cpu().data.numpy().astype(np.float16),Y.cpu().data.numpy().astype(np.int32)
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr),np.vstack(Y_preArr)
        return Y_preArr, YArr
    def calculate_y_by_iterator(self, dataStream, threshold=0.2):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        isONE = Y_preArr>threshold
        Y_preArr[isONE],Y_preArr[~isONE] = 1,0
        return Y_preArr, YArr
    def to_train_mode(self):
        for module in self.moduleList:
            module.train()
    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()
    def _train_step(self, X, Y,candidate, optimizer, attackTrain,isBeginEMA):
        loss = self.calculate_loss(X, Y, candidate)
        loss.backward()
        if attackTrain:
            self.fgm.attack() # 在embedding上添加对抗梯度
            lossAdv = self.calculate_loss(X, Y,candidate)
            lossAdv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            self.fgm.restore() # 恢复embedding参数
        nn.utils.clip_grad_norm_(self.moduleList.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        if isBeginEMA:
            self.ema.update()
        optimizer.zero_grad()
        return loss

    def get_optimizer(self, optimType, schedulerType, lr, weightDecay, momentum,num_training_steps,num_warmup_steps,eta_min):     
        
        # Prepare optimizer and schedule (linear warmup and decay)
        model_lr={'others':lr}
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        for layer_name in model_lr:
            lr = model_lr[layer_name]    
            if layer_name != 'others':  
                 optimizer_grouped_parameters += [
                    {
                        "params": [p for n, p in self.moduleList.named_parameters() if (not any(nd in n for nd in no_decay) 
                                                                              and layer_name in n)],
                        "weight_decay": weightDecay,
                        "lr": lr,
                    },
                    {
                        "params": [p for n, p in self.moduleList.named_parameters() if (any(nd in n for nd in no_decay) 
                                                                              and layer_name in n)],
                        "weight_decay": 0.0,
                        "lr": lr,
                    },
                 ]
            else:  
                optimizer_grouped_parameters += [
                    {
                        "params": [p for n, p in self.moduleList.named_parameters() if (not any(nd in n for nd in no_decay) 
                                                                              and not any(name in n for name in model_lr))],
                        "weight_decay": weightDecay,
                        "lr": lr,
                    },
                    {
                        "params": [p for n, p in self.moduleList.named_parameters() if (any(nd in n for nd in no_decay) 
                                                                              and not any(name in n for name in model_lr))],
                        "weight_decay": 0.0,
                        "lr": lr,
                    },
                ]
             
        if optimType=='Adam':
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr, weight_decay=weightDecay)
        elif optimType=='AdamW':
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=weightDecay)
        elif optimType=='Lamb':
            optimizer = lamb.Lamb(optimizer_grouped_parameters, weight_decay=weightDecay)
        elif optimType=='SGD':
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=momentum, weight_decay=weightDecay)
        elif optimType=='Adadelta':
            optimizer = torch.optim.Adadelta(optimizer_grouped_parameters, lr=lr, weight_decay=weightDecay)        

        print("len(optimizer_grouped_parameters))")
        print(len(optimizer_grouped_parameters))

        if schedulerType=='cosine':
            schedulerRLR = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)        
        elif schedulerType=='cosine_Anneal':
            schedulerRLR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, T_mult=3,eta_min=eta_min)
        elif schedulerType=='None':
            schedulerRLR = None

        return optimizer, schedulerRLR



class FLASH_ICD_Candidates_2Inputs(BaseModel):
    def __init__(self,classNum,embedding,labDescVec,seqMaxLen,chunk_length,trans_s,attnList=[],
                 embDropout=0.2, hdnDropout=0.2,Dropout=0.3, fcDropout=0.5, numLayers=1, device=torch.device('cuda:0'), useCircleLoss=False, compress=False):
        self.embedding = TextEmbedding_1d(embedding, dropout=embDropout).to(device)
        self.labDescVec = torch.tensor(labDescVec, dtype=torch.float32).to(device)
        self.FLASH = FLASH(seqMaxLen, embedding.shape[1], numLayers,chunk_length,trans_s).to(device)
        self.icdAttn = DeepICDDescCandiAttention(embedding.shape[1]*2, classNum, labDescVec.shape[1], hdnDropout=hdnDropout, attnList=attnList, labDescVec=labDescVec).to(device)
        self.fcLinear = MLP(labDescVec.shape[1], 1, [], dropout=fcDropout).to(device)       
        self.moduleList = nn.ModuleList([self.embedding,self.FLASH,self.icdAttn,self.fcLinear])
        self.crition = nn.MultiLabelSoftMarginLoss() if not useCircleLoss else MultiLabelCircleLoss()
        self.device = device
        self.hdnDropout = hdnDropout
        self.fcDropout = fcDropout
        self.classNum = classNum
    def calculate_y_logit(self, input,candidate):
        x = input['noteArr']
        x = self.embedding(x)                   # => batchSize × seqLen × embSize
        # Linear Transformer
        x = self.FLASH(x)                         # => batchSize × SeqLen × embSize
        # x = self.LNandDP(x) # => batchSize × seqLen X embSize

        x = self.icdAttn(x,candidate)                     # => batchSize × Candidates Num (1000) × inSize
        x = self.fcLinear(x).squeeze(dim=2)     # => batchSize × Candidates Num (1000)
        return {'y_logit':x}

    def calculate_y_prob(self, X, candidate):
        Y_pre_Candidates = torch.sigmoid(self.calculate_y_logit(X,candidate)['y_logit'])
        Zero_matrix = torch.zeros(Y_pre_Candidates.size(0),self.classNum,device=self.device)
        Y_pre = Zero_matrix.scatter_(dim=1,index=torch.tensor(candidate,device=self.device),src=Y_pre_Candidates.to(self.device))
        return Y_pre     