import numpy as np
from sklearn import metrics as skmetrics
import warnings
warnings.filterwarnings("ignore")

def lgb_MaF(preds, dtrain):
    Y = np.array(dtrain.get_label(), dtype=np.int32)
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'macro_f1', float(MaF(preds.shape[0], Y_pre, Y)), True

def lgb_precision(preds, dtrain):
    Y = dtrain.get_label()
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'precision', float(Counter(Y==Y_pre)[True]/len(Y)), True

class Metrictor:
    def __init__(self, classNum):
        self.classNum = classNum
        self._reporter_ = {"MaF":self.MaF,"MiF":self.MiF,"MiP":self.MiP,"MiR":self.MiR,"skMiF":self.skMiF,"MiF@50":self.MiF50,"MiF@100":self.MiF100,"MiF@150":self.MiF150,"MiF@200":self.MiF200,"MiF@250":self.MiF250,"MiF@300":self.MiF300,"MiF@350":self.MiF350,"MiF@400":self.MiF400,"MiF@450":self.MiF450,"MiF@500":self.MiF500,"MiF@1000":self.MiF1000,"P@8":self.P8,"P@5":self.P5,"P@15":self.P15,"R@10":self.R10,"R@50":self.R50,"R@100":self.R100,"R@450":self.R450,"R@80":self.R80,"R@100":self.R100,"R@500":self.R500,"R@1000":self.R1000,"ACC":self.ACC, "LOSS":self.LOSS,"MaAUC":self.MaAUC, "MiAUC":self.MiAUC,"MaMCC":self.MaMCC, "MiMCC":self.MiMCC}
    def __call__(self, report, end='\n'):
        res = {}
        for mtc in report:
            v = self._reporter_[mtc]()
            print(" %s=%6.3f"%(mtc,v), end=';')
            res[mtc] = v
        print(end=end)
        return res
    def set_data(self, Y_prob_pre, Y, threshold=0.5, multi_label=True):
        self.Y_prob_pre = Y_prob_pre.copy()
        isONE = Y_prob_pre>threshold
        self.Y_pre = Y_prob_pre.copy()
        self.Y_pre[isONE],self.Y_pre[~isONE] = 1,0
        self.Y = Y.copy()

        self.N = len(Y)
    @staticmethod
    def table_show(resList, report, rowName='CV'):
        lineLen = len(report)*8 + 6
        print("="*(lineLen//2-6) + "FINAL RESULT" + "="*(lineLen//2-6))
        print("%6s"%('-', ) + "".join(["%8d"%i for i in report]))
        for i,res in enumerate(resList):
            print("%6s"%(rowName+'_'+str(i+1)) + "".join(["%8.3f}"%res[j] for j in report]))
        print("%6s"%('MEAN') + "".join(["%8.3f"%(np.mean([res[i] for res in resList])) for i in report]))
        print("======" + "========"*len(report))
    def each_class_indictor_show(self, id2lab):
        id2lab = np.array(id2lab)
        TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(self.classNum, self.Y_pre, self.Y)
        MCCi = fill_inf((TPi*TNi - FPi*FNi) / np.sqrt( (TPi+FPi)*(TPi+FNi)*(TNi+FPi)*(TNi+FNi) ), np.nan)
        Pi = fill_inf(TPi/(TPi+FPi))
        Ri = fill_inf(TPi/(TPi+FNi))
        Fi = fill_inf(2*Pi*Ri/(Pi+Ri))
        sortedIndex = np.argsort(id2lab)
        classRate = self.Y.sum(axis=0)[sortedIndex] / self.N
        id2lab,MCCi,Pi,Ri,Fi = id2lab[sortedIndex],MCCi[sortedIndex],Pi[sortedIndex],Ri[sortedIndex],Fi[sortedIndex]
        print("-"*28 + "MACRO INDICTOR" + "-"*28)
        print("%30s%8s%8s%8s%8s%8s"%(' ','rate','MCCi','Pi','Ri','Fi'))
        for i,c in enumerate(id2lab):
            print("%30s%8.2f%8.3f%8.3f%8.3f%8.3f"%(c,classRate[i],MCCi[i],Pi[i],Ri[i],Fi[i]))
        print("-"*70)
    def MaF(self):
        return F1(self.classNum,  self.Y_pre, self.Y, average='macro')
    def MiP(self):
        return Precision(self.classNum, self.Y_pre, self.Y, average='micro')
    def MiR(self):
        return Recall(self.classNum, self.Y_pre, self.Y, average='micro') 
    def MiF(self, showInfo=False):
        return F1(self.classNum, self.Y_pre, self.Y, average='micro')
    def skMiF(self):
        return skmetrics.f1_score(self.Y, self.Y_pre, average='micro')
    def ACC(self):
        return ACC(self.classNum, self.Y_pre, self.Y)
    def MaMCC(self):
        return MCC(self.classNum, self.Y_pre, self.Y, average='macro')
    def MiMCC(self):
        return MCC(self.classNum, self.Y_pre, self.Y, average='micro')
    def MaAUC(self):
        return AUC(self.classNum, self.Y_prob_pre, self.Y, average='macro')
    def MiAUC(self):
        return AUC(self.classNum, self.Y_prob_pre, self.Y, average='micro')
    def LOSS(self):
        return LOSS(self.Y_prob_pre,self.Y)
    def P5(self):
        return PrecisionInTop(self.Y_prob_pre,self.Y, n=5)
    def P8(self):
        return PrecisionInTop(self.Y_prob_pre,self.Y, n=8)
    def P15(self):
        return PrecisionInTop(self.Y_prob_pre,self.Y, n=15)    
    def R10(self):
        return RecallInTop(self.Y_prob_pre,self.Y, n=10)
    def R50(self):
        return RecallInTop(self.Y_prob_pre,self.Y, n=50)    
    def R80(self):
        return RecallInTop(self.Y_prob_pre,self.Y, n=80)
    def R100(self):
        return RecallInTop(self.Y_prob_pre,self.Y, n=100)
    def R450(self):
        return RecallInTop(self.Y_prob_pre,self.Y, n=450)    
    def R500(self):
        return RecallInTop(self.Y_prob_pre,self.Y, n=500)
    def R1000(self):
        return RecallInTop(self.Y_prob_pre,self.Y, n=1000)        
    def MiF50(self):
        return MiF50Uper(self.Y_prob_pre,self.Y, n=50)
    def MiF100(self):
        return MiF50Uper(self.Y_prob_pre,self.Y, n=100)
    def MiF150(self):
        return MiF50Uper(self.Y_prob_pre,self.Y, n=150)
    def MiF200(self):
        return MiF50Uper(self.Y_prob_pre,self.Y, n=200)    
    def MiF250(self):
        return MiF50Uper(self.Y_prob_pre,self.Y, n=250)
    def MiF300(self):
        return MiF50Uper(self.Y_prob_pre,self.Y, n=300)
    def MiF350(self):
        return MiF50Uper(self.Y_prob_pre,self.Y, n=350)
    def MiF400(self):
        return MiF50Uper(self.Y_prob_pre,self.Y, n=400)    
    def MiF450(self):
        return MiF50Uper(self.Y_prob_pre,self.Y, n=450)
    def MiF500(self):
        return MiF50Uper(self.Y_prob_pre,self.Y, n=500)    
    def MiF1000(self):
        return MiF50Uper(self.Y_prob_pre,self.Y, n=1000)        

    
    
def MiF50Uper(Y_prob_pre, Y, n):
    Y_pre_ =(1-Y_prob_pre).argsort(axis=1)[:,:n] # 取出前50个的索引（位置）
    Y_pre = np.zeros(Y.shape)
    for i in range(len(Y_pre)):
        Y_pre[i][Y_pre_[i]] = Y[i][Y_pre_[i]]
    return skmetrics.f1_score(Y,Y_pre, average='micro')
    
def _TPiFPiTNiFNi(classNum, Y_pre, Y):
    isValid = (Y.sum(axis=0) + Y_pre.sum(axis=0))>0
    Y,Y_pre = Y[:,isValid],Y_pre[:,isValid]
    TPi = np.array([Y_pre[:,i][Y[:,i]==1].sum() for i in range(Y.shape[1])], dtype='float32')
    FPi = Y_pre.sum(axis=0) - TPi
    TNi = (1^Y).sum(axis=0) - FPi
    FNi = Y.sum(axis=0) - TPi
    return TPi,FPi,TNi,FNi

def PrecisionInTop(Y_prob_pre, Y, n):
    Y_pre = (1-Y_prob_pre).argsort(axis=1)[:,:n]
    return sum([sum(y[yp]) for yp,y in zip(Y_pre, Y)]) / (len(Y)*n)

def RecallInTop(Y_prob_pre, Y, n):
    Y_pre  =(1-Y_prob_pre).argsort(axis=1)[:,:n]
    return np.mean([sum(y[yp])/sum(y) for yp,y in zip(Y_pre,Y) if sum(y)>0])

def ACC(classNum, Y_pre, Y):
    TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    return (TPi.sum()+TNi.sum()) / (len(Y)*classNum)

def AUC(classNum, Y_prob_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    return skmetrics.roc_auc_score(Y, Y_prob_pre, average=average)

def MCC(classNum, Y_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    if average=='micro':
        TP,FP,TN,FN = TPi.sum(),FPi.sum(),TNi.sum(),FNi.sum()
        MiMCC = fill_inf((TP*TN - FP*FN) / np.sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) ), np.nan)
        return MiMCC
    else:
        MCCi = fill_inf( (TPi*TNi - FPi*FNi) / np.sqrt((TPi+FPi)*(TPi+FNi)*(TNi+FPi)*(TNi+FNi)), np.nan )
        return MCCi.mean()

def Precision(classNum, Y_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    if average=='micro':
        MiP = fill_inf(TPi.sum() / (TPi.sum() + FPi.sum()))
        return MiP
    else:
        Pi = fill_inf(TPi/(TPi+FPi))
        return Pi.mean()

def Recall(classNum, Y_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    if average=='micro':
        MiR = fill_inf(TPi.sum() / (TPi.sum() + FNi.sum()))
        return MiR
    else:
        Ri = fill_inf(TPi/(TPi + FNi))
        return Ri.mean()

def F1(classNum, Y_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    if average=='micro':
        MiP,MiR = Precision(classNum, Y_pre, Y, average='micro'),Recall(classNum, Y_pre, Y, average='micro')
        MiF = fill_inf(2*MiP*MiR/(MiP+MiR))
        return MiF
    else:
        TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
        Pi,Ri = TPi/(TPi + FPi),TPi/(TPi + FNi)
        Pi[Pi==np.inf],Ri[Ri==np.inf] = 0.0,0.0
        Fi = fill_inf(2*Pi*Ri/(Pi+Ri))
        return Fi.mean()

def LOSS(Y_prob_pre, Y):
    Y = Y.reshape(-1,1)
    Y_prob_pre = Y_prob_pre.reshape(len(Y),-1)
    Y_prob_pre[Y_prob_pre>0.99] -= 1e-3
    Y_prob_pre[Y_prob_pre<0.01] += 1e-3
    lossArr = Y*np.log(Y_prob_pre) + (1-Y)*np.log(1-Y_prob_pre)
    return -lossArr.mean(axis=1).mean()

from collections import Iterable
def fill_inf(x, v=0.0):
    if isinstance(x, Iterable):
        x[x==np.inf] = v
        x[np.isnan(x)] = v
    else:
        x = v if x==np.inf else x
        x = v if np.isnan(x) else x
    return x

