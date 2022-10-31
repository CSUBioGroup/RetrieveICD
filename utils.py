import pickle, random, torch, logging, time, gensim, os, re, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts
from tqdm import tqdm
from collections import Counter
from glove import Glove
from glove import Corpus
# from transformers import RobertaModel,RobertaTokenizer,BertTokenizer, BertForMaskedLM, BertConfig, BertLMHeadModel, AutoTokenizer, AutoModelWithLMHead, AutoModelForMaskedLM, XLNetLMHeadModel, BertModel

# from radical import Radical


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class DataClass:
    def __init__(self, dataPath, mimicPath='mimic3/', stopWordPath="stopwords.txt", validSize=0.2, testSize=0.0,
                 minCount=10, noteMaxLen=768, seed=9527, topICD=-1, samples=-1):
        term_pattern = re.compile('[A-Za-z0-9]+|[,;.!?()]|<br>|<:>', re.I)
        self.minCount = minCount
        validSize *= 1.0 / (1.0 - testSize)
        # Open files and load data
        print('Loading the data...')
        data = pd.read_csv(dataPath, usecols=['HADM_ID', 'TEXT', 'ICD9_CODE'])
        if samples > 0:
            data = data.sample(n=samples)
        # Get word-splited notes and icd codes
        print('Getting the word-splited notes and icd codes...')
        self.hadmId = data['HADM_ID'].values.tolist()
        NOTE, ICD = [term_pattern.findall(i) for i in tqdm(data['TEXT'])], list(data['ICD9_CODE'].values)
        self.rawNOTE = [i + ["<EOS>"] for i in NOTE]
        del data
        gc.collect()
        # Calculate the word count
        print('Calculating the word count...')
        wordCount = Counter()
        for s in tqdm(NOTE):
            wordCount += Counter(s)
        # Drop low-frequency words and stopwords
        with open(stopWordPath, 'r') as f:
            stopWords = [i[:-1].lower() for i in f.readlines()]
        for i, s in tqdm(enumerate(NOTE)):
            NOTE[i] = [w if ((wordCount[w] >= minCount) and (w not in stopWords) and (len(w) > 2)) else "<UNK>" for w in
                       s]
        # Drop invalid data
        keepIndexs = np.array([len(i) for i in NOTE]) > 0
        print('Find %d invalid data, drop them!' % (sum(~keepIndexs)))
        NOTE, ICD = list(np.array(NOTE)[keepIndexs]), list(np.array(ICD)[keepIndexs])
        # Drop low TF-IDF words
        print('Dropping the unimportant words...')
        NOTE = self._drop_unimportant_words(NOTE, noteMaxLen)
        self.notes = [i + ['<EOS>'] for i in NOTE]
        # Get the mapping variables for note-word and id
        print('Getting the mapping variables for note-word and id...')
        self.nword2id, self.id2nword = {"<EOS>": 0, "<UNK>": 1}, ["<EOS>", "<UNK>"]
        cnt = 2
        for note in tqdm(self.notes):
            for w in note:
                if w not in self.nword2id:
                    self.nword2id[w] = cnt
                    self.id2nword.append(w)
                    cnt += 1
        self.nwordNum = cnt
        # Get mapping variables for icd and id
        print('Getting the mapping variables for icd and id...')
        self.icd2id, self.id2icd = {}, []
        cnt, tmp = 0, []
        for icds in ICD:
            icds = icds.split(';')
            for icd in icds:
                if icd not in self.icd2id:
                    self.icd2id[icd] = cnt
                    self.id2icd.append(icd)
                    cnt += 1
            tmp.append([self.icd2id[icd] for icd in icds])
        self.icdNum = cnt
        self.Lab = np.zeros((len(ICD), cnt), dtype='int32')
        for i, icds in enumerate(tmp):
            self.Lab[i, icds] = 1
        if topICD > 0:
            icdCtr = self.Lab.sum(axis=0)
            usedIndex = np.argsort(icdCtr)[-topICD:]
            self.Lab = self.Lab[:, usedIndex]
            self.icdNum = topICD
            self.icdIndex = usedIndex
        # Get the mapping variables for title-word and id
        print('Getting the mapping variables for title-word and id...')
        self.tword2id, self.id2tword = {"<EOS>": 0}, ["<EOS>"]
        cnt = 1

        dIcdDiagnoses = pd.read_csv(os.path.join(mimicPath, 'D_ICD_DIAGNOSES.csv'))
        dIcdDiagnoses['ICD9_CODE'] = 'dia_' + dIcdDiagnoses['ICD9_CODE'].astype('str')
        dIcdDiagnoses = dIcdDiagnoses.set_index('ICD9_CODE')
        dicdProcedures = pd.read_csv(os.path.join(mimicPath, 'D_ICD_PROCEDURES.csv'))
        dicdProcedures['ICD9_CODE'] = 'pro_' + dicdProcedures['ICD9_CODE'].astype('str')
        dicdProcedures = dicdProcedures.set_index('ICD9_CODE')
        icdTitles = pd.concat([dIcdDiagnoses, dicdProcedures])
        self.titles = []
        for icd in self.id2icd:
            try:
                desc = (icdTitles.loc[icd]['SHORT_TITLE'] + ' <:> ' + icdTitles.loc[icd]['LONG_TITLE']).lower().split()
            except:
                desc = " <:> ".split()
            self.titles.append(desc + ["<EOS>"])
            for w in desc:
                if w not in self.tword2id:
                    self.tword2id[w] = cnt
                    self.id2tword.append(w)
                    cnt += 1
        self.titleLen = [len(i) for i in self.titles]
        titleMaxLen = max(self.titleLen)
        self.twordNum = cnt
        # Tokenize the notes and titles
        print('Tokenizing the notes and the titles...')
        self.tokenizedNote = np.array([[self.nword2id[w] for w in n] for n in tqdm(self.notes)], dtype='int32')
        self.tokenizedTitle = np.array(
            [[self.tword2id[w] for w in t] + [0] * (titleMaxLen - len(t)) for t in self.titles], dtype='int32')
        # Get some variables might be used
        self.totalSampleNum = len(self.tokenizedNote)
        restIdList, testIdList = train_test_split(range(self.totalSampleNum), test_size=testSize,
                                                  random_state=seed) if testSize > 0.0 else (
        list(range(self.totalSampleNum)), [])
        trainIdList, validIdList = train_test_split(restIdList, test_size=validSize,
                                                    random_state=seed) if validSize > 0.0 else (restIdList, [])

        self.trainIdList, self.validIdList, self.testIdList = trainIdList, validIdList, testIdList
        self.trainSampleNum, self.validSampleNum, self.testSampleNum = len(self.trainIdList), len(
            self.validIdList), len(self.testIdList)

        self.classNum, self.vector = self.icdNum, {}

        self.dataEnhance = False
        self.dataEnhanceRatio = 0.0

    def change_seed(self, seed=20201247, validSize=0.2, testSize=0.0):
        restIdList, testIdList = train_test_split(range(self.totalSampleNum), test_size=testSize,
                                                  random_state=seed) if testSize > 0.0 else (
        list(range(self.totalSampleNum)), [])
        trainIdList, validIdList = train_test_split(restIdList, test_size=validSize,
                                                    random_state=seed) if validSize > 0.0 else (restIdList, [])

        self.trainIdList, self.validIdList, self.testIdList = trainIdList, validIdList, testIdList
        self.trainSampleNum, self.validSampleNum, self.testSampleNum = len(self.trainIdList), len(
            self.validIdList), len(self.testIdList)

    def vectorize(self, method=["skipgram"], noteFeaSize=320, titleFeaSize=192, window=5, sg=1, iters=10,batchWords=1000000,noteCorpusPath=None, workers=8, loadCache=True, suf=""):
        path = 'wordEmbedding/note_%s_d%d%s.pkl' % (method, noteFeaSize, suf)
        if os.path.exists(path) and loadCache:
            with open(path, 'rb') as f:
                self.vector['noteEmbedding'] = pickle.load(f)
            print('Loaded cache from cache/%s' % path)
        else:
            corpus = self.rawNOTE if noteCorpusPath is None else LineSentence(noteCorpusPath)
            embeddings = []
            if 'skipgram' in method:
                model = Word2Vec(corpus, min_count=self.minCount, window=window, vector_size=noteFeaSize, workers=workers,
                                 sg=1, epochs=iters, batch_words=batchWords)
                word2vec = np.zeros((self.nwordNum, noteFeaSize), dtype=np.float32)
                for i in range(self.nwordNum):
                    if self.id2nword[i] in model.wv:
                        word2vec[i] = model.wv[self.id2nword[i]]
                    else:
                        print('word %s not in word2vec.' % self.id2nword[i])
                        word2vec[i] = np.random.random(noteFeaSize)
                embeddings.append(word2vec)
                # self.vector['noteEmbedding'] = word2vec
            if 'glove' in method:
                gCorpus = Corpus()
                gCorpus.fit(corpus, window=window)
                glove = Glove(no_components=noteFeaSize)
                glove.fit(gCorpus.matrix, epochs=iters, no_threads=workers, verbose=True)
                glove.add_dictionary(gCorpus.dictionary)
                word2vec = np.zeros((self.nwordNum, noteFeaSize), dtype=np.float32)
                for i in range(self.nwordNum):
                    if self.id2nword[i] in glove.dictionary:
                        word2vec[i] =glove.word_vectors[glove.dictionary[self.id2nword[i]]]
                    else:
                        print('word %s not in word2vec.' % self.id2nword[i])
                        word2vec[i] = np.random.random(noteFeaSize)
                embeddings.append(word2vec)
            if 'fasttext' in method:
                model =  FastText(corpus, min_count=self.minCount, window=window, vector_size=noteFeaSize, workers=workers,
                                 sg=1, epochs=iters, batch_words=batchWords)
                word2vec = np.zeros((self.nwordNum, noteFeaSize), dtype=np.float32)
                for i in range(self.nwordNum):
                    if self.id2nword[i] in model.wv:
                        word2vec[i] = model.wv[self.id2nword[i]]
                    else:
                        print('word %s not in word2vec.' % self.id2nword[i])
                        word2vec[i] = np.random.random(noteFeaSize)
                embeddings.append(word2vec)
            self.vector['noteEmbedding'] = np.hstack(embeddings)
            with open(path, 'wb') as f:
                pickle.dump(self.vector['noteEmbedding'], f, protocol=4)

    def random_batch_data_stream(self, batchSize=128, type='train', device=torch.device('cpu'), candidate=False):
        if type == 'train':
            idList = list(self.trainIdList)
        elif type == 'valid':
            idList = list(self.validIdList)
        elif type == 'test':
            idList = list(self.testIdList)
        noteLen = np.array((self.tokenizedNote == self.nword2id["<EOS>"]).argmax(axis=1) + 1, dtype=np.int32)
        while True:
            random.shuffle(idList)
            for i in range((len(idList) + batchSize - 1) // batchSize):
                samples = idList[i * batchSize:(i + 1) * batchSize]

                batchNoteArr = torch.tensor(self.tokenizedNote[samples], dtype=torch.long)
                batchNoteLab = torch.tensor(self.Lab[samples].sum(axis=1), dtype=torch.int32)
                batchNoteLen = torch.tensor(noteLen[samples], dtype=torch.int32)
                if self.dataEnhance:
                    # print("dataEnhanceRatio: {}".format(self.dataEnhanceRatio))
                    for sampleId in range(len(batchNoteArr)):  # 数据增强
                        if random.random() < self.dataEnhanceRatio / 2:  # 随机排列
                            batchNoteArr[sampleId][:batchNoteLen[sampleId]] = \
                            batchNoteArr[sampleId][:batchNoteLen[sampleId]][
                                np.random.permutation(int(batchNoteLen[sampleId]))]
                        if random.random() < self.dataEnhanceRatio:  # 逆置
                            batchNoteArr[sampleId][:batchNoteLen[sampleId]] = \
                            batchNoteArr[sampleId][:batchNoteLen[sampleId]][range(int(batchNoteLen[sampleId]))[::-1]]
                if candidate:
                    candi = np.array([candidate[key] for key in samples])
                    lab_matrix = torch.gather(torch.tensor(self.Lab[samples],dtype=torch.float, device=device),1,torch.tensor(candi,device=device))
                else:
                    lab_matrix=torch.tensor(self.Lab[samples], dtype=torch.float, device=device)
                    candi=False
                yield {
                          "noteArr": batchNoteArr.to(device), \
                          "lab": batchNoteLab.to(device), \
                          "noteLen": batchNoteLen.to(device), \
                          "noteIdx": torch.tensor(samples, dtype=torch.long, device=device),
                      },lab_matrix,candi

    def one_epoch_batch_data_stream(self, batchSize=128, type='valid', device=torch.device('cpu'), candidate=False):
        if type == 'train':
            idList = self.trainIdList
        elif type == 'valid':
            idList = self.validIdList
        elif type == 'test':
            idList = self.testIdList
        noteLen = np.array((self.tokenizedNote == self.nword2id["<EOS>"]).argmax(axis=1) + 1, dtype=np.int32)
        for i in range((len(idList) + batchSize - 1) // batchSize):
            samples = idList[i * batchSize:(i + 1) * batchSize]
            if candidate:              
                candi = np.array([candidate[key] for key in samples])
                yield {
                          "noteArr": torch.tensor(self.tokenizedNote[samples], dtype=torch.long, device=device), \
                          "lab": torch.tensor(self.Lab[samples], dtype=torch.int, device=device), \
                          "noteLen": torch.tensor(noteLen[samples], dtype=torch.float32), \
                          "noteIdx": torch.tensor(samples, dtype=torch.long, device=device), \
                          }, torch.tensor(self.Lab[samples], dtype=torch.float, device=device),candi                
            else:  
                yield {
                          "noteArr": torch.tensor(self.tokenizedNote[samples], dtype=torch.long, device=device), \
                          "lab": torch.tensor(self.Lab[samples], dtype=torch.int, device=device), \
                          "noteLen": torch.tensor(noteLen[samples], dtype=torch.float32), \
                          "noteIdx": torch.tensor(samples, dtype=torch.long, device=device), \
                          }, torch.tensor(self.Lab[samples], dtype=torch.float, device=device)


