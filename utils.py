SEED = 42

import logging
logging.captureWarnings(True)
from transformers import logging
logging.set_verbosity_error()

import os
from os.path import dirname as up
import sys
import numpy as np
import pickle 
import random
from collections import defaultdict, Counter
import copy
import gc
import re
from scipy.stats import entropy
import scipy.stats as st
from matplotlib import pylab
import pandas as pd
from scipy.cluster.vq import kmeans,vq
import time

import transformers
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import BertModel

import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier as SGDC
from sklearn.svm import SVC, LinearSVC as LSVC
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn import datasets
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, Adam, lr_scheduler
import torch.nn.functional as F

from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

def seed_everything(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
class SGDClassifier(object):
  def __init__(self, random_state):
    self.clf = CalibratedClassifierCV(SGDC(loss='hinge', random_state=random_state))
  def fit(self, X,y):
    self.clf.fit(X,y)
  def predict(self,X):
    return self.clf.predict(X)
  def predict_proba(self,X):
    return self.clf.predict_proba(X)

class LinearSVC(object):
  def __init__(self, random_state):
    self.clf = CalibratedClassifierCV(LSVC(random_state=random_state))
  def fit(self, X,y):
    self.clf.fit(X,y)
  def predict(self,X):
    return self.clf.predict(X)
  def predict_proba(self,X):
    return self.clf.predict_proba(X)

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
      self.data = data
      self.targets = torch.LongTensor(targets)
      self.transform = transform
    def __getitem__(self, index):
      x = self.data[index]
      x = self.transform(x)
      y = self.targets[index]
      return x, y  
    def __len__(self):
      return len(self.data)

class BertClassifier(nn.Module):
  def __init__(self, n_classes, dropout, path_model): 
    super(BertClassifier,self).__init__() 
    self.bert = BertModel.from_pretrained(path_model) 
    self.drop = nn.Dropout( dropout ) 
    self.linear = nn.Linear(768, n_classes)
  def forward(self,ids,mask) : 
    _,pooled_output = self.bert( 
            ids , 
            attention_mask = mask,
            return_dict=False
            )
    pooled_output = self.drop(pooled_output)
    output = self.linear(pooled_output)
    return output

class Wrapper(object):
  def __init__(self, 
    n_classes,
    model,
    domain,
    transform_method,
    scheduler_opt,
    early_stopping,
    validation_split,
    val_loss_min, 
    patience, 
    batch_size, 
    epochs,
    dropout,
    MAX_SENT_LEN,
    lr, 
    RUNS,
    SEED = SEED):

    if torch.cuda.is_available():    
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")   

    self.model = model
    self.scheduler_opt = scheduler_opt
    self.domain = domain
    self.transform_method = transform_method

    if self.model == 'bert':
      self.path_model = os.path.join( os.getcwd(), self.domain, 'models', 'bert-base-multilingual-uncased')
      cased = 'uncased' in self.path_model
      self.tokenizer = BertTokenizer.from_pretrained(self.path_model, do_lower_case=cased)
    
    self.batch_size = batch_size
    self.epochs = epochs
    self.validation_split = validation_split
    self.early_stopping = early_stopping 
    self.val_loss_min = val_loss_min
    self.patience = patience
    self.dropout = dropout
    self.MAX_SENT_LEN = MAX_SENT_LEN
    self.lr = lr
    self.RUNS = RUNS
    self.n_classes = n_classes
    self.SEED = SEED

  def encode_bert(self, texts, MAX_SENT_LEN):
    input_ids = [self.tokenizer.encode(sent, add_special_tokens = True) for sent in texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_SENT_LEN, dtype="int", 
                              value=0, truncating="post", padding="post")
    attention_masks = [ [float(ids>0) for ids in segment]  for segment in input_ids ]
    return np.array(input_ids), np.array(attention_masks)

  def reset_linear(self, m):
    if type(m) == nn.Linear:            
      m.reset_parameters()

  def fit(self, X, y):
    if self.validation_split>0:
      X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=self.validation_split, shuffle = False, random_state = self.SEED ) 
    else:
      X_train = copy.deepcopy(X)
      y_train = copy.deepcopy(y)

    if self.domain == 'texts':
      if 'bert' in self.model:
        input_ids, attention_masks = self.encode_bert(X_train, self.MAX_SENT_LEN)      
     
      train_inputs = torch.tensor(input_ids)
      train_inputs = train_inputs.long()
      train_masks = torch.tensor(attention_masks)   
      train_masks = train_masks.long()         
      train_labels = torch.tensor(y_train)
      train_labels = train_labels.long()
      train_data = TensorDataset(train_inputs, train_masks, train_labels)
      train_sampler = RandomSampler(train_data)
      train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size, num_workers = 0, shuffle = False)
      del input_ids
      del attention_masks
      del y_train

      if self.validation_split>0:
        input_ids, attention_masks = self.encode_bert(X_val, self.MAX_SENT_LEN)        
        val_inputs = torch.tensor(input_ids)
        val_inputs = val_inputs.long()
        val_masks = torch.tensor(attention_masks)  
        val_masks = val_masks.long()
        val_labels = torch.tensor(y_val)
        val_labels = val_labels.long()      
        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = RandomSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=self.batch_size, num_workers = 0, shuffle = False)
        del input_ids
        del attention_masks
        del y_val

      if 'bert' in self.model:
        self.clf = BertClassifier( self.n_classes, self.dropout, self.path_model)     
   
      optimizer = Adam(self.clf.parameters(), lr=self.lr)     

      if self.validation_split>0:
        val = MyDataset(X_val, y_val, transform=self.transform_img['test'])
        val_dataloader = DataLoader(val, batch_size = self.batch_size, shuffle = False)
        del val
        del y_val
        del X_val

    self.clf.to(self.device)

    torch.cuda.empty_cache()
    gc.collect() 

    if self.scheduler_opt:
      total_steps = len(train_dataloader) * self.epochs
      scheduler = get_linear_schedule_with_warmup(optimizer, 
                                          num_warmup_steps = 0, 
                                          num_training_steps = total_steps)

    epochs_stop = 0 
    self.loss_training = []       
    self.loss_val = []
    fcn = nn.CrossEntropyLoss()

    for epoch_i in range(0, self.epochs):
      # training
      train_loss = 0
      self.clf.train()
      for step, batch in enumerate(train_dataloader):    

        if self.domain == 'texts':      
          input_ids, input_masks, labels = tuple(t.to(self.device) for t in batch)    

        optimizer.zero_grad()

        if 'bert' in self.model:
          logits = self.clf(input_ids, input_masks)
          batch_loss = fcn( logits.view(-1, self.n_classes), labels.view(-1) )
          del input_ids
          del input_masks
          del labels

        gc.collect()
        torch.cuda.empty_cache()
        
        train_loss += batch_loss.item()                
        batch_loss.backward()
        
        if self.scheduler_opt:
          clip_grad_norm_(parameters = self.clf.parameters(), max_norm = 1.0)

        optimizer.step()

        if self.scheduler_opt:
          scheduler.step()

      train_loss /= len(train_dataloader.dataset)
      self.loss_training.append( train_loss )
      
      # validation
      if self.validation_split>0:
        val_loss = 0
        self.clf.eval()
        for step, batch in enumerate(val_dataloader):       
          
          if self.domain == 'texts':          
            input_ids, input_masks, labels = tuple(t.to(self.device) for t in batch) 

          if 'bert' in self.model:
            logits = self.clf(input_ids, input_masks)
            batch_loss = fcn( logits.view(-1, self.n_classes), labels.view(-1) )
            del input_ids
            del input_masks
            del labels
            gc.collect()

          gc.collect()
          torch.cuda.empty_cache()

          val_loss += batch_loss.item()

        val_loss /= len(val_dataloader.dataset)
        self.loss_val.append( val_loss )
        
        # early stopping     
        if self.early_stopping:
          if val_loss<self.val_loss_min:
            self.val_loss_min = val_loss
            epochs_stop = 0
            params_model = copy.deepcopy( self.clf.state_dict() )
          else:
            epochs_stop+=1
          if epochs_stop>=self.patience:
            self.clf.load_state_dict( params_model )
            #print(epoch_i, 'epochs')
            break

    del train_dataloader

    if self.validation_split>0:
      del val_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
  def predict(self, X_test):        

    if self.domain == 'texts':
      if 'bert' in self.model:
        input_ids, attention_masks = self.encode_bert(X_test, self.MAX_SENT_LEN)

      prediction_inputs = torch.tensor(input_ids)
      prediction_inputs = prediction_inputs.long()
      prediction_masks = torch.tensor(attention_masks)
      prediction_masks = prediction_masks.long()
      # fake labels
      y = np.zeros(len(X_test))
      test_labels = torch.tensor(y)
      test_labels = test_labels.long()
      prediction_data = TensorDataset(prediction_inputs, prediction_masks,test_labels)
      prediction_sampler = SequentialSampler(prediction_data)
      test_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size, num_workers = 0, shuffle = False)

      del input_ids
      del attention_masks
      del y
      del X_test

    gc.collect()
    torch.cuda.empty_cache()

    self.clf.eval()
    predictions = []
    with torch.no_grad():
      for step, batch in enumerate(test_dataloader):    

        if self.domain == 'texts':    
          input_ids, input_masks, labels = tuple(t.to(self.device) for t in batch)

        if 'bert' in self.model:
          logits = self.clf(input_ids, input_masks)
          del input_ids
          del input_masks
          del labels

        gc.collect()
        torch.cuda.empty_cache()

        logits = F.softmax(logits, dim=1)
        logits = logits.detach().cpu().numpy()        
        predictions += list( np.argmax(logits,  axis = 1) )
                            
    return np.array(predictions, dtype = int)
        
  def apply_dropout(self, m):
    if type(m) == nn.Dropout:
      m.train()

  def predict_proba(self, X_u):

    if self.domain == 'texts':
      if 'bert' in self.model:
        input_ids, attention_masks = self.encode_bert(X_u, self.MAX_SENT_LEN)

      prediction_inputs = torch.tensor(input_ids)
      prediction_inputs = prediction_inputs.long()
      prediction_masks = torch.tensor(attention_masks)
      prediction_masks = prediction_masks.long()
      # fake labels
      y = np.zeros(len(X_u))
      test_labels = torch.tensor(y)
      test_labels = test_labels.long()
      prediction_data = TensorDataset(prediction_inputs, prediction_masks, test_labels)
      prediction_sampler = SequentialSampler(prediction_data)
      test_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size, num_workers = 0, shuffle = False)

      del input_ids
      del attention_masks
      del y
      del X_u

    self.clf.eval()
    self.clf.apply(self.apply_dropout)
    probs = []
    for times in range(self.RUNS):
        
      logits_sum = np.array([])
      with torch.no_grad():
        for step, batch in enumerate(test_dataloader):        
          
          if self.domain == 'texts':
            input_ids, input_masks, labels = tuple(t.to(self.device) for t in batch)

          if 'bert' in self.model:
            logits = self.clf(input_ids, input_masks)
            del input_ids
            del input_masks
            del labels

          gc.collect()
          torch.cuda.empty_cache()

          logits = F.softmax(logits, dim=1)
          logits = logits.detach().cpu().numpy()    
          if len(logits_sum)==0:
            logits_sum = copy.deepcopy(logits)
          else:
            logits_sum = np.vstack((logits_sum, logits))                    
      probs.append( logits_sum )
    probs = np.mean(probs, axis = 0)      
    return probs

class AL(object):
  def __init__(self, X_u, X_l, clf, n_classes, curve, domain, transform_method):
    self.X_l = copy.deepcopy( X_l )
    if clf.__class__.__name__ not in ['SVC', 'LinearSVC', 'RandomForestClassifier', 'MultinomialNB', 'GaussianNB']: #DL
      self.X_u = copy.deepcopy( X_u )
    else:
      if domain == 'texts':
        if transform_method == 'tfidf':
          tfidf = TfidfVectorizer()
          tfidf.fit(self.X_l)
          self.X_u = copy.deepcopy( tfidf.transform(X_u) ).toarray()
    self.clf = copy.deepcopy(clf)
    self.n_classes = n_classes

  def score_function(self):
    scores = {}
    probs = self.clf.predict_proba( self.X_u )
    scores_aux = []
    for p in probs:
      scores_aux.append( entropy(p, base=2) )
    scores_aux = np.array(scores_aux)
    scores['scores'] = scores_aux
    scores['probs'] = probs   
    return scores

class Curves(object):
  def __init__(self, X_train, y_train, X_test, y_test, batch, curve, model, net_params, clf_params, grid_params, folds, ssl, pge, thr_prb=0.9, pval = 0.5, seed = SEED):
    self.X_train = copy.deepcopy(X_train)
    self.y_train = copy.deepcopy(y_train)
    self.X_test  = copy.deepcopy(X_test)
    self.y_test = copy.deepcopy(y_test)
    self.batch = batch
    self.curve = curve
    self.seed = seed
    self.results = {}
    self.net_params = copy.deepcopy(net_params)
    self.clf_params = copy.deepcopy(clf_params)
    self.grid_params = grid_params 
    self.folds = folds
    self.model = model
    self.net_params['model'] = self.model
    self.n_classes = net_params['n_classes']
    self.domain = net_params['domain']
    self.transform_method = net_params['transform_method']
    self.ssl = ssl
    self.PGE = pge
    self.THR_PRB = thr_prb
    self.pval = pval
    if self.domain == 'texts':
      if self.transform_method == 'tfidf': 
        self.tfidf = TfidfVectorizer()

  def model_selection(self, model, X_l, y_l, X_test, return_clf=False):

    if self.grid_params and len(y_l) == self.batch and model in ['svm', 'rf', 'nb-multinomial', 'nb-gaussian']:
      #print('Update params')
      self.clf_params = grid_search(X_l, y_l, self.folds, model, self.domain, self.transform_method)
      
    if self.domain == 'texts':
      if self.transform_method == 'tfidf':
        X_l_aux = copy.deepcopy( self.tfidf.fit_transform(X_l).toarray() )
        X_test_aux = copy.deepcopy( self.tfidf.transform( X_test ).toarray() )      

    if 'svm' in model:
      seed_everything()
      clf = SVC(**self.clf_params)
      clf.fit(X_l_aux, y_l)
      prob = clf.predict_proba(X_test_aux) 
      del X_l_aux
      del X_test_aux
      gc.collect()
      torch.cuda.empty_cache() 

    elif 'rf' in model:
      seed_everything()
      clf = RFC(**self.clf_params)
      clf.fit(X_l_aux, y_l)
      prob = clf.predict_proba(X_test_aux)
      del X_l_aux
      del X_test_aux
      gc.collect()
      torch.cuda.empty_cache() 

    elif 'nb' in model:
      seed_everything()
      type_ = model.split('-')[-1]
      if type_ == 'multinomial':
        clf = MNB(**self.clf_params)
      clf.fit(X_l_aux, y_l)
      prob = clf.predict_proba(X_test_aux)
      del X_l_aux
      del X_test_aux
      gc.collect()
      torch.cuda.empty_cache() 

    else: #bert
      seed_everything()
      clf = Wrapper(**self.net_params)
      clf.fit(X_l, y_l)
      prob = clf.predict_proba(X_test)
      del X_l_aux
      del X_test_aux
      gc.collect()
      torch.cuda.empty_cache() 

    if return_clf:
      return prob, clf
    else:
      del clf
      gc.collect()
      torch.cuda.empty_cache() 
      return prob

  def start(self):  
      
    x = []
    y = []       
    X_l = np.array([])
    y_l = np.array([])

    if self.pval>0:
      VAL_LENGTH = int( np.ceil(self.batch*self.pval) )
      classes_ = copy.deepcopy(self.y_train[ : VAL_LENGTH ] )
      while len(set(classes_)) != self.n_classes:
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state = self.seed)
        classes_ = copy.deepcopy(self.y_train[ : VAL_LENGTH ] )
      del classes_
      gc.collect()
      torch.cuda.empty_cache() 

      self.X_val = self.X_train[ : VAL_LENGTH ]
      self.y_val = self.y_train[ : VAL_LENGTH ]
      self.X_train = self.X_train[VAL_LENGTH:]
      self.y_train = self.y_train[VAL_LENGTH:]

      classes_ = copy.deepcopy(self.y_train[:self.batch])
      while len(set(classes_)) != self.n_classes:
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state = self.seed)
        classes_ = copy.deepcopy(self.y_train[:self.batch])
      del classes_
      gc.collect()
      torch.cuda.empty_cache() 

    else:
      classes_ = copy.deepcopy(self.y_train[:self.batch])
      while len(set(classes_)) != self.n_classes:
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state = self.seed)
        classes_ = copy.deepcopy(self.y_train[:self.batch])
      del classes_
      gc.collect()
      torch.cuda.empty_cache() 
      self.X_val = np.array([])
      self.y_val = np.array([])
                 
    X_u = copy.deepcopy( self.X_train )
    y_u = copy.deepcopy( self.y_train )  
    X_l = X_u[:self.batch]
    y_l = y_u[:self.batch]
    indexes = list(range(self.batch))                    
    X_u = np.delete(X_u, indexes, axis = 0)
    y_u = np.delete(y_u, indexes, axis = 0)

    return X_l, y_l, X_u, y_u, [], []


  def valida_examples(self, X_l, y_l, X_u, y_clf, indexes):
    X_l_aux = copy.deepcopy( X_l )
    X_l_aux = np.concatenate( ( X_l_aux, X_u[indexes] ) )
    y_l_aux = copy.deepcopy( y_l )
    y_l_aux = np.concatenate( ( y_l_aux, y_clf[indexes] ) ) 
    pred = self.model_selection(self.model, X_l_aux, y_l_aux, self.X_val)
    new_error = 1-accuracy_score( self.y_val, np.argmax(pred, axis=1) )
    del X_l_aux
    del y_l_aux
    gc.collect()
    torch.cuda.empty_cache() 
    return new_error, pred

  def pge_exp(self, x, r=1,y0=20/100):
    y = y0*(1+r)**x
    return y

  def learningCurve(self):
    
    #query strategy
    scores = []
    samples = [] #human effort
    distribution = [] #human vs classifier  
    probs_qs = [] #probabilities (unlabeled samples)
    #validation
    probs_val = [] #probabilities (labeled samples)
        
    prob = (1-self.THR_PRB)/(self.n_classes-1)
    THR_ENTROPY = -self.THR_PRB*np.log2(self.THR_PRB)
    for times in range(self.n_classes-1):
      THR_ENTROPY += -prob*np.log2(prob)

    COUNT = -1
    if self.PGE is not None:
      PBATCH = self.PGE  #--- to +++ (5%)
    else:
      X_PGE = 0
      PBATCH = self.pge_exp(X_PGE)

    batch_incr = False
    count_incr = -1
    if self.batch is None:
      self.PGE = 1 #todo el batch
      PBATCH = 1 #todo el batch
      self.batch = 16
      batch_incr = True
      count_incr = 0

    NCLUSTERS = 5

    X_l, y_l, X_u, y_u, x, y = self.start()    

    if 'SSL' in self.curve:
      samples.append( self.batch )
      distribution.append( ['H']*self.batch )
      y_train_semi_aux = copy.deepcopy( y_l )

    if self.curve != 'PL':
      scores.append( [] )
      probs_qs.append( [] )

    probs_val.append( [] ) #impar

    while len(X_u)>=0:   

      #print(self.batch)

      COUNT +=1
      pred, clf = self.model_selection(self.model, X_l, y_l, self.X_test, True)
      
      x.append( len(y_l) )    
      y.append( np.argmax(pred, axis=1) )
      print( 'acc:', 100*accuracy_score( self.y_test, y[-1] ) )
      if len(X_u)==0:
        break
      indexes = np.array([], dtype = int)
      half = 0

      if self.curve == 'PL':
        indexes = shuffle( np.arange(len(X_u)), random_state=self.seed )
        
        if (COUNT%2)==0:
          if batch_incr:
            if (count_incr%2)==0 and count_incr>=2:
              self.batch*=2 #16,16,16,32,32,64,64,128,128
            count_incr+=1

      else:

        if self.curve == 'AL':
          if (COUNT%2)==0:
            if batch_incr:
              if (count_incr%2)==0 and count_incr>=2:
                self.batch*=2 #16,16,16,32,32,64,64,128,128
              count_incr+=1

        objeto = AL( X_u, X_l, clf, self.n_classes, self.curve, self.domain, self.transform_method )
        scores_fx = copy.deepcopy(objeto.score_function())
        scores_aux = copy.deepcopy( scores_fx['scores'] )
        probs_aux = copy.deepcopy( scores_fx['probs'] )        
        y_clf = np.argmax( probs_aux, axis=1 )

        if self.curve == 'SSLAL' and (COUNT%2)==0:                    

          ####################################################################################

          if self.PGE is None:
            X_PGE+=1

          #count_incr+=1
          if batch_incr:
            if (count_incr%2)==0 and count_incr>=2:
              self.batch*=2 #16,16,16,32,32,64,64,128,128
            count_incr+=1

          indexes_probs_aux = np.where( scores_aux<THR_ENTROPY )[0][:int(self.batch*PBATCH)]
          pred = self.model_selection(self.model, X_l, y_l, self.X_val)
          probs_val_aux = copy.deepcopy(pred)
          error_base = 1-accuracy_score( self.y_val, np.argmax(pred, axis=1) ) 

          del pred
          gc.collect()
          torch.cuda.empty_cache() 

          #Devolver: indexes, probs_val_aux
          if self.ssl == 'all':
            new_error, pred = self.valida_examples(X_l, y_l, X_u, y_clf, indexes_probs_aux)
            if new_error<=error_base:
              indexes = copy.deepcopy(indexes_probs_aux)

            del pred
            gc.collect()
            torch.cuda.empty_cache() 

          if self.ssl == 'add-each-one': #add from -entropy

            idxs_scores_aux_sorted = indexes_probs_aux[np.argsort(scores_aux[indexes_probs_aux])] #entropy --- to +++

            probs_val_aux = [] #limipiar para anhadir

            for index_aux in idxs_scores_aux_sorted:

              new_error, pred = self.valida_examples(X_l, y_l, X_u, y_clf, [index_aux])
              probs_val_aux.append(pred)

              if new_error<=error_base:
                indexes = np.concatenate( ( indexes, [index_aux] ) )             

              del pred
              gc.collect()
              torch.cuda.empty_cache() 

            probs_val_aux = np.array(probs_val_aux)

          elif self.ssl == 'del-each-one': #del from +entropy

              new_error = np.Inf
              idxs_scores_aux_sorted = indexes_probs_aux[np.argsort(scores_aux[indexes_probs_aux])[::-1]] #entropy +++ to ---

              while new_error>error_base and len(indexes_probs_aux)>0:
                                
                new_error, pred = self.valida_examples(X_l, y_l, X_u, y_clf, indexes_probs_aux)
                probs_val_aux = copy.deepcopy( pred ) 
                indexes = copy.deepcopy(indexes_probs_aux) #desordenado

                del pred
                gc.collect()
                torch.cuda.empty_cache() 

                val_idx_probs = idxs_scores_aux_sorted[0] #idx mayor entropia
                idx_probs = np.where( idxs_scores_aux_sorted==val_idx_probs )  
                idxs_scores_aux_sorted = np.delete( idxs_scores_aux_sorted, idx_probs, axis=0 ) #ok
                idx_probs = np.where( indexes_probs_aux==val_idx_probs )  
                indexes_probs_aux = np.delete( indexes_probs_aux, idx_probs , axis=0 )

          elif self.ssl == 'del-clustering':

            if len(indexes_probs_aux)>=NCLUSTERS:          
              if self.domain=='texts':
                X_prob = copy.deepcopy( self.tfidf.fit_transform( X_u[indexes_probs_aux] ).toarray() )
              cluster = AgglomerativeClustering( n_clusters=NCLUSTERS, affinity='cosine', linkage='average' )
              cluster.fit( X_prob )
              clusters = copy.deepcopy( cluster.labels_ )

              del X_prob
              del cluster
              gc.collect()
              torch.cuda.empty_cache() 

              CLUSTER = 0
              new_error = np.Inf
              while new_error>error_base and len(indexes_probs_aux)>0:
                
                new_error, pred = self.valida_examples(X_l, y_l, X_u, y_clf, indexes_probs_aux)
                probs_val_aux = copy.deepcopy( pred ) 
                indexes = copy.deepcopy(indexes_probs_aux)

                del pred
                gc.collect()
                torch.cuda.empty_cache() 

                idx_clusters = np.where( clusters==CLUSTER )[0]
                indexes_probs_aux = np.delete( indexes_probs_aux, idx_clusters, axis=0 )
                clusters = np.delete( clusters, idx_clusters, axis=0 )
                CLUSTER += 1

            else:
              indexes = copy.deepcopy(indexes_probs_aux)

          probs_val.append(probs_val_aux)
          del indexes_probs_aux
          del probs_val_aux
          gc.collect()
          torch.cuda.empty_cache() 

          if self.PGE is not None:
            if PBATCH<1: #--- to +++ (5%)
              PBATCH += self.PGE
          else:
            PBATCH = self.pge_exp(X_PGE)
          if PBATCH>1:
            PBATCH = 1

        if self.curve != 'PL' and len(indexes)<=self.batch: #AL/SSL(par, imcompleto)/SSL(impar)
          half = len(indexes)
          indexes_entropy = np.argsort( scores_aux )[::-1] # entropy +++ to ---
          for idx in indexes_entropy:
            if idx not in indexes:
              indexes = np.concatenate( (indexes, np.array([idx]) ) )
          if 'SSL' in self.curve:
            samples.append( x[-1]+len(indexes[half:self.batch]) ) 
            distribution.append( ['C']*half+['H']*len(indexes[half:self.batch]) )
            if (COUNT%2) != 0: #impar
              probs_val.append( [] ) 

      if self.curve in ['PL', 'AL']:  
        X_l = np.concatenate((X_l, X_u[indexes[:self.batch]] ))
        y_l = np.concatenate((y_l, y_u[indexes[:self.batch]] ))
        if self.curve == 'AL':
          scores.append( scores_aux[indexes] )  #all but sorted scores according indexes

      else: #SSLAL
        scores.append( scores_aux[indexes] )  #all but sorted scores according indexes
        
        X_l = np.concatenate((X_l, X_u[indexes[:self.batch]] ))
        if self.curve == 'SSL':
          y_l = np.concatenate((y_l, y_clf[indexes[:self.batch]] ))#self-training
          probs_qs.append( probs_aux[indexes] ) #all but sorted according indexes
        elif self.curve == 'SSLAL':
          y_l_aux = copy.deepcopy( y_clf[indexes[:half]] )
          y_l_aux = np.concatenate((y_l_aux, y_u[indexes[half:self.batch]] ))
          y_l = np.concatenate((y_l, y_l_aux ))

          p_aux = copy.deepcopy( probs_aux[indexes[:half]] )
          p_aux = np.concatenate((p_aux, probs_aux[indexes[half:]] )) #all but sorted according indexes
          probs_qs.append( p_aux )

          del y_l_aux
          del p_aux
          gc.collect()
          torch.cuda.empty_cache()           

          y_train_semi_aux = np.concatenate((y_train_semi_aux, y_clf[indexes[:half]] ))
          if half>0:
            y_train_semi_aux[-half:] = -1
          y_train_semi_aux = np.concatenate((y_train_semi_aux, y_u[indexes[half:self.batch]] ))

      X_u = np.delete(X_u, indexes[:self.batch], axis = 0)
      y_u = np.delete(y_u, indexes[:self.batch], axis = 0)
      
      if len(X_u)<=0 and 'SSL' not in self.curve:
        X_l = copy.deepcopy( self.X_train )
        y_l = copy.deepcopy( self.y_train )

        del self.X_train
        del self.y_train
        gc.collect()
        torch.cuda.empty_cache()      

      if 'SSL' in self.curve:  
        indexes_semi_aux = np.where( y_train_semi_aux<0 )[0] #pseudo-labels
        indexes_l_aux = np.where( y_train_semi_aux>=0 )[0] #labels
        if len(indexes_semi_aux)>0:
          pred = self.model_selection(self.model, X_l[indexes_l_aux], y_l[indexes_l_aux], X_l[indexes_semi_aux])
          y_l[indexes_semi_aux] = np.argmax(pred, axis=1)
          del pred
        del indexes_semi_aux
        del indexes_l_aux
        gc.collect()
        torch.cuda.empty_cache() 
      
      del clf
      gc.collect()
      torch.cuda.empty_cache()
            
    del X_l
    del y_l
    del X_u
    del y_u
    del self.X_test
    del self.y_test
    del self.X_val
    del self.y_val
    gc.collect()
    torch.cuda.empty_cache()

    x = np.array(x)
    y = np.array(y)
    self.results['x'] = x
    self.results['y'] = y
    self.results['scores'] = np.array(scores)
    self.results['samples'] = np.array(samples)
    self.results['distribution'] = np.array(distribution)
    self.results['probs_qs'] = np.array(probs_qs)
    self.results['probs_val'] = np.array(probs_val)

def grid_search(DATA, CLASSES, FOLDS, CLF, DOMAIN, METHOD, SEED = SEED):
  if CLF in ['svm', 'rf', 'nb-multinomial', 'nb-gaussian']:
    best_params = {}
    if DOMAIN == 'texts':
      tfidf = TfidfVectorizer()
      DATA_X = tfidf.fit_transform(DATA).toarray()
    if CLF == 'svm':
      seed_everything()
      param_grid = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100, 1000]}
      clf = SVC( probability=True, random_state=SEED )
      best_params['probability'] = True
      best_params['random_state'] = SEED
    elif CLF == 'rf':
      seed_everything()
      param_grid = {'criterion':('entropy', 'gini'), 'n_estimators':[10, 100, 500, 1000]}
      clf = RFC( random_state=SEED)
      best_params['random_state'] = SEED
    elif CLF == 'nb-multinomial':
      seed_everything()
      param_grid = {'alpha': [0, 0.25, 0.75, 1]}
      clf = MNB()

    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=FOLDS, scoring='accuracy')
    grid_search.fit(DATA_X, CLASSES)
    del DATA_X
    gc.collect()
    torch.cuda.empty_cache()   
        
    best_params.update(grid_search.best_params_)
  
    return best_params

  else:

    return None