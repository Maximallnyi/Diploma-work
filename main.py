from FCN import Classifier_FCN
from RESNET import Classifier_RESNET
from MLP import Classifier_MLP
from GTN import Global_Dataset
import GTN
import GTN_BOSS
import BOSS

import pandas as pd
import numpy as np
import wfdb
import ast
#import neurokit2 as nk
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.io import loadmat
import math
import torch

import os
import operator

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import normalize

from scipy.interpolate import interp1d
from scipy.io import loadmat

from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
import sklearn

def calculate_metrics(y_true, y_pred, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 3), dtype=np.float64), index=[0],
                       columns=['precision', 'accuracy', 'recall'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    return res


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
    else:
        data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
    data = np.array([signal for signal, meta in data])
    return data

def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data


# ECG2000
ecg200_train = pd.read_csv('/kaggle/input/ecg200/ECG200_TRAIN.txt', header=None, sep='  ')
ecg200_test = pd.read_csv('/kaggle/input/ecg200/ECG200_TEST.txt', header=None, sep='  ')

y_train = ecg200_train[0].to_numpy()
y_test = ecg200_test[0].to_numpy()
X_train = ecg200_train.iloc[:,1:].to_numpy()
X_test = ecg200_test.iloc[:,1:].to_numpy()

# GTN_BOSS
EPOCH = 40
BATCH_SIZE = 20
LR = 1e-4
d_model = 512
d_hidden = 2048
q = 8
v = 8
h = 4
N = 4


train_dataset = Global_Dataset(X=torch.from_numpy(X_train).to(torch.float32).unsqueeze(1), Y=torch.from_numpy(y_train).to(torch.int64))
test_dataset = Global_Dataset(X=torch.from_numpy(X_test).to(torch.float32).unsqueeze(1), Y=torch.from_numpy(y_test).to(torch.int64))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'use device:{DEVICE}')
boss = BOSS(word_size=4, n_bins=3, window_size=12, sparse=False)
boss.fit(train_dataset)
net = GTN_BOSS.Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=1, d_timestep=96, q=q, v=v, h=h, N=N, class_num=2, boss = boss)
net = net.to(DEVICE)

# 3. Create opitmizer and loss_function
optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

# 4. training

for epoch_index in range(EPOCH):
    loss_sum = 0.0
    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pre = net(x, 'train')
        loss = loss_function(pre, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print(f'EPOCH:{epoch_index + 1}\tloss:{loss_sum}')

correct = 0
total = 0

y_pred = []
with torch.no_grad():
    net.eval()
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre = net(x, 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        y_pred.extend(map(int,label_index))
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()
    print(f'Accuracy: %.2f %%' % (100 * correct / total))

y_pred = np.array(y_pred)
copy_y_test = y_test.copy()
copy_y_test[copy_y_test==-1] = 0

print(calculate_metrics(copy_y_test, y_pred))

# GTN
net = GTN.Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=1, d_timestep=96, q=q, v=v, h=h, N=N, class_num=2, boss = boss)
net = net.to(DEVICE)

optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

for epoch_index in range(EPOCH):
    loss_sum = 0.0
    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pre = net(x, 'train')
        loss = loss_function(pre, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print(f'EPOCH:{epoch_index + 1}\tloss:{loss_sum}')

correct = 0
total = 0

y_pred = []
with torch.no_grad():
    net.eval()
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre = net(x, 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        y_pred.extend(map(int,label_index))
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()
    print(f'Accuracy: %.2f %%' % (100 * correct / total))

y_pred = np.array(y_pred)
copy_y_test = y_test.copy()
copy_y_test[copy_y_test==-1] = 0

print(calculate_metrics(copy_y_test, y_pred))

#RESNET

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# save orignal y because later we will use binary
y_true = np.argmax(y_test, axis=1)

if len(X_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
input_shape = X_train.shape[1:]

nb_epochs = 150

model = Classifier_RESNET(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

# MLP
nb_epochs = 300

model = Classifier_MLP(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

#FCN

nb_epochs = 300

model = Classifier_FCN(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))




# ECG5000
ecg5000_train = pd.read_csv('/kaggle/input/ecg5000/ECG5000_TRAIN.txt', header=None, sep='  ')
ecg5000_test = pd.read_csv('/kaggle/input/ecg5000/ECG5000_TEST.txt', header=None, sep='  ')

y_train = ecg5000_train[0].to_numpy()
y_test = ecg5000_test[0].to_numpy()
X_train = ecg5000_train.iloc[:,1:].to_numpy()
X_test = ecg5000_test.iloc[:,1:].to_numpy()

# GTN_BOSS
EPOCH = 40
BATCH_SIZE = 20
LR = 1e-4
d_model = 512
d_hidden = 2048
q = 8
v = 8
h = 4
N = 4


train_dataset = Global_Dataset(X=torch.from_numpy(X_train).to(torch.float32).unsqueeze(1), Y=torch.from_numpy(y_train).to(torch.int64))
test_dataset = Global_Dataset(X=torch.from_numpy(X_test).to(torch.float32).unsqueeze(1), Y=torch.from_numpy(y_test).to(torch.int64))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'use device:{DEVICE}')
boss = BOSS(word_size=4, n_bins=3, window_size=12, sparse=False)
boss.fit(train_dataset)
net = GTN_BOSS.Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=1, d_timestep=96, q=q, v=v, h=h, N=N, class_num=2, boss = boss)
net = net.to(DEVICE)

# 3. Create opitmizer and loss_function
optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

# 4. training

for epoch_index in range(EPOCH):
    loss_sum = 0.0
    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pre = net(x, 'train')
        loss = loss_function(pre, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print(f'EPOCH:{epoch_index + 1}\tloss:{loss_sum}')

correct = 0
total = 0

y_pred = []
with torch.no_grad():
    net.eval()
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre = net(x, 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        y_pred.extend(map(int,label_index))
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()
    print(f'Accuracy: %.2f %%' % (100 * correct / total))

y_pred = np.array(y_pred)
copy_y_test = y_test.copy()
copy_y_test[copy_y_test==-1] = 0

print(calculate_metrics(copy_y_test, y_pred))

# GTN
net = GTN.Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=1, d_timestep=96, q=q, v=v, h=h, N=N, class_num=2, boss = boss)
net = net.to(DEVICE)

optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

for epoch_index in range(EPOCH):
    loss_sum = 0.0
    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pre = net(x, 'train')
        loss = loss_function(pre, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print(f'EPOCH:{epoch_index + 1}\tloss:{loss_sum}')

correct = 0
total = 0

y_pred = []
with torch.no_grad():
    net.eval()
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre = net(x, 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        y_pred.extend(map(int,label_index))
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()
    print(f'Accuracy: %.2f %%' % (100 * correct / total))

y_pred = np.array(y_pred)
copy_y_test = y_test.copy()
copy_y_test[copy_y_test==-1] = 0

print(calculate_metrics(copy_y_test, y_pred))

#RESNET

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# save orignal y because later we will use binary
y_true = np.argmax(y_test, axis=1)

if len(X_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
input_shape = X_train.shape[1:]

nb_epochs = 150

model = Classifier_RESNET(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

# MLP
nb_epochs = 300

model = Classifier_MLP(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

#FCN

nb_epochs = 300

model = Classifier_FCN(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))


# Arrhytmia Dataset

arrhythmia_train = pd.read_csv('/kaggle/input/arrhythmia-dataset/mitbih_train.csv', header = None)
arrhythmia_test = pd.read_csv('/kaggle/input/arrhythmia-dataset/mitbih_test.csv', header = None)

y_train = arrhythmia_train[arrhythmia_train.shape[1]-1].to_numpy()
y_test = arrhythmia_test[arrhythmia_train.shape[1]-1].to_numpy()
X_train = arrhythmia_train.iloc[:,:-1].to_numpy()
X_test = arrhythmia_test.iloc[:,:-1].to_numpy()

# GTN_BOSS
EPOCH = 40
BATCH_SIZE = 20
LR = 1e-4
d_model = 512
d_hidden = 2048
q = 8
v = 8
h = 4
N = 4


train_dataset = Global_Dataset(X=torch.from_numpy(X_train).to(torch.float32).unsqueeze(1), Y=torch.from_numpy(y_train).to(torch.int64))
test_dataset = Global_Dataset(X=torch.from_numpy(X_test).to(torch.float32).unsqueeze(1), Y=torch.from_numpy(y_test).to(torch.int64))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'use device:{DEVICE}')
boss = BOSS(word_size=4, n_bins=3, window_size=12, sparse=False)
boss.fit(train_dataset)
net = GTN_BOSS.Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=1, d_timestep=96, q=q, v=v, h=h, N=N, class_num=2, boss = boss)
net = net.to(DEVICE)

# 3. Create opitmizer and loss_function
optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

# 4. training

for epoch_index in range(EPOCH):
    loss_sum = 0.0
    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pre = net(x, 'train')
        loss = loss_function(pre, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print(f'EPOCH:{epoch_index + 1}\tloss:{loss_sum}')

correct = 0
total = 0

y_pred = []
with torch.no_grad():
    net.eval()
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre = net(x, 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        y_pred.extend(map(int,label_index))
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()
    print(f'Accuracy: %.2f %%' % (100 * correct / total))

y_pred = np.array(y_pred)
copy_y_test = y_test.copy()
copy_y_test[copy_y_test==-1] = 0

print(calculate_metrics(copy_y_test, y_pred))

# GTN
net = GTN.Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=1, d_timestep=96, q=q, v=v, h=h, N=N, class_num=2, boss = boss)
net = net.to(DEVICE)

optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

for epoch_index in range(EPOCH):
    loss_sum = 0.0
    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pre = net(x, 'train')
        loss = loss_function(pre, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print(f'EPOCH:{epoch_index + 1}\tloss:{loss_sum}')

correct = 0
total = 0

y_pred = []
with torch.no_grad():
    net.eval()
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre = net(x, 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        y_pred.extend(map(int,label_index))
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()
    print(f'Accuracy: %.2f %%' % (100 * correct / total))

y_pred = np.array(y_pred)
copy_y_test = y_test.copy()
copy_y_test[copy_y_test==-1] = 0

print(calculate_metrics(copy_y_test, y_pred))

#RESNET

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# save orignal y because later we will use binary
y_true = np.argmax(y_test, axis=1)

if len(X_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
input_shape = X_train.shape[1:]

nb_epochs = 150

model = Classifier_RESNET(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

# MLP
nb_epochs = 300

model = Classifier_MLP(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

#FCN

nb_epochs = 300

model = Classifier_FCN(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

# PTB-XL

path = '../input/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sampling_rate = 100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# # Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass


# GTN_BOSS
EPOCH = 40
BATCH_SIZE = 20
LR = 1e-4
d_model = 512
d_hidden = 2048
q = 8
v = 8
h = 4
N = 4


train_dataset = Global_Dataset(X=torch.from_numpy(X_train).to(torch.float32).unsqueeze(1), Y=torch.from_numpy(y_train).to(torch.int64))
test_dataset = Global_Dataset(X=torch.from_numpy(X_test).to(torch.float32).unsqueeze(1), Y=torch.from_numpy(y_test).to(torch.int64))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'use device:{DEVICE}')
boss = BOSS(word_size=4, n_bins=3, window_size=12, sparse=False)
boss.fit(train_dataset)
net = GTN_BOSS.Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=1, d_timestep=96, q=q, v=v, h=h, N=N, class_num=2, boss = boss)
net = net.to(DEVICE)

# 3. Create opitmizer and loss_function
optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

# 4. training

for epoch_index in range(EPOCH):
    loss_sum = 0.0
    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pre = net(x, 'train')
        loss = loss_function(pre, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print(f'EPOCH:{epoch_index + 1}\tloss:{loss_sum}')

correct = 0
total = 0

y_pred = []
with torch.no_grad():
    net.eval()
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre = net(x, 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        y_pred.extend(map(int,label_index))
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()
    print(f'Accuracy: %.2f %%' % (100 * correct / total))

y_pred = np.array(y_pred)
copy_y_test = y_test.copy()
copy_y_test[copy_y_test==-1] = 0

print(calculate_metrics(copy_y_test, y_pred))

# GTN
net = GTN.Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=1, d_timestep=96, q=q, v=v, h=h, N=N, class_num=2, boss = boss)
net = net.to(DEVICE)

optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

for epoch_index in range(EPOCH):
    loss_sum = 0.0
    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pre = net(x, 'train')
        loss = loss_function(pre, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print(f'EPOCH:{epoch_index + 1}\tloss:{loss_sum}')

correct = 0
total = 0

y_pred = []
with torch.no_grad():
    net.eval()
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre = net(x, 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        y_pred.extend(map(int,label_index))
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()
    print(f'Accuracy: %.2f %%' % (100 * correct / total))

y_pred = np.array(y_pred)
copy_y_test = y_test.copy()
copy_y_test[copy_y_test==-1] = 0

print(calculate_metrics(copy_y_test, y_pred))

#RESNET

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# save orignal y because later we will use binary
y_true = np.argmax(y_test, axis=1)

if len(X_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
input_shape = X_train.shape[1:]

nb_epochs = 150

model = Classifier_RESNET(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

# MLP
nb_epochs = 300

model = Classifier_MLP(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

#FCN

nb_epochs = 300

model = Classifier_FCN(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

#China Dataset

ecg_data=[]
ecg_label=[]

for subdir, dirs, files in sorted(os.walk('/kaggle/input/china-12lead-ecg-challenge-database/Training_2')):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".mat"):
                data, header_data = load_challenge_data(filepath)
                ecg_data.append(data)
                diag = header_data[-4].split(' ')[1][:-1].split(',')
                ecg_label.append([int(item) for item in diag])

a = reversed(sorted(ecg_label,key=ecg_label.count))
outlist = []
for element in a:
    if element not in outlist:
        outlist.append(element)
labels_for_model = outlist[:30]

ecg_data= [] 
ecg_label=[]
for subdir, dirs, files in sorted(os.walk('/kaggle/input/china-12lead-ecg-challenge-database/Training_2')):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".mat"):
                data, header_data = load_challenge_data(filepath)
                data = pad_sequences(data, maxlen=1000, truncating='post',padding="post")
                diag = header_data[-4].split(' ')[1][:-1].split(',')
                diag = [int(item) for item in diag]
                curr_data = []
                if diag in labels_for_model:
                    for i in range(len(data)):
                        curr_data.append(normalize(data[i][:,np.newaxis], axis=0).ravel())
                    ecg_data.append(np.array(curr_data))
                    ecg_label.append(diag[0])


ecg_label = [str(i) for i in ecg_label]
ecg_label = np.array(ecg_label)
labelencoder = LabelEncoder()
ecg_label_encode = labelencoder.fit_transform(ecg_label)
ecg_data = np.asarray(ecg_data)

X_train, X_test, y_train, y_test = train_test_split(ecg_data, ecg_label_encode, test_size=0.1, random_state=42)

# GTN_BOSS
EPOCH = 40
BATCH_SIZE = 20
LR = 1e-4
d_model = 512
d_hidden = 2048
q = 8
v = 8
h = 4
N = 4


train_dataset = Global_Dataset(X=torch.from_numpy(X_train).to(torch.float32).unsqueeze(1), Y=torch.from_numpy(y_train).to(torch.int64))
test_dataset = Global_Dataset(X=torch.from_numpy(X_test).to(torch.float32).unsqueeze(1), Y=torch.from_numpy(y_test).to(torch.int64))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'use device:{DEVICE}')
boss = BOSS(word_size=4, n_bins=3, window_size=12, sparse=False)
boss.fit(train_dataset)
net = GTN_BOSS.Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=1, d_timestep=96, q=q, v=v, h=h, N=N, class_num=2, boss = boss)
net = net.to(DEVICE)

# 3. Create opitmizer and loss_function
optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

# 4. training

for epoch_index in range(EPOCH):
    loss_sum = 0.0
    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pre = net(x, 'train')
        loss = loss_function(pre, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print(f'EPOCH:{epoch_index + 1}\tloss:{loss_sum}')

correct = 0
total = 0

y_pred = []
with torch.no_grad():
    net.eval()
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre = net(x, 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        y_pred.extend(map(int,label_index))
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()
    print(f'Accuracy: %.2f %%' % (100 * correct / total))

y_pred = np.array(y_pred)
copy_y_test = y_test.copy()
copy_y_test[copy_y_test==-1] = 0

print(calculate_metrics(copy_y_test, y_pred))

# GTN
net = GTN.Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=1, d_timestep=96, q=q, v=v, h=h, N=N, class_num=2, boss = boss)
net = net.to(DEVICE)

optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

for epoch_index in range(EPOCH):
    loss_sum = 0.0
    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pre = net(x, 'train')
        loss = loss_function(pre, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print(f'EPOCH:{epoch_index + 1}\tloss:{loss_sum}')

correct = 0
total = 0

y_pred = []
with torch.no_grad():
    net.eval()
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre = net(x, 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        y_pred.extend(map(int,label_index))
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()
    print(f'Accuracy: %.2f %%' % (100 * correct / total))

y_pred = np.array(y_pred)
copy_y_test = y_test.copy()
copy_y_test[copy_y_test==-1] = 0

print(calculate_metrics(copy_y_test, y_pred))

#RESNET

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# save orignal y because later we will use binary
y_true = np.argmax(y_test, axis=1)

if len(X_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
input_shape = X_train.shape[1:]

nb_epochs = 150

model = Classifier_RESNET(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

# MLP
nb_epochs = 300

model = Classifier_MLP(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

#FCN

nb_epochs = 300

model = Classifier_FCN(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

#Georgia Dataset

ecg_data=[]
ecg_label=[]

for subdir, dirs, files in sorted(os.walk('/kaggle/input/georgia-12lead-ecg-challenge-database/WFDB')):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".mat"):
                data, header_data = load_challenge_data(filepath)
                ecg_data.append(data)
                diag = header_data[-4].split(' ')[1][:-1].split(',')
                ecg_label.append([int(item) for item in diag])

a = reversed(sorted(ecg_label,key=ecg_label.count))
outlist = []
for element in a:
    if element not in outlist:
        outlist.append(element)
labels_for_model = outlist[:30]

ecg_data= [] 
ecg_label=[]
for subdir, dirs, files in sorted(os.walk('/kaggle/input/china-12lead-ecg-challenge-database/Training_2')):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".mat"):
                data, header_data = load_challenge_data(filepath)
                data = pad_sequences(data, maxlen=1000, truncating='post',padding="post")
                diag = header_data[-4].split(' ')[1][:-1].split(',')
                diag = [int(item) for item in diag]
                curr_data = []
                if diag in labels_for_model:
                    for i in range(len(data)):
                        curr_data.append(normalize(data[i][:,np.newaxis], axis=0).ravel())
                    ecg_data.append(np.array(curr_data))
                    ecg_label.append(diag[0])


ecg_label = [str(i) for i in ecg_label]
ecg_label = np.array(ecg_label)
labelencoder = LabelEncoder()
ecg_label_encode = labelencoder.fit_transform(ecg_label)
ecg_data = np.asarray(ecg_data)

X_train, X_test, y_train, y_test = train_test_split(ecg_data, ecg_label_encode, test_size=0.1, random_state=42)

# GTN_BOSS
EPOCH = 40
BATCH_SIZE = 20
LR = 1e-4
d_model = 512
d_hidden = 2048
q = 8
v = 8
h = 4
N = 4


train_dataset = Global_Dataset(X=torch.from_numpy(X_train).to(torch.float32).unsqueeze(1), Y=torch.from_numpy(y_train).to(torch.int64))
test_dataset = Global_Dataset(X=torch.from_numpy(X_test).to(torch.float32).unsqueeze(1), Y=torch.from_numpy(y_test).to(torch.int64))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'use device:{DEVICE}')
boss = BOSS(word_size=4, n_bins=3, window_size=12, sparse=False)
boss.fit(train_dataset)
net = GTN_BOSS.Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=1, d_timestep=96, q=q, v=v, h=h, N=N, class_num=2, boss = boss)
net = net.to(DEVICE)

# 3. Create opitmizer and loss_function
optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

# 4. training

for epoch_index in range(EPOCH):
    loss_sum = 0.0
    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pre = net(x, 'train')
        loss = loss_function(pre, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print(f'EPOCH:{epoch_index + 1}\tloss:{loss_sum}')

correct = 0
total = 0

y_pred = []
with torch.no_grad():
    net.eval()
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre = net(x, 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        y_pred.extend(map(int,label_index))
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()
    print(f'Accuracy: %.2f %%' % (100 * correct / total))

y_pred = np.array(y_pred)
copy_y_test = y_test.copy()
copy_y_test[copy_y_test==-1] = 0

print(calculate_metrics(copy_y_test, y_pred))

# GTN
net = GTN.Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=1, d_timestep=96, q=q, v=v, h=h, N=N, class_num=2, boss = boss)
net = net.to(DEVICE)

optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

for epoch_index in range(EPOCH):
    loss_sum = 0.0
    for x, y in train_dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        pre = net(x, 'train')
        loss = loss_function(pre, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print(f'EPOCH:{epoch_index + 1}\tloss:{loss_sum}')

correct = 0
total = 0

y_pred = []
with torch.no_grad():
    net.eval()
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre = net(x, 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        y_pred.extend(map(int,label_index))
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()
    print(f'Accuracy: %.2f %%' % (100 * correct / total))

y_pred = np.array(y_pred)
copy_y_test = y_test.copy()
copy_y_test[copy_y_test==-1] = 0

print(calculate_metrics(copy_y_test, y_pred))

#RESNET

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# save orignal y because later we will use binary
y_true = np.argmax(y_test, axis=1)

if len(X_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
input_shape = X_train.shape[1:]

nb_epochs = 150

model = Classifier_RESNET(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

# MLP
nb_epochs = 300

model = Classifier_MLP(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))

#FCN

nb_epochs = 300

model = Classifier_FCN(input_shape, nb_classes)
print(model.fit(X_train, y_train, X_test, y_test, y_true, nb_epochs))



