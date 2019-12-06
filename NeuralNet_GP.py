# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('max_colwidth', 999)
pd.set_option('max_columns', 1000)
pd.set_option('max_rows',100000)
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#%%
raw_data = load_digits()
print(raw_data.keys())


#%%
labels = raw_data['target']
target = pd.Series(labels)
data = pd.DataFrame(raw_data['data'])
data.head()

#%%
fig = plt.figure(figsize=(10,5))
li = [0, 99, 199, 299,999, 1099, 1199, 1299]
for i, r in enumerate(li):
    image_data = data.iloc[r].to_numpy().reshape(8,8)
    ax = fig.add_subplot(2,4,i+1)
    ax.imshow(image_data, cmap='gray_r')
plt.show()

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,mean_squared_error

def train(features,target,n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    fit = knn.fit(features,target)
    return fit
    
def test(fit,features,target):
    prediction = fit.predict(features)
    train_test_df = pd.DataFrame()
    train_test_df['correct_label'] = target
    train_test_df['prediction'] = prediction
    accuracy = sum(train_test_df['prediction']==train_test_df['correct_label'])/len(train_test_df)
    return accuracy

def cross_validate(features,target,folds):
    kf = KFold(n_splits=folds,shuffle=True)
    for train_index, test_index in kf.split(features):
        train_predictors = features.loc[train_index]
        train_target = target.loc[train_index]
        test_predictors = features.loc[test_index]
        test_target = target.loc[test_index]
        score_summary = dict()
        for i in range(1,11): # for n_neighbors parameters
            fit = train(train_predictors,train_target,i)
            accuracy_score = test(fit,test_predictors,test_target)
            score_summary[i] = accuracy_score
        return score_summary

model_summary = dict()
for f in range (4,11): # for folds parameters
    scores = cross_validate(data,target,f)
    model_summary[f] = scores
model_summary

#%%
fold_scores = []
for i in range(4,11):
    v = model_summary[i].values()
    fold_scores.append(v)
fig = plt.figure(figsize=(8,5))
for i in fold_scores:
    plt.plot(range(1,11),list(i))
    plt.grid(axis='y')
    plt.legend(range(4,11))
    plt.title('KNN Errors by folds')
plt.show()

k_scores = []
for k in range(1,11):
    li = []
    for i in range(4,11):
        score = model_summary[i][k]
        li.append(score)
    k_scores.append(li)
fig = plt.figure(figsize=(8,5))
for i in k_scores:
    plt.plot(range(4,11),list(i))
    plt.grid(axis='y')
    plt.legend(range(1,11))
    plt.title('KNN Errors by K')
    
    


#%%
##Using default activation = 'relu'

from sklearn.neural_network import MLPClassifier #relu activation

def train(features,target,neurons,**kvargs):
    mlp = MLPClassifier(hidden_layer_sizes=neurons,max_iter=1000,activation=kvargs['activation'])
    fit = mlp.fit(features,target)
    return fit
    
def test(fit,features,target):
    prediction = fit.predict(features)
    train_test_df = pd.DataFrame()
    train_test_df['correct_label'] = target
    train_test_df['prediction'] = prediction
    accuracy = sum(train_test_df['prediction']==train_test_df['correct_label'])/len(train_test_df)
    return accuracy

def cross_validate(features,target,folds):
    kf = KFold(n_splits=folds,shuffle=True)
    for train_index, test_index in kf.split(features):
        train_predictors = features.loc[train_index]
        train_target = target.loc[train_index]
        test_predictors = features.loc[test_index]
        test_target = target.loc[test_index]
        score_summary = dict()
        neurons = [(8,), (16,), (32,), (64,), (128,), (256,)]
        for i in neurons: # for n_neighbors parameters
            fit = train(train_predictors,train_target,i,activation = 'relu')
            train_error = test(fit,train_predictors,train_target)
            test_error = test(fit,test_predictors,test_target)
            score_summary[f'train_error_{i}'] = train_error
            score_summary[f'test_error_{i}'] = test_error
        return score_summary

scores = cross_validate(data,target,4)
scores

#%%
li=list(scores.items())
train_error = [x[1] for x in li if x[0].startswith('train')]
test_error = [x[1] for x in li if x[0].startswith('test')]
fig = plt.figure(figsize=(6,5))
plt.plot([8, 16, 32, 64, 128, 256],train_error,label = 'train_error')
plt.plot([8, 16, 32, 64, 128, 256],test_error,label = 'test_error')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.title('Relu Errors by Neurons')
print(f'Train Error: {train_error}')
print(f'Test Error: {test_error}')

plt.show()


#%%
##Using activation = 'logistic'

def cross_validate(features,target,folds):
    kf = KFold(n_splits=folds,shuffle=True)
    for train_index, test_index in kf.split(features):
        train_predictors = features.loc[train_index]
        train_target = target.loc[train_index]
        test_predictors = features.loc[test_index]
        test_target = target.loc[test_index]
        score_summary = dict()    
        neurons = [(8,), (16,), (32,), (64,), (128,), (256,)]
        for i in neurons: # for n_neighbors parameters
            fit = train(train_predictors,train_target,i,activation = 'logistic')
            train_error = test(fit,train_predictors,train_target)
            test_error = test(fit,test_predictors,test_target)
            score_summary[f'train_error_{i}'] = train_error
            score_summary[f'test_error_{i}'] = test_error
        return score_summary

scores = cross_validate(data,target,4)


#%%
li=list(scores.items())
train_error = [x[1] for x in li if x[0].startswith('train')]
test_error = [x[1] for x in li if x[0].startswith('test')]
fig = plt.figure(figsize=(6,5))
plt.plot([8, 16, 32, 64, 128, 256],train_error,label = 'train_error')
plt.plot([8, 16, 32, 64, 128, 256],test_error,label = 'test_error')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.title('Logistic Errors by Neurons')
print(f'Train Error: {train_error}')
print(f'Test Error: {test_error}')

plt.show()


#%%
##Using activation = 'tanh'

def cross_validate(features,target,folds):
    kf = KFold(n_splits=folds,shuffle=True)
    for train_index, test_index in kf.split(features):
        train_predictors = features.loc[train_index]
        train_target = target.loc[train_index]
        test_predictors = features.loc[test_index]
        test_target = target.loc[test_index]
        score_summary = dict()
        neurons = [(8,), (16,), (32,), (64,), (128,), (256,)]
        for i in neurons: # for n_neighbors parameters
            fit = train(train_predictors,train_target,i,activation = 'tanh')
            train_error = test(fit,train_predictors,train_target)
            test_error = test(fit,test_predictors,test_target)
            score_summary[f'train_error_{i}'] = train_error
            score_summary[f'test_error_{i}'] = test_error
        return score_summary

scores = cross_validate(data,target,4)


#%%
li=list(scores.items())
train_error = [x[1] for x in li if x[0].startswith('train')]
test_error = [x[1] for x in li if x[0].startswith('test')]
fig = plt.figure(figsize=(6,5))
plt.plot([8, 16, 32, 64, 128, 256],train_error,label = 'train_error')
plt.plot([8, 16, 32, 64, 128, 256],test_error,label = 'test_error')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.title('Tanh Errors by Neurons')
print(f'Train Error: {train_error}')
print(f'Test Error: {test_error}')

plt.show()


#%%
##Using default activation = 'relu' with 2 layers

def cross_validate(features,target,folds):
    kf = KFold(n_splits=folds,shuffle=True)
    for train_index, test_index in kf.split(features):
        train_predictors = features.loc[train_index]
        train_target = target.loc[train_index]
        test_predictors = features.loc[test_index]
        test_target = target.loc[test_index]
        score_summary = dict()
        neurons = [(64,64),(128,128),(256,256)]
        for i in neurons: # for n_neighbors parameters
            fit = train(train_predictors,train_target,i,activation = 'relu')
            train_error = test(fit,train_predictors,train_target)
            test_error = test(fit,test_predictors,test_target)
            score_summary[f'train_error_{i}'] = train_error
            score_summary[f'test_error_{i}'] = test_error
        return score_summary

scores = cross_validate(data,target,4)


#%%
li=list(scores.items())
train_error = [x[1] for x in li if x[0].startswith('train')]
test_error = [x[1] for x in li if x[0].startswith('test')]
fig = plt.figure(figsize=(6,5))
plt.plot([64, 128, 256],train_error,label = 'train_error')
plt.plot([64, 128, 256],test_error,label = 'test_error')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.title('Relu Errors by Neurons: 2 Layers')
print(f'Train Error: {train_error}')
print(f'Test Error: {test_error}')

plt.show()


#%%
##Using activation = 'logistic' with 2 layers

def cross_validate(features,target,folds):
    kf = KFold(n_splits=folds,shuffle=True)
    for train_index, test_index in kf.split(features):
        train_predictors = features.loc[train_index]
        train_target = target.loc[train_index]
        test_predictors = features.loc[test_index]
        test_target = target.loc[test_index]
        score_summary = dict()
        neurons = [(8,8), (16,16), (32,32), (64,64), (128,126), (256,256)]
        for i in neurons: # for n_neighbors parameters
            fit = train(train_predictors,train_target,i,activation = 'logistic')
            train_error = test(fit,train_predictors,train_target)
            test_error = test(fit,test_predictors,test_target)
            score_summary[f'train_error_{i}'] = train_error
            score_summary[f'test_error_{i}'] = test_error
        return score_summary

scores = cross_validate(data,target,4)


#%%
li=list(scores.items())
train_error = [x[1] for x in li if x[0].startswith('train')]
test_error = [x[1] for x in li if x[0].startswith('test')]
fig = plt.figure(figsize=(6,5))
plt.plot([8,16,32,64, 128, 256],train_error,label = 'train_error')
plt.plot([8,16,32,64, 128, 256],test_error,label = 'test_error')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.title('Logistic Errors by Neurons: 2 Layers')
print(f'Train Error: {train_error}')
print(f'Test Error: {test_error}')

plt.show()


#%%
##Using activation = 'tanh' with 2 layers

def cross_validate(features,target,folds):
    kf = KFold(n_splits=folds,shuffle=True)
    for train_index, test_index in kf.split(features):
        train_predictors = features.loc[train_index]
        train_target = target.loc[train_index]
        test_predictors = features.loc[test_index]
        test_target = target.loc[test_index]
        score_summary = dict()
        neurons = [(8,8), (16,15), (32,32), (64,64), (128,128), (256,256)]
        for i in neurons: # for n_neighbors parameters
            fit = train(train_predictors,train_target,i,activation = 'relu')
            train_error = test(fit,train_predictors,train_target)
            test_error = test(fit,test_predictors,test_target)
            score_summary[f'train_error_{i}'] = train_error
            score_summary[f'test_error_{i}'] = test_error
        return score_summary

scores = cross_validate(data,target,4)


#%%
li=list(scores.items())
train_error = [x[1] for x in li if x[0].startswith('train')]
test_error = [x[1] for x in li if x[0].startswith('test')]
fig = plt.figure(figsize=(6,5))
plt.plot([8,16,32,64, 128, 256],train_error,label = 'train_error')
plt.plot([8,16,32,64, 128, 256],test_error,label = 'test_error')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.title('Tanh Errors by Neurons: 2 Layers')
print(f'Train Error: {train_error}')
print(f'Test Error: {test_error}')

plt.show()


#%%
##Using activation = 'relu' with 3 layers

def cross_validate(features,target,folds):
    kf = KFold(n_splits=folds,shuffle=True)
    for train_index, test_index in kf.split(features):
        train_predictors = features.loc[train_index]
        train_target = target.loc[train_index]
        test_predictors = features.loc[test_index]
        test_target = target.loc[test_index]
        score_summary = dict()
        neurons = [(10, 10, 10),(64, 64, 64),(128, 128, 128)]
        for i in neurons: # for n_neighbors parameters
            fit = train(train_predictors,train_target,i,activation = 'relu')
            train_error = test(fit,train_predictors,train_target)
            test_error = test(fit,test_predictors,test_target)
            score_summary[f'train_error_{i}'] = train_error
            score_summary[f'test_error_{i}'] = test_error
        return score_summary

scores = cross_validate(data,target,6)


#%%
li=list(scores.items())
train_error = [x[1] for x in li if x[0].startswith('train')]
test_error = [x[1] for x in li if x[0].startswith('test')]
fig = plt.figure(figsize=(6,5))
plt.plot([10,64,126],train_error,label = 'train_error')
plt.plot([10,64,126],test_error,label = 'test_error')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.title('Relu Errors by Neurons: 3 Layers')
print(f'Train Error: {train_error}')
print(f'Test Error: {test_error}')

plt.show()


#%%
##Using activation = 'logistinc' with 3 layers


def cross_validate(features,target,folds):
    kf = KFold(n_splits=folds,shuffle=True)
    for train_index, test_index in kf.split(features):
        train_predictors = features.loc[train_index]
        train_target = target.loc[train_index]
        test_predictors = features.loc[test_index]
        test_target = target.loc[test_index]
        score_summary = dict()
        neurons = [(10, 10, 10),(64, 64, 64),(128, 128, 128)]
        for i in neurons: # for n_neighbors parameters
            fit = train(train_predictors,train_target,i,activation = 'logistic')
            train_error = test(fit,train_predictors,train_target)
            test_error = test(fit,test_predictors,test_target)
            score_summary[f'train_error_{i}'] = train_error
            score_summary[f'test_error_{i}'] = test_error
        return score_summary

scores = cross_validate(data,target,6)


#%%
li=list(scores.items())
train_error = [x[1] for x in li if x[0].startswith('train')]
test_error = [x[1] for x in li if x[0].startswith('test')]
fig = plt.figure(figsize=(6,5))
plt.plot([10,64,126],train_error,label = 'train_error')
plt.plot([10,64,126],test_error,label = 'test_error')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.title('Logistic Errors by Neurons: 3 Layers')
print(f'Train Error: {train_error}')
print(f'Test Error: {test_error}')

plt.show()


#%%
##Using activation = 'tanh' with 3 layers

def cross_validate(features,target,folds):
    kf = KFold(n_splits=folds,shuffle=True)
    for train_index, test_index in kf.split(features):
        train_predictors = features.loc[train_index]
        train_target = target.loc[train_index]
        test_predictors = features.loc[test_index]
        test_target = target.loc[test_index]
        score_summary = dict()
        neurons = [(10, 10, 10),(64, 64, 64),(128, 128, 128)]
        for i in neurons: # for n_neighbors parameters
            fit = train(train_predictors,train_target,i,activation = 'tanh')
            train_error = test(fit,train_predictors,train_target)
            test_error = test(fit,test_predictors,test_target)
            score_summary[f'train_error_{i}'] = train_error
            score_summary[f'test_error_{i}'] = test_error
        return score_summary

scores = cross_validate(data,target,6)


#%%
li=list(scores.items())
train_error = [x[1] for x in li if x[0].startswith('train')]
test_error = [x[1] for x in li if x[0].startswith('test')]
fig = plt.figure(figsize=(6,5))
plt.plot([10,64,126],train_error,label = 'train_error')
plt.plot([10,64,126],test_error,label = 'test_error')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.title('Tanh Errors by Neurons: 3 Layers')
print(f'Train Error: {train_error}')
print(f'Test Error: {test_error}')

plt.show()


