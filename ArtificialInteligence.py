# preprocessing
from unicodedata    import normalize
from sklearn.utils  import resample

import pandas as pd

# general import
import numpy as np
from   numpy import random

# machine learning
import  tensorflow  as tf
import  keras

from    keras.layers.normalization  import BatchNormalization
from    keras.utils                 import to_categorical
from    keras                       import optimizers
from    numpy                       import argmax

from    sklearn.model_selection     import KFold
from    sklearn.model_selection     import StratifiedKFold
from    sklearn.model_selection     import train_test_split


# machine learning metrics
import sklearn.metrics as sklm

from numpy import argmax





def getMetrics():
    class Metrics(keras.callbacks.TensorBoard):

        def on_train_begin(self, logs={}):
            self.confusion = []
            self.precision = []
            self.recall = []
            self.f1s = []
            self.kappa = []
            self.auc = []
            self.accuracy = []
            self.counter = 0

        def on_epoch_end(self, epoch, logs={}):
            self.counter += 1
            score = np.asarray(self.model.predict(self.validation_data[0]))
            predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
            targ = self.validation_data[1]
            # self.auc.append(sklm.roc_auc_score(targ, score))
            targ = argmax(targ, axis=1)
            predict = argmax(predict, axis=1)

            scores = dict(
                ephoc_f1        = sklm.f1_score(targ, predict),
                ephoc_recall    = sklm.recall_score(targ, predict),
                ephoc_precision = sklm.precision_score(targ, predict),
                ephoc_kappa     = sklm.cohen_kappa_score(targ, predict),
                ephoc_acc       = sklm.accuracy_score(targ, predict),
                epoch_roc       = sklm.roc_auc_score(targ, predict,average='weighted')
            )
            for score in scores:
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = float(scores[score])
                summary_value.tag = score
                self.writer.add_summary(summary, self.counter)
                print(score, scores[score])
                # tf.summary.scalar(score[0],random.random())
            self.writer.flush()
            self.confusion.append(sklm.confusion_matrix(targ, predict))
            print(sklm.confusion_matrix(targ, predict))
            # self.precision.append(scores['precision'])
            # self.recall.append(scores['recall'])
            # self.f1s.append(scores['f1'])
            # self.kappa.append(scores['kappa'])
            # self.accuracy.append(scores['accuracy'])

            super().on_epoch_end(epoch, logs)

        # def on_batch_end(self, batch, logs=None):
        #     logs = logs or {}

        #     # if self.write_batch_performance == True:
        #     for name, value in logs.items():
        #         if name in ['batch', 'size']:
        #             continue

        #         summary = tf.Summary()
        #         summary_value = summary.value.add()
        #         summary_value.simple_value = value.item()
        #         summary_value.tag = name
        #         self.writer.add_summary(summary)
        #     # self.writer.flush()
        #     # self.seen += self.batch_size
        #     super().on_batch_end(batch,logs)

    return Metrics

def mlpModel(input, regression=False, layers=[10, 20, 20, 16, 2], dropout=None, last_activation='softmax', activations='relu', optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy', 'mse']):
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout

    if type(activations) == str:
        activations = [activations]*len(layers)
    elif type(activations) == list:
        if len(activations) != len(layers):
            raise Exception(
                'activations parameter must have the same size of layers parameters; or it must be an string with the name of activation')
        if False in map(lambda x: type(x) == str, activations): raise Exception(
            'All values of parameter activations must be string typed.')

    if dropout:
        if type(dropout) == str:
            raise Exception('Can not use string value on dropout')
        elif type(dropout) == list:
            if len(dropout) != len(layers):
                raise Exception(
                    'Dropout parameter must have the same size of layers parameters')

    model = Sequential()
    model.add(Dense(layers[0], input_dim=input))
    for i in range(0, len(layers)):
        model.add(Activation(activations[i]))
        if(dropout != None):
            if(dropout[i] > 0):
                model.add(Dropout(dropout[i]))
        model.add(BatchNormalization())
        model.add(Dense(layers[i]))
    if(not regression):
        model.add(Activation(last_activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def light(features):
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.01, nesterov=True)
    model = mlpModel(features,
                     regression=False,
                     # layers=[5000, 6000, 5000, 1000, 100,
                     # 100, 100, 100, 60, 30, 20, 10, 2],
                     layers=[100, 50, 30, 10, 2],
                     # dropout=[0.9,0.3,0.1,0.1 ,0.1 ,0 ,0 ,.05 ,0],
                     # dropout=[0.1,0.2,0.1,0.1 ,0],
                     dropout=None,
                     # optimizer=sgd,
                     # optimizer='sgd',
                     optimizer=optimizers.Adam(),

                     )
    return model


def deep(features):
    from keras import losses
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.01, nesterov=True)
    model = mlpModel(   features,
                        regression=False,
                        layers=[2000, 1000, 500, 100, 100,
                            100, 100, 100, 60, 30, 20, 10, 2],
                        #  layers=[100, 50, 30, 10, 2],
                        dropout=[0.8,0.5,0.5,0.5 ,0.5 ,0 ,0 ,0,0,0,0,0,0],
                        # dropout=[0.1,0.2,0.1,0.1 ,0],
                        #  dropout=None,
                        # optimizer=sgd,
                        # optimizer='sgd',
                        optimizer=optimizers.Adam(),
                        loss=losses.mean_absolute_percentage_error,
                        metrics=['mean_squared_error','mean_absolute_percentage_error']
                    )
    return model


def train_keras(X_train, y_train, checkpoint='models/model', log_dir='log', kfold=10, model_function=light, class_weight=None):
    
    print('Keras Version: {}'.format(keras.__version__))
    print('Tensorflow Version: {}'.format(tf.__version__))

    Metrics = getMetrics()
    in_sz = X_train.shape[1]
    nclasses =  y_train.nunique()
    model = model_function(features = in_sz)
    try:
        model.load_weights(checkpoint)
    except Exception as ex:
        print(ex)
    # kf = KFold(n_splits = kfold)
    kf = StratifiedKFold(n_splits = kfold)
    while  True:
        for train_index,test_index in kf.split(X_train,y=y_train):

            y_train_binary, y_test_binary = \
                (to_categorical(y_train.loc[y_train.index[train_index]], num_classes=nclasses),
                to_categorical(y_train.loc[y_train.index[test_index]],num_classes=nclasses))
            
            x_kf_train, x_kf_test = \
                (X_train.loc[X_train.index[train_index]],
                X_train.loc[X_train.index[test_index]])
            
            print(x_kf_train.shape)
            print(y_train_binary.shape)
            
            print(x_kf_test.shape)
            print(y_test_binary.shape)
            from sklearn.utils import class_weight
            class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
            print('Class Weights: ',class_weights)
            model.fit(x_kf_train, y_train_binary,
                    epochs=200,
                    class_weight=class_weights,
                    # batch_size=100,
                    validation_data=(x_kf_test,y_test_binary),
                    # validation_split=0.2,
                    # steps_per_epoch=10,
                    callbacks=[
                        keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.02, patience=50, verbose=0, mode='auto', cooldown=10, min_lr=0.0001),
                        keras.callbacks.ModelCheckpoint(checkpoint, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1),
                        Metrics(log_dir=log_dir, histogram_freq=0, batch_size=100, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
                    ])

    # model.predict(self.validation_data[0])
    # score = model.evaluate(X_test, y_test_binary, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    pass

def remover_acentos(txt):
    return normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')


def fixnames(ds):
    return ds.str.lower().str.strip().apply(lambda x: remover_acentos(x))


def from_categorical(data):
    return argmax(data,axis=1) 




def ballanced_split(x, y,expand=False):
    if expand:
        count = y.value_counts().max()
    else: 
        count = y.value_counts().min()

    dataset = pd.concat([x, y], axis=1)
    
    d1 = dataset[dataset[y.name] == 1]
    d0 = dataset[dataset[y.name] == 0]

    # resampled = pd.concat([resample(d1,replace=True,n_samples=min_count),resample(d0,replace=True,n_samples=min_count)])
    # print(resampled.is_distrato.value_counts())
    # print(resampled.index)

    # test_size = round(count*2*test_size)
    # train_size = (count*2)-test_size

    # print('Train Size:', train_size)
    # print('Test Size:', test_size)
    d1_train_index = random.choice(list(d1.index.values), count)
    d0_train_index = random.choice(list(d0.index.values), count)
    train_index = np.concatenate((d1_train_index,d0_train_index))
    # y.rename()
    
    x = dataset.loc[train_index,dataset.columns[dataset.columns!=y.name]]
    y = dataset[y.name][train_index]
    
    print(y.value_counts())
    # correlation = pd.concat([X_train, y_train]).corr()['is_distrato'][:-1]
    # print(correlation)
    # well_correlated_features = correlation[correlation[0].abs()>=0.4].index
    # print(well_correlated_features)
    # X_train = X_train.loc[:,well_correlated_features]
    return x, y
