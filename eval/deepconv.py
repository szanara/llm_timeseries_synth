import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path_real', type=str,help='real data path')
parser.add_argument('--path_synthetic', type=str,help='synthetic data path')
parser.add_argument('--mode', type = str, help='tstr, trts, aug')
parser.add_argument('--dataset', type=str,help='UTD-MHAD1_1S.npz ,UTD-MHAD2_1S.npz, USCHAD.npz, MHEALTH.npz, WHARF.npz, WISDM.npz ')
parser.add_argument('--checkpoints',type =str, help='path to save the checkpoints')
args = parser.parse_args()






###############3 libraries#################3
from tensorflow import keras
import models
from models import make_model, model, ModelCheckpoint,EarlyStopping
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, f1_score 
from sklearn import metrics
import scipy.stats as st
import os
from sklearn.utils import shuffle
import sys
sys.path.append('/home/cz/mds3/GEMINI')
from utils import *
###################3 model setup #####################
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


EPOCH = 16
BATCH_SIZE = 32
LSTM_UNITS = 32
CNN_FILTERS = 3
NUM_LSTM_LAYERS = 1
LEARNING_RATE = 1e-4
PATIENCE = 20
SEED = 0
F = 32
D = 10
DATA_FILES = args.dataset
MODE = args.mode
SAVE_DIR =args.checkpoints+'/'+args.mode+'/'+args.dataset

os.makedirs(args.checkpoints+'/'+args.mode+'/'+args.dataset, exist_ok=True)
SEED = 0 
random.seed(SEED)
np.random.seed(SEED)
#tf.set_random_seed(0)
tf.set_random_seed(0)

#tf.random.set_seed(0)


avg_acc = []
avg_recall = []
avg_f1 = []
early_stopping_epoch_list = []

if '_' in args.mode:
    args.mode,n_augment = args.mode.split('_')
    print(f'AUG: {n_augment}')
    amount = int(n_augment)
for i in range(10):



    ############################ data #####################33

    if args.mode == 'aug':
        print('TRAINING Augmenttaion')
        
        get = Data(split = 'train', fold_idx = i , dataset= args.dataset,path=args.path_real)
        X_real, y_real = get.load()
        X_synt, y_synt = augmenting(fold =i, dataset=args.dataset,path_synthetic=args.path_synthetic, amount= amount)
        X_all = np.concatenate([X_real,X_synt])
        y_all = np.concatenate([y_real, y_synt])

        X_to_train, y_to_train = shuffle(X_all, y_all, random_state=42)
        y_hot_to_train = one_hot_encode(y_to_train)

        d = Data(split='test',fold_idx=i, dataset=args.dataset,path = args.path_real)
        X_to_test, y_to_test = d.load()
        y_hot_to_test = one_hot_encode(y_to_test)


    elif args.mode =='tstr':
        print('TRAINING TSTR') 

        X_synt, y_synt = adding_synthetic(fold=i,dataset=args.dataset, path=args.path_synthetic)
        X_to_train, y_to_train = shuffle(X_synt, y_synt, random_state=42)
        y_hot_to_train = one_hot_encode(y_to_train)


        d = Data(split='test',fold_idx=i, dataset=args.dataset,path = args.path_real)
        X_to_test, y_to_test = d.load()
        y_hot_to_test = one_hot_encode(y_to_test)
      
    else:
        print('TRAINING TRTS')

        X_synt, y_synt = TRTS(fold = i ,dataset=args.dataset, path_real = args.path_real, pathfinal= args.path_synthetic)
        X_to_test, y_to_test = shuffle(X_synt, y_synt ,random_state=42)
        y_hot_to_test = one_hot_encode(y_to_test)

        d = Data(split='train',fold_idx=i, dataset=args.dataset,path = args.path_real)
        X_to_train, y_to_train = d.load()
        y_hot_to_train = one_hot_encode(y_to_train)
        
 
    #################### model ###################################3

    NUM_LABELS = len(set(y_to_train))

      
        
    
    X_train, y_train, y_train_one_hot = X_to_train, y_to_train,y_hot_to_train
    X_test,y_test,  y_test_one_hot = X_to_test,y_to_test,y_hot_to_test
    
    
    X_train_ = np.expand_dims(X_train, axis = 3)
    X_test_ = np.expand_dims(X_test, axis = 3)

    train_trailing_samples =  X_train_.shape[0] % BATCH_SIZE
    test_trailing_samples =  X_test_.shape[0] % BATCH_SIZE


    if train_trailing_samples!= 0:
        X_train_ = X_train_[0:-train_trailing_samples]
        y_train_one_hot = y_train_one_hot[0:-train_trailing_samples]
        y_train = y_train[0:-train_trailing_samples]

    if test_trailing_samples!= 0:
        X_test_ = X_test_[0:-test_trailing_samples]
        y_test_one_hot = y_test_one_hot[0:-test_trailing_samples]
        y_test = y_test[0:-test_trailing_samples]

        print (y_train.shape, y_test.shape)   

    rnn_model = model(x_train = X_train_, num_labels = NUM_LABELS, LSTM_units = LSTM_UNITS, \
        num_conv_filters = CNN_FILTERS, batch_size = BATCH_SIZE, F = F, D= D)
    if args.mode != 'aug':
        model_filename = args.checkpoints+'/'+args.mode+'/'+args.dataset+f'/deep_cov-{i}.h5'
    else: 
        model_filename = args.checkpoints+'/'+args.mode+'/'+n_augment+'/'+args.dataset+f'/deep_cov-{i}.h5'
    model_directory = os.path.dirname(model_filename)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    callbacks = [ModelCheckpoint(filepath=model_filename, monitor = 'val_accuracy', save_weights_only=True, save_best_only=False), EarlyStopping(monitor='val_accuracy', patience=PATIENCE)]#, LearningRateScheduler()]

    opt = tf.keras.optimizers.Adam(clipnorm=1.)

    rnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    history = rnn_model.fit(X_train_, y_train_one_hot, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks, validation_data=(X_test_, y_test_one_hot))

    early_stopping_epoch = callbacks[1].stopped_epoch - PATIENCE + 1
    print('Early stopping epoch: ' + str(early_stopping_epoch))
    early_stopping_epoch_list.append(early_stopping_epoch)

    if early_stopping_epoch <= 0:
        early_stopping_epoch = -100

    # Evaluate model and predict data on TEST 
    print("******Evaluating TEST set*********")
    rnn_model.load_weights(model_filename)

    y_test_predict = rnn_model.predict(X_test_, batch_size = BATCH_SIZE)
    y_test_predict = np.array(y_test_predict)
    y_test_predict = np.argmax(y_test_predict, axis=1)
    print(y_test_predict.shape, 'PREDICS', y_test.shape)

       
    MAE = metrics.mean_absolute_error(y_test, y_test_predict, sample_weight=None, multioutput='uniform_average')

    acc_fold = accuracy_score(y_test, y_test_predict)
    avg_acc.append(acc_fold)

    recall_fold = recall_score(y_test, y_test_predict, average='macro')
    avg_recall.append(recall_fold)

    f1_fold  = f1_score(y_test, y_test_predict, average='macro')
    avg_f1.append(f1_fold)

    with open(SAVE_DIR + '/results_model_with_self_attn_' + MODE + '.csv', 'a') as out_stream:
        out_stream.write(str(SEED) + ', '  + ', ' + str(i) + ', ' + str(early_stopping_epoch) + ', '  + str(acc_fold) + ', ' + str(MAE) + ', ' + str(recall_fold) + ', ' + str(f1_fold) + '\n')


    print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold, i))
    print('______________________________________________________')
    #K.clear_session()
    tf.keras.backend.clear_session()

ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc = np.mean(avg_f1), scale=st.sem(avg_f1))
dict_final={'acc':{np.mean(avg_acc)},
             'recall':{np.mean(avg_recall)},
              'f1' :{np.mean(avg_f1)},
              'std_acc':{np.mean(avg_acc)-ic_acc[0]},
              'std_raecall':{np.mean(avg_recall)-ic_recall[0]},
              'std_f1':{np.mean(avg_f1)-ic_f1[0]}}
if args.mode!='aug':
    np.save(args.checkpoints+'/'+args.mode+'/'+args.dataset+f'/results.npy',dict_final)
else: 
    np.save(args.checkpoints+'/'+args.mode+'/'+n_augment+'/'+args.dataset+f'/results.npy',dict_final)
print('Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
print('Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
print('Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))