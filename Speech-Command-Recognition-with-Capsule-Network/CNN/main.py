###############
# CNN Main.py #
###############

import tensorflow as tf
from tqdm import tqdm
import numpy as np
from termcolor import colored, cprint
import time, os, sys, scipy
sys.path.append('../')
# User
import core.data_utils as du
import core.analysis as analysis
import model as md
from core.args import args, parameter_print
# Keras
from keras import backend as K
from keras import callbacks, layers, optimizers
from keras.utils import multi_gpu_model
from keras.losses import categorical_crossentropy

noise_list = ['doing_the_dishes','dude_miaowing','exercise_bike','pink_noise','running_tap','white_noise']

def train(multi_model, data, save_path, args):
    trX, trY, vaX, vaY = data
    #print(str(trX.shape),str(trY.shape),str(vaX.shape),str(vaY.shape))

    multi_model.compile(optimizer=optimizers.Adam(lr=args.learning_rate),
                  loss=categorical_crossentropy,
                  metrics=['accuracy']
                  )
    # callbacks
    log = callbacks.CSVLogger(save_path + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(save_path + '/weights-{epoch:03d}.h5py', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    earlystop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto')
    #tb = callbacks.TensorBoard(log_dir=save_path + '/tensorboard-logs', batch_size=args.batch_size, histogram_freq=args.debug)
    #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.learning_rate * (0.9 ** args.num_epoch))
    
    multi_model.fit(trX, trY,
              batch_size=args.batch_size, epochs=args.num_epoch,
              #validation_split = 0.1,
              validation_data=[vaX,vaY], 
              shuffle = True,
              callbacks=[log, checkpoint, earlystop])

def test(model, data, args, label_names, matrix_name=None):
    if matrix_name is None:
        matrix_name = '/ConfusionMatrix_'+args.ex_name+'_'+args.test_with
    a = analysis.Analysis(args)
    start_time = time.time()
    teX,teY = data
    print('-'*20 + 'Begin: test with ' + '-'*20)
    y_pred = model.predict(teX,batch_size=args.batch_size)

    # Test with all labels
    acc = float(np.sum(np.argmax(y_pred, 1) == np.argmax(teY, 1)))/float(teY.shape[0])
    print('Test with all labels acc:', acc )
    A = np.argmax(y_pred, 1)
    B = np.argmax(teY, 1)
    assert A.shape[0] == B.shape[0]
    teY = np.argmax(teY, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    
    confusion_matrix = a.ConfusionMatrix(y_pred=y_pred, teY=teY, labels=label_names)
    a.Matrix2Png(filename=matrix_name+'.png')
    a.Matrix2Csv(Array=confusion_matrix,filename=matrix_name+'.csv')

    return acc
    

if __name__ == "__main__":
    args = args()

    ex_name = args.ex_name+'_'+args.train_with
    ex_name += '_open_set' if args.open_set else ''
    parameter_print(args,ex_name=ex_name,ModelType='CNN')
    save_path = os.path.join(args.project_path,'save',args.model,ex_name)
    cprint('save_path: '+str(save_path),'yellow')
    
    all_label_names = []
    with open(os.path.join(args.label_names_path), 'r') as _file:
        all_label_names = _file.readlines()

    label_names = []
    open_labels = [] #remains empty and unused when using closed dataset
    if args.open_set:
        try:
            with open(args.open_labels_path) as file:
                for line in file.readlines():
                    open_labels.append(int(line))
                    label_names.append(all_label_names[int(line)])
        except:
            raise ValueError("To use the open set option you need to specify the path to the file that specify which label to keep.")
    else:
        label_names = all_label_names


    # Data Load
    data = du.DATA(args, open_labels)
    X,Y = data[0], data[1]

    # Define Model
    with tf.device('/cpu:0'):
        if args.ex_name == '0314':
            model = md.CNN_0314(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))),)
        elif args.ex_name == '0320':
            model = md.CNN_0320(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))),)
        elif args.ex_name == '0320_326464' or args.ex_name =='dim0_0929':
            model = md.CNN_0320_326464(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))),)
        elif 'onlydense' in args.ex_name.lower():
        	model = md.OnlyDense(input_shape=X.shape[1:],n_class=len(np.unique(np.argmax(Y, 1))),DenseChannel=args.DenseChannel)
        elif 'ref_cnn' in args.ex_name.lower():
            model = md.ref_cnn(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))), model_size_info=args.model_size_info)
        elif 'ref_2014icassp_dnn' in args.ex_name.lower():
        	model = md.ref_2014icassp_dnn(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))), model_size_info=args.model_size_info)
        elif 'ref_2015is_cnn' in args.ex_name.lower():
            model = md.ref_2015IS_cnn(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))), model_size_info=args.model_size_info)
        elif 'ref_rnn' in args.ex_name.lower():
            model = md.ref_rnn(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))), model_size_info=args.model_size_info)
        elif 'ref_crnn' in args.ex_name.lower():
        	model = md.ref_crnn(input_shape=X.shape[1:], n_class=len(np.unique(np.argmax(Y, 1))), model_size_info=args.model_size_info)
        else:
            cprint('No matched model name.','red')
            model = CNN(input_shape=X.shape[1:],
                n_class=len(np.unique(np.argmax(Y, 1))),
                CNNkernel=args.CNNkernel,
                CNNChannel=args.CNNChannel,
                DenseChannel=args.DenseChannel,
                )

    model.summary()
    multi_model = model

    # Save path and load model
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if args.keep:  # init the model weights with provided one
        cprint('load weight from:' + save_path + '/weights-%03d.h5py'%args.keep, 'yellow')
        multi_model.load_weights(save_path + '/weights-%03d.h5py'% args.keep)
        #model.load(save_path)
    else:
        cprint('save weight to: ' + save_path, 'yellow')

    # Model training or testing
    # Train
    if args.is_training == 'TRAIN':
        train(multi_model,data=data,save_path=save_path,args=args)
        model.save_weights(save_path + '/trained_model.h5py')
        print('Trained model saved to \'%s/trained_model.h5py\'' % save_path)
    # Test
    elif args.is_training == 'TEST':
        if args.keep == 0:
            raise ValueError('No weights are provided.')
        else:
        	# csv file
            test_result = save_path + '/' + args.test_with + '_test_result.csv'
            fd_test_result = open(test_result,'a')
            fd_test_result.write('test\n')
            fd_test_result.write('test_mode,accuracy\n')
            # clean test
            print('*'*30 + 'clean exp' + '*'*30)
            multi_model.load_weights(save_path + '/trained_model.h5py')
            acc = test(multi_model, data=data,args=args, matrix_name='/ConfusionMatrix_'+args.ex_name+'_clean', label_names=label_names)
            # for fair time comparison
            #label30_acc, label21_acc = test(multi_model, data=data,args=args)
            fd_test_result.write('clean,'+str(acc)+'\n')
            fd_test_result.flush()
            for i in range(6):
                print('*'*30 + 'Noisy '+ str(i+2) +' exp' + '*'*30)    
                if args.test_by=='noise':
                	cprint('Test by specific noise type: ' + str(noise_list[i]),'red')
                	teX, teY = du.load_specific_noisy_data(args, noise_list[i], open_labels=open_labels)
                elif args.test_by=='echo':
                	cprint('Test by echo noise.','red')
                	teX, teY = du.load_specific_noisy_data(args.data_path, 'echo', open_labels=open_labels)
                else:

                    cprint('Test by SNR value.','red')
                    for snr in range(-10, 20+1, 5):
                        
                        try:
                            cprint('Test with SNR'+str(snr),'red')
                            teX, teY = du.load_specific_noisy_data(args, 'doing_the_dishes_SNR'+str(snr), open_labels=open_labels)
                            #teX = np.expand_dims(teX[:,:,:,1],axis=3)
                            teX = du.Dimension(teX,args.dimension)
                            acc = test(multi_model, data=(teX, teY),args=args, matrix_name='/ConfusionMatrix_'+args.ex_name+'_noisy'+str(snr), label_names=label_names)
                            # csv write
                            fd_test_result.write('noisy'+str(snr)+','+str(acc)+'\n')

                            cprint('Test with Clean + SNR'+str(snr),'red')
                            teX, teY = du.load_random_noisy_data(args.data_path,'TEST',args.mode, args.feature_len, SNR=snr, open_set=args.open_set, open_labels=open_labels)
                            #teX = np.expand_dims(teX[:,:,:,1],axis=3)
                            teX = du.Dimension(teX,args.dimension)
                            acc = test(multi_model, data=(teX, teY),args=args, matrix_name='/ConfusionMatrix_'+args.ex_name+'_mixed'+str(snr), label_names=label_names)
                            # csv write
                            fd_test_result.write('mixed'+str(snr)+','+str(acc)+'\n')
                        except:
                            continue

                    fd_test_result.flush()
                    break

                #teX = np.expand_dims(teX[:,:,:,1],axis=3)
                teX = du.Dimension(teX,args.dimension)#teX = np.expand_dims(teX[:,:,:,1],axis=3)
                acc = test(multi_model, data=(teX, teY),args=args, label_names=label_names)
                # csv write
                fd_test_result.write('noisy'+str(i)+','+str(acc)+'\n')
                fd_test_result.flush()
            fd_test_result.close()
    else:
        raise ValueError('Wrong "is_training" value')#'could not find %c in %s' % (ch,str)) 
# Code end
# For not decreasing issue: https://github.com/XifengGuo/CapsNet-Keras/issues/48