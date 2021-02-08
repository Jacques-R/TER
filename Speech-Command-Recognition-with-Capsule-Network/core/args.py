import argparse
from termcolor import colored, cprint

def args():
    parser = argparse.ArgumentParser(description="Preprocessing for Google Speech Command dataset.")
    parser.add_argument('--mode', default='fbank',choices=['mfcc','fbank'], type=str)
    parser.add_argument('--feature_len', default=40, type=int)
    parser.add_argument('--noise_name', default='clean', 
        choices=['clean','exercise_bike','pink_noise','doing_the_dishes','running_tap','dude_miaowing','white_noise'],
        type=str)
    parser.add_argument('--is_training', default='TRAIN',choices=['TRAIN','TEST'], type=str)

    #experiment
    #Path
    parser.add_argument('--data_path', default= '/DATA/jsbae/KWS_feature_saved', type=str)
    parser.add_argument('--project_path', default='/DATA/jsbae/STT2/SCR_INTERSPEECH2018', type=str)
    parser.add_argument('-lnp','--labels_name_path', default='/DATA/jsbae/labels_name.txt', type=str)
    parser.add_argument('-olp','--open_labels_path', default='/DATA/jsbae/open_labels.txt', type=str)
    # Parameters
    parser.add_argument('-lr','--learning_rate', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gpus', default=2, type=int)
    parser.add_argument('--SNR', default=None, type=int)
    # Should type
    parser.add_argument('-m','--model', default=None, type=str)
    parser.add_argument('-ex','--ex_name', default=None, type=str)
    parser.add_argument('-tr','--train_with', default='clean', type=str)
    parser.add_argument('-te','--test_with', default='clean', type=str)
    parser.add_argument('--test_by', default=None,choices=['noise','echo'], type=str)
    parser.add_argument('--open_set', default=None, type=bool)
    parser.add_argument('-fte','--final_test_with', default=None,choices=['clean','noisy'], type=str)
    # Added
    parser.add_argument('--keep', default=None, type=int)
    parser.add_argument('-dim','--dimension', default=3, choices=[0,1,2,3],type=int)
    #
    parser.add_argument('--keep_prob', default=0.7, type=float)
    # CNN experiment
    parser.add_argument('-CNNk','--CNNkernel', default=5, type=int, help="CNN 3 layer's kernel size.")
    parser.add_argument('-CNNC','--CNNChannel', default=32, type=int, help="CNN 3 layer's channel size.")
    parser.add_argument('-Dense','--DenseChannel', default=128, type=int, help="Dense 2 layer's channel size.")
    parser.add_argument(
      '--model_size_info',
      type=int,
      nargs="+",
      default=[128,128,128],
      help='Model dimensions - different for various models')

    args = parser.parse_args()
    return args

def parameter_print(args,ex_name,ModelType):
    cprint('experiment name: '+ ex_name, 'cyan')
    cprint('batchsize: ' + str(args.batch_size), 'cyan')
    cprint('keep_prob: ' + str(args.keep_prob), 'cyan')
    cprint('learning_rate: ' + str(args.learning_rate), 'cyan')
    cprint('num_epoch: ' + str(args.num_epoch), 'cyan')
    cprint('*'*10 + str(ModelType) + '*'*10, 'cyan')
    if ModelType=='CNN':
        cprint('CNNk: ' + str(args.CNNkernel), 'cyan')
        cprint('CNNC: ' + str(args.CNNChannel), 'cyan')
        cprint('Dense: ' + str(args.DenseChannel), 'cyan')