import argparse

'''
Parser utils
'''

def train_multilabel():
    parser = argparse.ArgumentParser(description="Training model for multi-label classification")
    
    parser.add_argument('--model',           type=str,       default='baseline',        choices=['baseline', 'cnn2d', 'cnn2ddeep', 'cnntf', 'cnntfdeep', 'cnntf2'],    help='model architecture')
    parser.add_argument('--writer',          type=str,       default='',                                                                                               help='postfix for tensorboard file')
    parser.add_argument('--batch_size',      type=int,       default=16,                                                                                               help='batch size')
    parser.add_argument('--num_workers',     type=int,       default=2,                                                                                                help='number of workers')
    parser.add_argument('--sample_rate',     type=int,       default=16000,                                                                                            help='sampling rate for audio data')
    parser.add_argument('--duration',        type=int,       default=3,                                                                                                help='duration of each chunk')
    parser.add_argument('--optimizer',       type=str,       default='sgd',             choices=['adam', 'sgd'],                                                       help='optimizer')

    parser.add_argument('--num_epochs',      type=int,       default=10,                help='number of training epoch')
    parser.add_argument('--lr',              type=float,     default=1e-3,              help='learning rate')
    parser.add_argument('--sr',              type=float,     default=1e-5,              help='stopping rate')
    parser.add_argument('--momentum',        type=float,     default=0.9,               help='momentum for sgd')
    parser.add_argument('--weight_decay',    type=float,     default=0.0,               help='weight decay - L2 regularization weight')

    return parser.parse_args()


def train_metric():
    parser = argparse.ArgumentParser(description="Training model for metric learning")
    
    parser.add_argument('--model',           type=str,       default='linear',          choices=['linear', 'conv1d', 'conv2d', 'tf', 'tfdeep', 'tf2'],               help='model architecture')
    parser.add_argument('--writer',          type=str,       default='',                                                                                             help='postfix for tensorboard file')
    parser.add_argument('--batch_size',      type=int,       default=16,                                                                                             help='batch size')
    parser.add_argument('--num_workers',     type=int,       default=2,                                                                                              help='number of workers')
    parser.add_argument('--sample_rate',     type=int,       default=16000,                                                                                          help='sampling rate for audio data')
    parser.add_argument('--duration',        type=int,       default=3,                                                                                              help='duration of each chunk')
    parser.add_argument('--optimizer',       type=str,       default='sgd',             choices=['adam', 'sgd'],                                                     help='optimizer')

    parser.add_argument('--num_epochs',      type=int,       default=3,                 help='number of training epoch')
    parser.add_argument('--lr',              type=float,     default=1e-3,              help='learning rate')
    parser.add_argument('--sr',              type=float,     default=1e-5,              help='stopping rate')
    parser.add_argument('--momentum',        type=float,     default=0.9,               help='momentum for sgd')
    parser.add_argument('--weight_decay',    type=float,     default=0.0,               help='weight decay - L2 regularization weight')

    
    return parser.parse_args()