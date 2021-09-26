import argparse


'''
Parser utils
'''

def train_parser():
    parser = argparse.ArgumentParser(description="Training simple instrument classifier")

    parser.add_argument('--mfcc_delta',      type=str,       default='none',       choices=['none', 'd', 'dd'],         help="Whether MFCC delta summary will be used or not")
    parser.add_argument('--rms_delta',       type=str,       default='none',       choices=['none', 'd', 'dd'],         help="Whether RMS delta summary will be used or not")
    parser.add_argument('--compress',        type=str,       default='none',       choices=['none', 'pca', 'dct'],      help='Feature compression')
    parser.add_argument('--dimension',       type=int,       default=32,                                                help='Resulting dimension after feature compression')
    parser.add_argument('--classifier',      type=str,       default='sgd',        choices=['sgd', 'knn', 'svc'],       help='Classifier')

    return parser.parse_args()