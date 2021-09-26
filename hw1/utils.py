import argparse


'''
Parser utils
'''

def train_parser():
    parser = argparse.ArgumentParser(description="Training simple instrument classifier")

    # parser.add_argument('--features',              type=str,     required=True,    help="Type of features (mfcc, d_mfcc, )")
    parser.add_argument('--pca',             type=bool,      default=False,                                             help='PCA feature compression')
    parser.add_argument('--pca_dim',         type=int,       default=32,                                                help='PCA dimension')
    parser.add_argument('--classifier',      type=str,       default='sgd',        choices=['sgd', 'knn', 'svc'],       help='Classifier')

    return parser.parse_args()