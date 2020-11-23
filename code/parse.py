"""

"""
import argparse

parser = argparse.ArgumentParser(description="Go!")
parser.add_argument('--batch_size', type=int, default=200, help="the batch size for training procedure")
parser.add_argument('--batch_n_ID', type=int, default=25, help="ID sampling size for few-shot learning")
parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
parser.add_argument('--alpha', type=float, default=0.1, help="coefficient to balance `cold-start' and `warm-up'")
parser.add_argument('--emb_size', type=int, default=128, help="length of embedding vectors")
parser.add_argument('--model', type=str, default='deepFM', help='model name')

args = parser.parse_args()

args.log = "logs/{}.csv".format(args.model)
args.saver_path = "saver/model-" + args.model
