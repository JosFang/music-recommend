from data import *
from utils import *
from model import *

import argparse
import os
import time

parser = argparse.ArgumentParser(description='transformer for music recommendation')
parser.add_argument('--data_file', type=str, default='music_Recommend_train_data.txt',
                    help='train data source')
parser.add_argument('--trian_proportion', type=float, default=0.8, help='trian proprotion')
parser.add_argument('--batch_size', type=int, default=258, help='#sample of each minibatch')
parser.add_argument('--batch_size_test', type=int, default=8, help='#sample of each test minibatch')
parser.add_argument('--epoch_num', type=int, default=15, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=768, help='#dim of hidden state')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--n_layer', type=int, default=12)
parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
parser.add_argument('--sequence_length', type=int, default=20, help='sequence max length')
parser.add_argument('--num_attention_heads', type=int, default=12, help='num attention heads')
parser.add_argument('--intermediate_size', type=int, default=3072, help='intermediate_size')
parser.add_argument('--neg_num', type=int, default=10, help='negative example number')
parser.add_argument('--top_k', type=int, default=10, help='top k test')
parser.add_argument('--demo_model', type=str, default="", help='model file')


args = parser.parse_args()

data_file = os.path.join('.', 'data', args.data_file)
train_data, test_data, user_musics_dic, music_num = read_corpus(data_file, args.trian_proportion)

# train_data = train_data[:9]
# test_data = test_data[:9]

print("train session:", len(train_data))
print("test session:", len(test_data))
print("music num:", music_num)

# a = input()

paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', "model_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

if args.mode == 'train':
    model = music_recommend_model(args, music_num, user_musics_dic, paths)
    model.build_graph()
    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)