import os
import sys
import argparse

import numpy as np
import torch

from collections import OrderedDict
from dataloader import Dataset
from evaluation import Evaluator
from experiment import EarlyStop, train_model
from utils import Config, Logger, ResultTable, make_log_dir, set_random_seed


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='saves/TM_Generation/10_20210611-1719', metavar='P')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

if args.seed is not None:
    set_random_seed(args.seed)

# read configs
config = Config(main_conf_path=args.path, model_conf_path=args.path)
if args.gpu is not None:
    config.main_config['Experiment']['gpu'] = args.gpu


gpu = config.get_param('Experiment', 'gpu')
gpu = str(gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = config.get_param('Experiment', 'model_name')

log_dir = args.path
logger = Logger(log_dir)

dataset = Dataset(model_name, **config['Dataset'])

# early stop
early_stop = EarlyStop(**config['EarlyStop'])

# evaluator
evaluator = Evaluator(early_stop.early_stop_measure, **config['Evaluator'])

import model

MODEL_CLASS = getattr(model, model_name)

# build model
model = MODEL_CLASS(dataset, config['Model'], device)

# model load!
print("Model Restored!")
model.restore(logger.log_dir)
model.logger = logger

# train
try:   
    print("Train Continue...")
    valid_score, train_time = train_model(model, dataset, evaluator, early_stop, logger, config)
except (KeyboardInterrupt, SystemExit):
    valid_score, train_time = dict(), 0
    logger.info("학습을 중단하셨습니다.")

m, s = divmod(train_time, 60)
h, m = divmod(m, 60)
logger.info('\nTotal training time - %d:%d:%d(=%.1f sec)' % (h, m, s, train_time))

# test
model.eval()
model.restore(logger.log_dir)
model.logger = logger
print("Model Restored!")
test_score = dict()
for testset in dataset.testsets:
    test_score.update(evaluator.evaluate(model, dataset, testset))

# show result
evaluation_table = ResultTable(table_name='Best Result', header=list(test_score.keys()))
evaluation_table.add_row('Test', test_score)

# evaluation_table.show()
logger.info(evaluation_table.to_string())


logger.info("Saved to %s" % (log_dir))
