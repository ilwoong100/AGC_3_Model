import os
import sys
import argparse

import numpy as np
import torch

from collections import OrderedDict
from dataloader import Dataset
from evaluation import Evaluator
from utils import Config, Logger, ResultTable, make_log_dir, set_random_seed


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='saves/TM_Generation/10_20210611-1719', metavar='P')
parser.add_argument('--testsets', type=str, default=None)
parser.add_argument('--beam_size', type=int, default=None)
parser.add_argument('--gpu', type=int, default=None)
args = parser.parse_args()

# read configs
config = Config(main_conf_path=args.path, model_conf_path=args.path)
if args.testsets is not None:
    config.main_config['Dataset']['testsets'] = args.testsets[1:-1].split(',')
if args.gpu is not None:
    config.main_config['Experiment']['gpu'] = args.gpu
if args.beam_size is not None:
    config.model_config['Model']['beam_size'] = args.beam_size

gpu = config.get_param('Experiment', 'gpu')
gpu = str(gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = config.get_param('Experiment', 'model_name')

log_dir = args.path
logger = Logger(log_dir)

dataset = Dataset(model_name, **config['Dataset'])

# evaluator
evaluator = Evaluator(**config['Evaluator'])

import model

MODEL_CLASS = getattr(model, model_name)

# build model
model = MODEL_CLASS(dataset, config['Model'], device)

# test
model.eval()
model.restore(logger.log_dir)
model.logger = logger
print("Model Restored!")
test_score = dict()
for testset in dataset.testsets:
    test_score.update(evaluator.evaluate(model, dataset, testset))

if args.beam_size is not None:
    logger.info("beam_size %s" % (args.beam_size))


# show result
evaluation_table = ResultTable(table_name='Best Result', header=list(test_score.keys()))
evaluation_table.add_row('Test', test_score)

# evaluation_table.show()
logger.info(evaluation_table.to_string())


logger.info("Saved to %s" % (log_dir))
