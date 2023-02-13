import os
import argparse

import torch

import model
from dataloader import Dataset
from evaluation import Evaluator_ensemble
from utils import Config, Logger, ResultTable


parser = argparse.ArgumentParser()
parser.add_argument('--parant_path', type=str, default='saves/ensemble_221102_2200', metavar='P')
parser.add_argument('--path', type=str, default='[1step_seed4_base_bt,1step_seed1_base_bt,1step_seed2_base_bt,1step_seed3_base_bt,1step_seed5_base_bt]', metavar='P')
parser.add_argument('--testsets', type=str, default='[test90,test10,test_time]') # [test90,test10,test_time,test_KAERI]
parser.add_argument('--ensemble_type', type=str, default='max_voting')
parser.add_argument('--beam_size', type=int, default=None)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# set gpu
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# model 불러오기
model_paths = args.path[1:-1].split(',')
num_models = len(model_paths)
model_list = []

for model_path in model_paths:
    model_path = os.path.join(args.parant_path, model_path)
    # read configs
    config = Config(main_conf_path=model_path, model_conf_path=model_path)

    if args.testsets is not None:
        config.main_config['Dataset']['testsets'] = args.testsets[1:-1].split(',')
    if args.gpu is not None:
        config.main_config['Experiment']['gpu'] = args.gpu
    if args.beam_size is not None:
        config.model_config['Model']['beam_size'] = args.beam_size

    model_name = config.get_param('Experiment', 'model_name')

    dataset = Dataset(model_name, **config['Dataset'])
    log_dir = model_path
    logger = Logger(log_dir)

    # model 합치기
    MODEL_CLASS = getattr(model, model_name)

    # build model
    model_ = MODEL_CLASS(dataset, config['Model'], device)
    model_.eval()
    model_.restore(logger.log_dir)
    model_.logger = logger

    model_list.append(model_)
    print(f"Model {model_path} loaded")

# test
evaluator = Evaluator_ensemble(**config['Evaluator'])

test_score = dict()
for testset in dataset.testsets:
    test_score.update(evaluator.evaluate(model_list, dataset, testset, args.ensemble_type))

if args.beam_size is not None:
    logger.info("beam_size %s" % (args.beam_size))

# show result
evaluation_table = ResultTable(table_name='Best Result', header=list(test_score.keys()))
evaluation_table.add_row('Test', test_score)

# evaluation_table.show()
logger.info(evaluation_table.to_string())


logger.info("Saved to %s" % (log_dir))
