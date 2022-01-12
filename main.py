from collections import defaultdict
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from data import DInterface
from model import MInterface, SInterface
from utils import load_model_path_by_args


def load_callbacks():
    callbacks = []
    # callbacks.append(plc.EarlyStopping(
    #     monitor='val_acc',
    #     mode='max',
    #     patience=10,
    #     min_delta=0.001
    # ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_loss',
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=5,
        mode='min',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))
    
    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface(**vars(args))
        args.resume_from_checkpoint = load_path
        # model = MInterface.load_from_checkpoint(checkpoint_path=load_path)

    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)
    args.callbacks = load_callbacks()
    trainer = Trainer.from_argparse_args(args)
    # args.logger = logger
    if args.is_test:    
        trainer.test(model, data_module)
    else:
        trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--gpus', default='1', type=str, required=False, help="设置使用哪些显卡，用逗号分割")
    parser.add_argument('--seed', default=1104, type=int)
    parser.add_argument('--min_epochs', default=5, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--val_check_interval', default=100000, type=int)
    parser.add_argument('--default_root_dir', default='checkpoints', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--distributed_backend', default='dp', type=str)
    parser.add_argument('--fast_dev_run', action='store_true')
    parser.add_argument('--is_test', action='store_true')
    parser.add_argument('--find_unused_parameters', action='store_true')
    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine','warmup'], type=str)
    parser.add_argument('--optimizer', choices=['Adam', 'AdamW'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)
    parser.add_argument('--warm_up_steps', default=4000, type=int, required=False, help="warm up步数")
    parser.add_argument('--accumulate_grad_batches', default=1, type=int, required=False)
    parser.add_argument('--precision', default=32, type=int, required=False)
    parser.add_argument('--gradient_clip_val', default=0.5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default='checkpoints', type=str)
    parser.add_argument('--load_ver', default='version', type=str)
    parser.add_argument('--load_v_num', default=3, type=int)

    # Training Info
    parser.add_argument('--dataset', default='dialo_dataset', type=str)
    parser.add_argument('--vocab_path', default='pretrained/gpt2-chinese-cluecorpussmall/vocab.txt', type=str)
    parser.add_argument('--train_data_dir', default='ref/Selected_Weibo/train.txt', type=str)
    parser.add_argument('--valid_data_dir', default='ref/Selected_Weibo/dev.txt', type=str)
    parser.add_argument('--test_data_dir', default='ref/Selected_Weibo/test.txt', type=str)
    parser.add_argument('--model_name', default='MSP', type=str)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained_generator_path', default=None, type=str)
    parser.add_argument('--pretrained_selector_path', default=None, type=str)
    parser.add_argument('--word_embeddings', default=None, type=str)

    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    
    # Model Hyperparameters
    parser.add_argument('--generator_config', default='pretrained/gpt2-chinese-cluecorpussmall/config.json', type=str)
    parser.add_argument('--max_length', default=512, type=int)

    args = parser.parse_args()

    main(args)
