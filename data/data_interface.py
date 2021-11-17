import inspect
import importlib
import torch
import pickle as pkl
import pytorch_lightning as pl
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


class DInterface(pl.LightningDataModule):

    def __init__(self, num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.load_data_module()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(train=True)
            self.valset = self.instancialize(train=False)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(train=False)

        # # If you need to balance your data using Pytorch Sampler,
        # # please uncomment the following lines.
        # with open('./data/ref/samples_weight.pkl', 'rb') as f:
        #     self.sample_weight = pkl.load(f)

    # def train_dataloader(self):
    #     sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset)*20)
    #     return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, sampler = sampler)

    def collate_fn(self, batch):
        post = list(map(lambda x: x['post'], batch))
        resp = list(map(lambda x: x['resp'], batch))
        input_ids = list(map(lambda x: x['input_ids'], batch))
        ref = list(map(lambda x: x['ref'], batch))

        post = rnn_utils.pad_sequence(post, padding_value=0, batch_first=True)
        resp = rnn_utils.pad_sequence(resp, padding_value=0, batch_first=True)
        input_ids = rnn_utils.pad_sequence(input_ids, padding_value=0, batch_first=True)

        return {'post': post, 'resp': resp, 'input_ids': input_ids, 'ref': torch.stack(ref, dim=0)}

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_workers, shuffle=False)

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
