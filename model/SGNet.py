from typing import Generator
import pytorch_lightning as pl
from .SMN import SMN
from .GPT2 import GPT2
class SGNet(pl.LightningModule):
    def __init__(self, generator_config=None) -> None:
        super(SGNet, self).__init__()
        self.generator = GPT2(generator_config)
        self.selector = SMN()

        

    
