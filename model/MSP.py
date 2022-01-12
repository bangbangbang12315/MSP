from typing import Generator
import pytorch_lightning as pl
# from .SMN import SMN
from .SNet import SNet
from .GPT2 import GPT2
class MSP(pl.LightningModule):
    def __init__(self, generator_config=None,word_embeddings=None) -> None:
        super(MSP, self).__init__()
        print("Initing MSP...")
        self.generator = GPT2(generator_config)
        self.selector = SNet(word_embeddings)