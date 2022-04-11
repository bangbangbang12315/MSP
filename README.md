# MSP
Source code of Our NAACL 2022 Paper: 

Less is More: Learning to Refine Dialogue History for Personalized Dialogue Generation.

# Preinstallation
First, install the python packages in your **Python3** environment:
```
  git clone MSP
  cd MSP
  pip install -r requirements.txt
```

Then, you should download the pre-trained models to initialize the model training. We provide two word embeddings in the Google Drive:
- [Bert](https://huggingface.co/bert-base-chinese)
- [DialoGPT](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)

You can pre-train your own language model, and use it.

After downloading, you should put the pretrained file to the path ```word_embeddings``` and  ```pretrained_path```.



# Data Processing

Firstly, You should construct the user refiner and topic refiner with the pretrained model and extract the current user and similar users' reference.

Then, you should provide the dialogue history of users for training the model. One line data format should as:

 ```post '#' + 'resp' + '#' + ref```

You can refer to ```data/dialo_dataset.py``` for more details about the data processing.

If you are interested in the dataset [PChatbot](https://github.com/qhjqhj00/SIGIR2021-Pchatbot), please go to its official repository for more details. 

# Model Training

We provide a shell script ```train.sh``` to start model training. You should modify the ```word_embeddings``` and ```pretrained_path``` to your own paths. Then, you can start training by the following command: 
```
bash scripts/train_chat.sh
```

The hyper-parameters are defined and set in the ```main.py```.

After training, the trained checkpoints are saved in ```checkpoints```. 


# Evaluating

For calculating varities of evaluation metrics(e.g. BLEU, P-Cover...), we provide a shell script ```scripts/eval.sh```. The inferenced result and metrics result are saved in ```evaluate```. You can evaluate your model by the following command: 
```
bash evaluate.sh
```

# Citations

If our code helps you, please cite our work by:
```

```

# Links
- [Pytorch](https://pytorch.org)
- [PyTorch-lightning](https://www.pytorchlightning.ai/)
- [Transformers](https://huggingface.co/)
- [PChatbot Dataset](https://github.com/qhjqhj00/SIGIR2021-Pchatbot)




