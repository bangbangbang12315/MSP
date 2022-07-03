python main.py  --lr_scheduler warmup \
                --optimizer AdamW  \
                --batch_size 8 \
                --accumulate_grad_batches 16 \
                --precision 16 \
                --model_name MSP \
                --gpus 4 \
                --word_embeddings ./pretrained/bert-base-chinese \
