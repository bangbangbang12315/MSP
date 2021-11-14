python main.py  --lr_scheduler warmup \
                --optimizer AdamW  \
                --pretrained \
                --batch_size 32 \
                --model_name SGNet \
                --pretrained_selector_path ./pretrained/snet/best-epoch=03-val_loss=2.497.ckpt \
                --pretrained_generator_path ./pretrained/cdial/min_ppl_model