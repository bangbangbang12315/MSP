python main.py  --lr_scheduler warmup \
                --optimizer AdamW  \
                --batch_size 32 \
                --model_name SGNet \
                --pretrained \
                --pretrained_generator_path ./pretrained/cdial/min_ppl_model \
                --pretrained_selector_path ./pretrained/snet/best-epoch=17-val_loss=2.438.ckpt 