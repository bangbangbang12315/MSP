python main.py  --lr_scheduler warmup \
                --optimizer AdamW  \
                --pretrained \
                --batch_size 64 \
                --model_name SNet
                # --pretrained_selector_path ./pretrained/snet/best-epoch=56-val_loss=1.787.ckpt \