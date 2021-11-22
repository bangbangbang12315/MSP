python main.py  --lr_scheduler warmup \
                --optimizer AdamW  \
                --batch_size 32 \
                --model_name SGNet \
                --pretrained \
                --pretrained_generator_path ./pretrained/cdial/min_ppl_model \
                --pretrained_selector_path ./pretrained/snet/best-epoch=18-val_loss=0.286.ckpt \
                --word_embeddings ./pretrained/bert-base-chinese
                # --model_path ./checkpoints/lightning_logs/version_24/checkpoints/best-epoch=04-val_loss=2.567.ckpt