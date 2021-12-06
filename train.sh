python main.py  --lr_scheduler warmup \
                --optimizer AdamW  \
                --batch_size 16 \
                --accumulate_grad_batches 16 \
                --precision 16 \
                --model_name SGNet \
                --word_embeddings ./pretrained/bert-base-chinese \
                --pretrained \
                --pretrained_generator_path ./pretrained/cdial/min_ppl_model \
                --pretrained_selector_path ./pretrained/snet/best-epoch=18-val_loss=0.286.ckpt
                # --model_path ./checkpoints/lightning_logs/version_48/checkpoints/last.ckpt \ 
                # --load_v_num 48 \
                # --load_best 
