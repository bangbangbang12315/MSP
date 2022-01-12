python main.py  --lr_scheduler warmup \
                --optimizer AdamW  \
                --batch_size 8 \
                --accumulate_grad_batches 16 \
                --precision 16 \
                --model_name MSP \
                --gpus 3 \
                --word_embeddings ./pretrained/bert-base-chinese \
                --pretrained \
                --pretrained_generator_path ./pretrained/cdial/min_ppl_model \
                --pretrained_selector_path ./pretrained/snet/best-epoch=18-val_loss=0.286.ckpt
                # --load_v_num 81 \
                # --load_best 
                # --model_path ./checkpoints/lightning_logs/version_48/checkpoints/last.ckpt \ 

