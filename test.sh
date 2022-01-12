python main.py  --lr_scheduler warmup \
                --optimizer AdamW  \
                --batch_size 1 \
                --model_name SNet \
                --load_best \
                --load_v_num 87 \
                --is_test \
                --word_embeddings ./pretrained/bert-base-chinese
                # --pretrained \
                # --pretrained_generator_path ./pretrained/cdial/min_ppl_model \
                # --pretrained_selector_path ./pretrained/snet/best-epoch=17-val_loss=2.438.ckpt 