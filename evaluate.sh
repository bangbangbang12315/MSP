DEVICES='0'
INFERENCE_PATH="./evaluate/Selected_Weibo/inference_text.txt"
MODEL_TEST_PATH='./ref/Selected_Weibo/test.txt'
TEST_ANS_PATH="./evaluate/Selected_Weibo/infer/test_ans.txt"
MODEL_PATH='checkpoints/lightning_logs/version_28/checkpoints/best-epoch=04-val_loss=2.320.ckpt'
DIR_PATH='./evaluate/Selected_Weibo'
LOG_PATH='./ref/Selected_Weibo/interact.log'
python inference.py \
            --gpus $DEVICES \
            --inference_path $INFERENCE_PATH \
            --test_data_dir $MODEL_TEST_PATH \
            --log_path $LOG_PATH \
            --model_path $MODEL_PATH

python ./evaluate/perplexity.py \
            --test $TEST_ANS_PATH \
            --infer $INFERENCE_PATH \
            --dir_path $DIR_PATH \
            --split_type 1