DEVICES=0
INFERENCE_PATH="./evaluate/Selected_Weibo/inference.txt"
MODEL_TEST_PATH='./ref/Selected_Weibo/test.txt'
TEST_ANS_PATH="./evaluate/Selected_Weibo/infer/test_ans.txt"
MODEL_PATH='checkpoints/lightning_logs/version_15/checkpoints/last.ckpt'
DIR_PATH='./evaluate/Selected_Weibo'
LOG_PATH='./ref/Selected_Weibo/interact.log'
python inference.py \
            --device $DEVICES \
            --inference_path $INFERENCE_PATH \
            --test_data_dir $MODEL_TEST_PATH \
            --log_path $LOG_PATH \
            --model_path $MODEL_PATH 

python ./evaluate/perplexity.py \
            --test $TEST_ANS_PATH \
            --infer $INFERENCE_PATH \
            --dir_path $DIR_PATH \
            --split_type 1