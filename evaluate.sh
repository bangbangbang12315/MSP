DEVICES=3
INFERENCE_PATH="./evaluate/ref/inference.txt"
MODEL_TEST_PATH='./ref/test/test.pkl'
TEST_ANS_PATH="./evaluate/ref/infer/test_ans.txt"
MODEL_PATH='checkpoints/lightning_logs/version_5/checkpoints/last.ckpt'
DIR_PATH='./evaluate/ref'
LOG_PATH='./ref/interact.log'
python inference.py \
            --device $DEVICES \
            --inference_path $INFERENCE_PATH \
            --test_path $MODEL_TEST_PATH \
            --log_path $LOG_PATH \
            --model_path $MODEL_PATH \
            --no_cuda

python ./evaluate/perplexity.py \
            --test $TEST_ANS_PATH \
            --infer $INFERENCE_PATH \
            --dir_path $DIR_PATH \
            --split_type 1