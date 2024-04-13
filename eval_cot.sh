python run_eval.py \
--dataset hotpotcot \
--model $CHECKPOINT_PATH \
--parallel seq \
--n-shots 6 \
--subsample-test-set 500 \
--n-runs 3 \
--output-dir output \
--wbits 16 \
--generation \