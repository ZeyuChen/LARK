export CUDA_VISIBLE_DEVICES=0

BERT_BASE_PATH="uncased_L-12_H-768_A-12"
TASK_NAME='MRPC'
DATA_PATH=./glue_data/MRPC/
CKPT_PATH=./output/MRPC/checkpoints/

python -u run_classifier.py --task_name ${TASK_NAME} \
                   --use_cuda true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 8192 \
                   --in_tokens true \
                   --init_pretraining_params ${BERT_BASE_PATH}/params \
                   --data_dir ${DATA_PATH} \
                   --vocab_path ${BERT_BASE_PATH}/vocab.txt \
                   --checkpoints ${CKPT_PATH} \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 25 \
                   --epoch 3 \
                   --max_seq_len 512 \
                   --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
                   --learning_rate 1e-4 \
                   --skip_steps 10 \
                   --random_seed 1
