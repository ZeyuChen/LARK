export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=3

BERT_1="cased_L-12_H-768_A-12"
BERT_2="cased_L-24_H-1024_A-16"
BERT_3="uncased_L-12_H-768_A-12"
BERT_4="uncased_L-24_H-1024_A-16"
BERT_5="chinese_L-12_H-768_A-12"
BERT_6="multi_cased_L-12_H-768_A-12"

for BERT_MODEL_NAME in $BERT_1 $BERT_2 $BERT_3 $BERT_4 $BERT_5 $BERT_6; do
  echo $BERT_MODEL_NAME
  BERT_MODEL_DIR="./bert_models/$BERT_MODEL_NAME"
  BERT_MODULE_DIR="./bert_modules/bert_$BERT_MODEL_NAME"
  
  TASK_NAME='chnsenticorp'
  DATA_PATH=chnsenticorp_data
  CKPT_PATH=chn_checkpoints
  
  python -u create_module.py --task_name ${TASK_NAME} \
                     --use_cuda true \
                     --do_train true \
                     --do_val true \
                     --do_test true \
                     --batch_size 32 \
                     --in_tokens false \
                     --bert_model_dir ${BERT_MODEL_DIR} \
                     --module_dir ${BERT_MODULE_DIR} \
                     --init_pretraining_params ${BERT_MODEL_DIR}/params \
                     --data_dir ${DATA_PATH} \
                     --vocab_path ${BERT_MODEL_DIR}/vocab.txt \
                     --checkpoints ${CKPT_PATH} \
                     --save_steps 100 \
                     --weight_decay  0.01 \
                     --warmup_proportion 0.0 \
                     --validation_steps 50 \
                     --epoch 3 \
                     --max_seq_len 512 \
                     --bert_config_path ${BERT_MODEL_DIR}/bert_config.json \
                     --learning_rate 5e-5 \
                     --skip_steps 10

done
