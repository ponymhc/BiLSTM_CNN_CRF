#!/bin/bash

python  run.py \
--model_name "BilstmCnnCrf" \
--clean_data 1 \
--train_file "train.txt" \
--val_file "val.txt" \
--test_file "test.txt" \
--case 1 \
--min_word_freq 10 \
--min_char_freq 10 \
--pretrain_vector 1 \
--reduce_vector 1 \
--pretrain_vector_file "un_cbow_20230824" \
--max_sentence_length 64 \
--max_word_length 20 \
--train_batch_size  512 \
--val_batch_size 2048 \
--test_batch_size 2048 \
--word_embedding_dim 300 \
--char_embedding_dim 20 \
--num_filters 30 \
--filter_size 3 \
--hidden_dim 256 \
--num_layers 2 \
--drop_out 0.5 \
--learning_rate 0.01 \
--lr_decay_step 5000 \
--lr_decay_gamma 0.7 \
--epochs 1 \
--early_stop 50000 \

