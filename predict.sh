#!/bin/bash

python predict.py \
--model_file '0' \
--pretrain_vector_file '0' \
--predict_file "test_no_label.txt" \
--with_label 0 \
--infer_batch_size 1024 \
