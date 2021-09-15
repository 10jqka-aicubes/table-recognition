#!/usr/bin/bash/

basepath=$(cd $(dirname $0); pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath

python train.py \
     --data_dir=$TRAIN_FILE_DIR \
     --save_model_dir=$SAVE_MODEL_DIR
