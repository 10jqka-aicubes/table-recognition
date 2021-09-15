#!/usr/bin/bash/

basepath=$(cd $(dirname $0); pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath

python metrics.py \
     --result_path=$PREDICT_RESULT_FILE_DIR \
     --gt_path=$GROUNDTRUTH_FILE_DIR/gt_answer \
     --output_path=$RESULT_JSON_FILE
