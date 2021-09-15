#!/usr/bin/bash/

basepath=$(cd $(dirname $0); pwd)
cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath

python predict.py \
     --input=$PREDICT_FILE_DIR \
     --output=$PREDICT_RESULT_FILE_DIR \
     --model_path=$SAVE_MODEL_DIR'/checkpoints/CP_epoch50_score_0.7072926266440029.pth' \
     --text_model_path=../../../model/weights/craft_mlt_25k.pth
