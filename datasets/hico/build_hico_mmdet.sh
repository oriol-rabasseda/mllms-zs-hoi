ANN_FILE=$1
IMG_DIR=$2
MODEL=$3

RAW_JSON="./annotations/hico_test.json"
QA_JSON="./annotations/hico_qa_test_unknown.json"
OD_JSON="./annotations/hico_qa_test_unknown_$MODEL.json"

if [ ! -e $RAW_JSON ]; then
    python hico.py --hico-ann-filepath $ANN_FILE --output-filepath $RAW_JSON
fi

if [ ! -e $QA_JSON ]; then
    python hico_q_a.py --hico-ann-filepath $RAW_JSON --output-qa-filepath $QA_JSON --mode "Unknown"
fi

if [ ! -e $OD_JSON ]; then
    python hico_questions_from_objects_mmdet.py --img-dir $IMG_DIR --qa-filepath $QA_JSON --output-filepath $OD_JSON --model $MODEL
fi