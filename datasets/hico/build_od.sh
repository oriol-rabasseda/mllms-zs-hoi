ANN_FILE=$1
IMG_DIR=$2
MODE=$3 # Options are OD, GT and MLLM

RAW_JSON="./annotations/hico_test.json"
QA_JSON="./annotations/hico_qa_test_unknown.json"
KO_JSON="./annotations/hico_qa_test_ko.json"

OD_JSON="./annotations/hico_qa_test_unknown_od.json"
GT_JSON="./annotations/hico_qa_test_unknown_od.json"
MLLM_JSON="./annotations/hico_qa_test_unknown_mllm.json"

if [ ! -e $RAW_JSON ]; then
    python hico.py --hico-ann-filepath $ANN_FILE --output-filepath $RAW_JSON
fi

if [ ! -e $QA_JSON ]; then
    python hico_q_a.py --hico-ann-filepath $RAW_JSON --output-qa-filepath $QA_JSON --mode "Unknown"
fi

if [ $MODE = "OD" ]; then
    if [ ! -e $OD_JSON ]; then
        python hico_questions_from_objects_od.py --img-dir $IMG_DIR --qa-filepath $QA_JSON --output-filepath $OD_JSON
    fi
elif [ $MODE = "GT" ]; then
    if [ ! -e $KO_JSON ]; then
        python hico_q_a.py --hico-ann-filepath $RAW_JSON --output-qa-filepath $KO_JSON --mode "KnownObjects"
    fi

    if [ ! -e $GT_JSON ]; then
        python hico_questions_from_objects_gt.py --qa-filepath $QA_JSON --aux-qa-filepath $KO_JSON --output-filepath $GT_JSON
    fi
elif [ $MODE = "MLLM" ]; then
    MLLM_TYPE=$4
    if [ ! -e $MLLM_JSON ]; then
        python hico_questions_from_objects_mllm.py --model-name $MLLM_TYPE --qa-filepath $QA_JSON --img-dir $IMG_DIR --output-filepath $MLLM_JSON
    fi
fi