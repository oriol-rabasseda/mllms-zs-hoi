ANN_FILE=$1
IMG_DIR=$2
MODE=$3 # Options are OD, GT and MLLM
NUM_TEMPLATES=$4

RAW_JSON="./annotations/hico_test.json"
ENSEMBLE_JSON="./annotations/Ensemble/$NUM_TEMPLATES/hico_qa_test_unknown.json"
KO_JSON="./annotations/Ensemble/$NUM_TEMPLATES/hico_qa_test_ko.json"
HOI_INFO="./hoi_info.json"

OD_JSON="./annotations/Ensemble/$NUM_TEMPLATES/hico_qa_test_unknown_od.json"
GT_JSON="./annotations/Ensemble/$NUM_TEMPLATES/hico_qa_test_unknown_KO.json"
MLLM_JSON="./annotations/Ensemble/$NUM_TEMPLATES/hico_qa_test_unknown_mllm.json"

if [ ! -e $RAW_JSON ]; then
    python hico.py --hico-ann-filepath $ANN_FILE --output-filepath $RAW_JSON
fi

if [ ! -e $ENSEMBLE_JSON ]; then
    python hico_q_a_ensemble.py --hico-ann-filepath $RAW_JSON --hoi-info-filepath $HOI_INFO --output-qa-filepath $ENSEMBLE_JSON --mode "Unknown" --num-templates $NUM_TEMPLATES
fi

if [ $MODE = "OD" ]; then
    if [ ! -e $OD_JSON ]; then
        python hico_questions_from_objects_od.py --img-dir $IMG_DIR --qa-filepath $ENSEMBLE_JSON --output-filepath $OD_JSON
    fi
elif [ $MODE = "GT" ]; then
    if [ ! -e $KO_JSON ]; then
        python hico_q_a_ensemble.py --hico-ann-filepath $RAW_JSON --hoi-info-filepath $HOI_INFO --output-qa-filepath $KO_JSON --mode "KnownObjects" --num-templates $NUM_TEMPLATES
    fi

    if [ ! -e $GT_JSON ]; then
        python hico_questions_from_objects_gt.py --qa-filepath $ENSEMBLE_JSON --aux-qa-filepath $KO_JSON --output-filepath $GT_JSON
    fi
elif [ $MODE = "MLLM" ]; then
    MLLM_TYPE=$5
    if [ ! -e $MLLM_JSON ]; then
        python hico_questions_from_objects_mllm.py --model-name $MLLM_TYPE --qa-filepath $ENSEMBLE_JSON --img-dir $IMG_DIR --output-filepath $MLLM_JSON
    fi
fi