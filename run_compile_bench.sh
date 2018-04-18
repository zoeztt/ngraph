#!/usr/bin/env bash

# Parameters
NBENCH="/home/yixing/repo/ngraph/build/src/tools/nbench/nbench"
MODEL="/home/yixing/repo/ngraph-models/models/tensorflow/resnet50_I1k/tf_function_cluster_8.v1836.json"
BACKEND=CPU
ITERATION=10

# Make
make -j -C /home/yixing/repo/ngraph/build

# Run
for FRONTEND_OPT_LEVEL in 0 1 2 3
do
    for BACKEND_OPT_LEVEL in 0 1 2 3
    do
        for REPEAT in 0
        do
            echo "###############################"
            echo "[FRONTEND_OPT_LEVEL=${FRONTEND_OPT_LEVEL} BACKEND_OPT_LEVEL=${BACKEND_OPT_LEVEL} REPEAT=${REPEAT}]"
            FRONTEND_OPT_LEVEL=${FRONTEND_OPT_LEVEL} BACKEND_OPT_LEVEL=${BACKEND_OPT_LEVEL} \
                ${NBENCH} -f ${MODEL} -b ${BACKEND} -i ${ITERATION}
        done
    done
done
