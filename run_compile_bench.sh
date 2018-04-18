#!/usr/bin/env bash

# In `build` dir, run with
# ../run_compile_bench.sh | tee compile_bench_out.txt

# Parameters
NBENCH="/home/yixing/repo/ngraph/build/src/tools/nbench/nbench"
BACKEND=CPU
ITERATION=10

# MODEL="/home/yixing/repo/ngraph-models/models/tensorflow/resnet50_I1k/tf_function_cluster_8.v1836.json"
# MODEL="/home/yixing/repo/ngraph-models/models/tensorflow/mnist_mlp/tf_function_cluster_4.v61.json"
MODEL="/home/yixing/repo/ngraph/test/models/mxnet/LSTM_backward.json"

# Make
make -j -C /home/yixing/repo/ngraph/build

# Run
for REPEAT in 0 1 2 3 4 5 6 7 8 9
    do
    for FRONTEND_OPT_LEVEL in 0 1 2 3
    do
        for BACKEND_OPT_LEVEL in 0 1 2 3
        do
            echo ""
            echo "###############################"
            echo "[FRONTEND_OPT_LEVEL=${FRONTEND_OPT_LEVEL} BACKEND_OPT_LEVEL=${BACKEND_OPT_LEVEL} REPEAT=${REPEAT}]"
            FRONTEND_OPT_LEVEL=${FRONTEND_OPT_LEVEL} BACKEND_OPT_LEVEL=${BACKEND_OPT_LEVEL} \
                ${NBENCH} -f ${MODEL} -b ${BACKEND} -i ${ITERATION}
        done
    done
done
