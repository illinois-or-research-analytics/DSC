#!/bin/sh

output_dir=bin
mkdir -p ${output_dir}

for algo in fista-int fista-frac flow flow-iter
do  
    if [[ ${algo} == fista* ]]; then
        fista_dir="src/${algo}"
        fista_exec="${algo}"
        echo "Compiling ${fista_exec}"
        g++ -O3 -unroll-loops -fopenmp -std=c++17 -DPARALLEL=1 ${fista_dir}/${fista_exec}.cpp -o ${output_dir}/${fista_exec}
        chmod +x ${output_dir}/${fista_exec}
    elif [[ ${algo} == flow* ]]; then
        flow_dir="src/${algo}"
        flow_exec="${algo}"
        echo "Compiling ${flow_exec}"
        cd ${flow_dir}
        make clean
        make
        cd ../..
        chmod +x ${flow_dir}/${flow_exec}
        mv ${flow_dir}/${flow_exec} ${output_dir}/${flow_exec}
    else
        echo "Unknown algo: ${algo}"
        exit 1
    fi
done