#!/bin/sh

edgelist=$1
out_root=$2

if [ ! -f ${edgelist} ]; then
    echo "Error: Edgelist file ${edgelist} not found!"
    exit 1
fi

mkdir -p ${out_root}

out_leiden_mod=${out_root}/leiden-mod/
mkdir -p ${out_leiden_mod}
{ /usr/bin/time -v python src/leiden/run_leiden.py \
    --edgelist ${edgelist} \
    --output-directory ${out_leiden_mod} \
    --model mod; } 2> ${out_leiden_mod}/error.log

if [ ! -f ${out_leiden_mod}/com.tsv ]; then
    echo "Error: Leiden-Mod did not produce a community file at ${out_leiden_mod}/com.tsv"
    exit 1
fi

out_flow=${out_root}/dsc-flow-iter/
mkdir -p ${out_flow}
{ /usr/bin/time -v ./bin/flow-iter \
    ${edgelist} \
    ${out_flow}/com.tsv \
    ${out_flow}/density.tsv; } 1> ${out_flow}/run.log 2> ${out_flow}/error.log

if [ ! -f ${out_flow}/com.tsv ]; then
    echo "Error: DSC-Flow-Iter did not produce a community file at ${out_flow}/com.tsv"
    exit 1
fi

out_merge=${out_root}/merged/
mkdir -p ${out_merge}

echo "${out_leiden_mod}/com.tsv" > ${out_merge}/clustering_list.txt
echo "${out_flow}/com.tsv" >> ${out_merge}/clustering_list.txt

{ /usr/bin/time -v ./ClusterMerger/cluster_merger \
    Weighted \
    --edgelist ${edgelist} \
    --clustering-list ${out_merge}/clustering_list.txt \
    --weighting-strategy 0 \
    --threshold 0.5 \
    --num-processors 1 \
    --output-file "" \
    --output-weighted-graph ${out_merge}/edge.tsv \
    --log-file ${out_merge}/run.log \
    --log-level 1; } 2> ${out_merge}/error.log

if [ ! -f ${out_merge}/edge.tsv ]; then
    echo "Error: Merger did not produce an edge file at ${out_merge}/edge.tsv"
    exit 1
fi

out_unweighted=${out_root}/unweighted/
mkdir -p ${out_unweighted}
{ /usr/bin/time -v python src/pipeline/unweight.py \
    --input-network ${out_merge}/edge.tsv \
    --output-network ${out_unweighted}/edge.tsv; } 1> ${out_unweighted}/run.log 2> ${out_unweighted}/error.log

if [ ! -f ${out_unweighted}/edge.tsv ]; then
    echo "Error: Unweighted did not produce an edgelist file at ${out_unweighted}/edge.tsv"
    exit 1
fi

out_final=${out_root}/final/
mkdir -p ${out_final}
{ /usr/bin/time -v python src/leiden/run_leiden.py \
    --edgelist ${out_unweighted}/edge.tsv \
    --output-directory ${out_final} \
    --model cpm \
    --resolution 0.01; } 2> ${out_final}/error.log

if [ ! -f ${out_final}/com.tsv ]; then
    echo "Error: Leiden-CPM(0.01) did not produce a community file at ${out_final}/com.tsv"
    exit 1
fi

out_wcc=${out_root}/wcc/
mkdir -p ${out_wcc}
{ /usr/bin/time -v ./constrained-clustering/constrained_clustering \
    MincutOnly \
    --edgelist ${out_unweighted}/edge.tsv \
    --existing-clustering ${out_final}/com.tsv \
    --num-processors 1 \
    --output-file ${out_wcc}/com.tsv \
    --log-file ${out_wcc}/wcc.log \
    --log-level 1 \
    --connectedness-criterion 1; } 2> ${out_wcc}/error.log

if [ ! -f ${out_wcc}/com.tsv ]; then
    echo "Error: WCC did not produce a community file at ${out_wcc}/com.tsv"
    exit 1
fi