#!/bin/bash

edgelist=$1
out_root=$2

# -------------------------------------------------------------------
# Define the methods to run and merge here
# Supports: 
#   - dsc-flow-iter
#   - leiden-mod
#   - rtrex
#   - ikc-<kvalue>           (e.g., ikc-5, ikc-10)
#   - leiden-cpm-<resolution> (e.g., leiden-cpm-0.1, leiden-cpm-0.01)
# -------------------------------------------------------------------
METHODS=(
    "dsc-flow-iter"
    "leiden-mod"
    "rtrex"
    "ikc-5"
)

if [ ! -f ${edgelist} ]; then
    echo "Error: Edgelist file ${edgelist} not found!"
    exit 1
fi

mkdir -p ${out_root}

# Prepare the merge list file
out_merge=${out_root}/merged/
mkdir -p ${out_merge}
merge_list_file="${out_merge}/clustering_list.txt"
> "${merge_list_file}"

# ---------------------------------------------------------
# PIPELINE EXECUTION LOOP
# ---------------------------------------------------------
echo "Starting pipeline with methods: ${METHODS[*]}"

for method in "${METHODS[@]}"; do
    echo "Processing method: $method"
    
    # Defaults
    run_cmd=""
    check_file=""
    method_out_dir="${out_root}/${method}/"
    mkdir -p "${method_out_dir}"
    
    case "$method" in
        dsc-flow-iter)
            run_cmd="/usr/bin/time -v ./bin/flow-iter ${edgelist} ${method_out_dir}/com.csv ${method_out_dir}/density.csv"
            check_file="${method_out_dir}/com.csv"
            ;;
            
        leiden-mod)
            run_cmd="/usr/bin/time -v python src/leiden/run_leiden.py --edgelist ${edgelist} --output-directory ${method_out_dir} --model mod"
            check_file="${method_out_dir}/com.csv"
            ;;
            
        rtrex)
            run_cmd="/usr/bin/time -v python src/RTRex/run_RTRex.py --edgelist ${edgelist} --output-directory ${method_out_dir}"
            check_file="${method_out_dir}/com.csv"
            ;;
            
        ikc-*)
            # Extract k-value (assumes format ikc-<k>)
            k_val=$(echo "${method}" | cut -d'-' -f2)
            run_cmd="/usr/bin/time -v python src/ikc/run_ikc.py --edgelist ${edgelist} --output-directory ${method_out_dir} --kvalue ${k_val}"
            check_file="${method_out_dir}/com.csv"
            ;;
            
        leiden-cpm-*)
            # Extract resolution (assumes format leiden-cpm-<res>)
            res_val=$(echo "${method}" | cut -d'-' -f3)
            run_cmd="/usr/bin/time -v python src/leiden/run_leiden.py --edgelist ${edgelist} --output-directory ${method_out_dir} --model cpm --resolution ${res_val}"
            check_file="${method_out_dir}/com.csv"
            ;;
            
        *)
            echo "Warning: Unknown method '$method'. Skipping."
            continue
            ;;
    esac

    # Execute the command
    echo "Running: $method"
    eval "${run_cmd}" 1> "${method_out_dir}/output.log" 2> "${method_out_dir}/error.log"

    # Verify Output
    if [ ! -f "${check_file}" ]; then
        echo "Error: $method did not produce a community file at ${check_file}"
        exit 1
    fi

    # Append to clustering list for merger
    echo "${check_file}" >> "${merge_list_file}"
done

# ---------------------------------------------------------
# MERGER STEP
# ---------------------------------------------------------
echo "Running Merger..."

{ /usr/bin/time -v ./ClusterMerger/cluster_merger \
    Weighted \
    --edgelist ${edgelist} \
    --clustering-list ${merge_list_file} \
    --weighting-strategy 0 \
    --threshold 0.5 \
    --num-processors 1 \
    --output-file "" \
    --output-weighted-graph ${out_merge}/edge.csv \
    --log-file ${out_merge}/run.log \
    --log-level 1; } 1> ${out_merge}/output.log 2> ${out_merge}/error.log

if [ ! -f ${out_merge}/edge.csv ]; then
    echo "Error: Merger did not produce an edge file at ${out_merge}/edge.csv"
    exit 1
fi

# ---------------------------------------------------------
# FINAL CLUSTERING STEP AND POST-PROCESSING
# ---------------------------------------------------------
echo "Running Final Clustering and Post-processing..."

out_unweighted=${out_root}/unweighted/
mkdir -p ${out_unweighted}
{ /usr/bin/time -v python src/unweight.py \
    --input-network ${out_merge}/edge.csv \
    --output-network ${out_unweighted}/edge.csv; } 1> ${out_unweighted}/output.log 2> ${out_unweighted}/error.log

if [ ! -f ${out_unweighted}/edge.csv ]; then
    echo "Error: Unweighted did not produce an edgelist file at ${out_unweighted}/edge.csv"
    exit 1
fi

out_final=${out_root}/final/
mkdir -p ${out_final}
{ /usr/bin/time -v python src/leiden/run_leiden.py \
    --edgelist ${out_unweighted}/edge.csv \
    --output-directory ${out_final} \
    --model cpm \
    --resolution 0.01; } 1> ${out_final}/output.log 2> ${out_final}/error.log

if [ ! -f ${out_final}/com.csv ]; then
    echo "Error: Leiden-CPM(0.01) did not produce a community file at ${out_final}/com.csv"
    exit 1
fi

out_wcc=${out_root}/final+wcc/
mkdir -p ${out_wcc}
{ /usr/bin/time -v ./constrained-clustering/constrained_clustering \
    MincutOnly \
    --connectedness-criterion "1log_10(n)" \
    --edgelist ${out_unweighted}/edge.csv \
    --existing-clustering ${out_final}/com.csv \
    --num-processors 1 \
    --output-file ${out_wcc}/com.csv \
    --log-file ${out_wcc}/wcc.log \
    --log-level 1; } 1> ${out_wcc}/output.log 2> ${out_wcc}/error.log

if [ ! -f ${out_wcc}/com.csv ]; then
    echo "Error: WCC did not produce a community file at ${out_wcc}/com.csv"
    exit 1
fi

echo "Pipeline finished successfully."
