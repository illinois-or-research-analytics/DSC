# --- Settings ---
IS_RUN_WCC=1
IS_RUN_CM=0

input_algos=(
    "flow-iter"
    "leiden-mod"
    # "RTRex"
    "ikc-5"
)

algo="leiden-cpm-0.01"

# Base Output Directory
DSC_BASE="data/dsc"

method="ec-sbm"
clustering="sbm+wcc"
start=0
end=0

network_id=$1
weighting_strategy=$2
threshold=$3
unweight_param=$4

# Convert unweight param (true/false string) to actual boolean for logic if needed,
# though string comparison is safer in bash.
if [[ "${unweight_param}" == "true" ]]; then
    unweight=true
else
    unweight=false
fi

if [ "${unweight}" = false ]; then
    IS_RUN_CM=0
fi

# --- Helper Functions ---
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# ==========================================
# Function: Run Statistics
# ==========================================
run_stats() {
    local edge_file=$1
    local com_file=$2
    local stats_dir=$3

    echo "[$(timestamp)] Computing stats..."
    if [ ! -f "${stats_dir}/done" ]; then
        mkdir -p "${stats_dir}"
        { /usr/bin/time -v python network_evaluation/network_stats/compute_cluster_stats.py \
            --network "${edge_file}" \
            --community "${com_file}" \
            --outdir "${stats_dir}"; } 2> "${stats_dir}/error.log"
    else
        echo "[$(timestamp)] Stats already done."
    fi
}

# ==========================================
# Function: Run Accuracy
# ==========================================
run_accuracy() {
    local edge_file=$1
    local gt_file=$2
    local est_file=$3
    local acc_dir=$4

    echo "[$(timestamp)] Computing accuracy..."
    if [ ! -f "${acc_dir}/done" ]; then
        mkdir -p "${acc_dir}"
        { /usr/bin/time -v python network_evaluation/commdet_acc/compute_cd_accuracy.py \
            --input-network "${edge_file}" \
            --gt-clustering "${gt_file}" \
            --est-clustering "${est_file}" \
            --output-prefix "${acc_dir}/result"; } 2> "${acc_dir}/error.log"
    else
        echo "[$(timestamp)] Accuracy already done."
    fi
}

# --- ID Construction (Dynamic) ---
# Joins input algos with '-x-'
# e.g., "flow-iter-x-leiden-mod"
joined_algos=$(printf "%s-x-" "${input_algos[@]}")
joined_algos=${joined_algos%-x-} # Remove trailing -x-

partial_merge_id="${joined_algos}--${weighting_strategy}-${threshold}"

if [ "${unweight}" = true ]; then
    partial_merge_id="${partial_merge_id}-U"
fi
merge_id="${partial_merge_id}--${algo}"

# ==========================================
# Main Loop
# ==========================================
for seed in $(seq ${start} ${end})
do
    echo "============================"
    echo "[$(timestamp)] Starting: ${network_id} ${method} ${clustering} ${merge_id} ${seed}"

    # --- Input Paths ---
    gt_dir="data/synthetic_networks/${method}/${clustering}/${network_id}/${seed}/"
    inp_edge="${gt_dir}/edge.csv"
    inp_gt="${gt_dir}/com.csv"

    if [ ! -f "${inp_edge}" ]; then
        echo "[$(timestamp)] Input file ${inp_edge} missing. Skipping."
        continue
    fi

    clustering_root="${DSC_BASE}/clusterings/${method}/${clustering}/${network_id}/${seed}/"
    merge_dir="${DSC_BASE}/merge/${method}/${clustering}/${network_id}/${seed}/${partial_merge_id}/"
    
    # ==========================================
    # 1. Run Merging (Dynamic Inputs)
    # ==========================================
    echo "[$(timestamp)] Checking/Running merging into: ${merge_dir}"
    mkdir -p "${merge_dir}"
    
    if [ ! -f "${merge_dir}/done" ]; then
        
        # Build the dynamic arguments array for python
        merge_args=()
        for alg in "${input_algos[@]}"; do
            merge_args+=(--input-clusterings "${clustering_root}/${alg}/com.csv")
        done

        # Run with dynamic args
        { timeout 3d /usr/bin/time -v python src/run_merger.py \
            --weighting-strategy ${weighting_strategy} \
            --threshold ${threshold} \
            --input-network "${inp_edge}" \
            "${merge_args[@]}" \
            --output-prefix "${merge_dir}/" \
            --num-processors 1; } 1> "${merge_dir}/output.log" 2> "${merge_dir}/error.log"

        [ -f "${merge_dir}/edge.csv" ] && touch "${merge_dir}/done"
    fi

    if [ ! -f "${merge_dir}/edge.csv" ]; then
        echo "[$(timestamp)] Merging failed or timed out. Skipping seed ${seed}."
        continue
    fi

    # ==========================================
    # 2. Run Unweighting
    # ==========================================
    edgelist="${merge_dir}/edge.csv"
    if [ "${unweight}" = true ]; then
        echo "[$(timestamp)] Checking/Running unweighting..."
        edgelist="${merge_dir}/edge-U.csv"
        if [ ! -f "${edgelist}" ]; then
            { timeout 3d /usr/bin/time -v python src/unweight.py \
                --input-network "${merge_dir}/edge.csv" \
                --output-network "${edgelist}"; } 1> "${merge_dir}/unweight_output.log" 2> "${merge_dir}/unweight_error.log"
        fi
        
        if [ ! -f "${edgelist}" ]; then
             echo "[$(timestamp)] Unweighting failed. Skipping seed ${seed}."
             continue
        fi
    fi

    # ==========================================
    # 3. Run Final Clustering (Single Algo)
    # ==========================================
    leiden_model=""
    leiden_res=""
    ikc_k=""
    
    if [[ ${algo} == leiden* ]]; then
        leiden_model=$(echo ${algo} | cut -d'-' -f2)
        if [[ ${leiden_model} == cpm ]]; then
            leiden_res=$(echo ${algo} | cut -d'-' -f3)
        fi
    elif [[ ${algo} == ikc* ]]; then
        ikc_k=$(echo ${algo} | cut -d'-' -f2)
    fi

    out_dir="${DSC_BASE}/clusterings/${method}/${clustering}/${network_id}/${seed}/${merge_id}/"
    if [ ! -f "${out_dir}/done" ]; then
        mkdir -p "${out_dir}"
        echo "[$(timestamp)] Running final clustering (${algo})..."
        if [[ ${algo} == leiden* ]]; then
            weighted_flag=""
            [ "${unweight}" = false ] && weighted_flag="--weighted"
            
            if [[ ${leiden_model} == cpm ]]; then
                { timeout 3d /usr/bin/time -v python src/leiden/run_leiden.py ${weighted_flag} \
                    --edgelist "${edgelist}" --output-directory "${out_dir}" \
                    --model cpm --resolution ${leiden_res}; } 2> "${out_dir}/error.log"
            elif [[ ${leiden_model} == mod ]]; then
                { timeout 3d /usr/bin/time -v python src/leiden/run_leiden.py ${weighted_flag} \
                    --edgelist "${edgelist}" --output-directory "${out_dir}" \
                    --model mod; } 2> "${out_dir}/error.log"
            fi
        fi
        [ -f "${out_dir}/com.csv" ] && touch "${out_dir}/done"
    fi

    if [ ! -f "${out_dir}/com.csv" ]; then
        echo "[$(timestamp)] Final clustering failed or timed out."
        continue
    fi

    # ==========================================
    # 4. Run WCC & Evaluation
    # ==========================================
    if [ "${IS_RUN_WCC}" -eq 1 ]; then
        suffix="${merge_id}+wcc"
        
        out_wcc_dir="${DSC_BASE}/clusterings/${method}/${clustering}/${network_id}/${seed}/${suffix}/"
        stats_wcc_dir="${DSC_BASE}/stats/${method}/${clustering}/${network_id}/${seed}/${suffix}/"
        acc_wcc_dir="${DSC_BASE}/acc/${method}/${clustering}/${network_id}/${seed}/${suffix}/"
        
        mkdir -p "${out_wcc_dir}"
        if [ ! -f "${out_wcc_dir}/done" ]; then
            echo "[$(timestamp)] Running WCC..."
            { timeout 3d /usr/bin/time -v ./constrained-clustering/constrained_clustering \
                MincutOnly \
                --connectedness-criterion "1log_10(n)" \
                --edgelist "${edgelist}" \
                --existing-clustering "${out_dir}/com.csv" \
                --num-processors 1 \
                --output-file "${out_wcc_dir}/com.csv" \
                --log-file "${out_wcc_dir}/wcc.log" \
                --log-level 1; } 2> "${out_wcc_dir}/error.log"
            [ -f "${out_wcc_dir}/com.csv" ] && touch "${out_wcc_dir}/done"
        fi

        if [ -f "${out_wcc_dir}/com.csv" ]; then
            run_stats "${inp_edge}" "${out_wcc_dir}/com.csv" "${stats_wcc_dir}"
            run_accuracy "${inp_edge}" "${inp_gt}" "${out_wcc_dir}/com.csv" "${acc_wcc_dir}"
        fi
    fi

    # ==========================================
    # 5. Run CM
    # ==========================================
    if [ "${IS_RUN_CM}" -eq 1 ]; then
        suffix="${merge_id}+cm"
        out_cm_dir="${DSC_BASE}/clusterings/${method}/${clustering}/${network_id}/${seed}/${suffix}/"
        stats_cm_dir="${DSC_BASE}/stats/${method}/${clustering}/${network_id}/${seed}/${suffix}/"
        acc_cm_dir="${DSC_BASE}/acc/${method}/${clustering}/${network_id}/${seed}/${suffix}/"
        cm_com="${out_cm_dir}/com.csv"
        cm_done="${out_cm_dir}/done"
        
        if [ ! -f "${cm_done}" ]; then
            echo "[$(timestamp)] Running CM..."
            mkdir -p "${out_cm_dir}"
            
            if [[ ${algo} == leiden* ]]; then
                if [[ ${leiden_model} == cpm ]]; then
                    { timeout 3d /usr/bin/time -v python cm_pipeline/scripts/run_cm.py --no-prune \
                        --input "${edgelist}" --existing-clustering "${out_dir}/com.csv" \
                        --working-directory "${out_cm_dir}" \
                        --output "${cm_com}" --threshold 1log10 --clusterer leiden \
                        --resolution "${leiden_res}"; } 1> "${out_cm_dir}/output.log" 2> "${out_cm_dir}/error.log"
                elif [[ ${leiden_model} == mod ]]; then
                    { timeout 3d /usr/bin/time -v python cm_pipeline/scripts/run_cm.py --no-prune \
                        --input "${edgelist}" --existing-clustering "${out_dir}/com.csv" \
                        --working-directory "${out_cm_dir}" \
                        --output "${cm_com}" --threshold 1log10 --clusterer leiden_mod; } 1> "${out_cm_dir}/output.log" 2> "${out_cm_dir}/error.log"
                fi
            elif [[ ${algo} == infomap ]]; then
                { timeout 3d /usr/bin/time -v python cm_pipeline/scripts/run_cm.py --no-prune \
                    --input "${edgelist}" --existing-clustering "${out_dir}/com.csv" \
                    --working-directory "${out_cm_dir}" \
                    --output "${cm_com}" --threshold 1log10 --clusterer external \
                    --clusterer_args infomap_cm_cargs.json \
                    --clusterer_file cm_pipeline/hm01/clusterers/external_clusterers/infomap_wrapper.py; } 1> "${out_cm_dir}/output.log" 2> "${out_cm_dir}/error.log"
            elif [[ ${algo} == ikc* ]]; then
                { timeout 3d /usr/bin/time -v python cm_pipeline/scripts/run_cm.py --no-prune \
                    --input "${edgelist}" --existing-clustering "${out_dir}/com.csv" \
                    --working-directory "${out_cm_dir}" \
                    --output "${cm_com}" --threshold 1log10 --clusterer ikc --k "${ikc_k}"; } 1> "${out_cm_dir}/output.log" 2> "${out_cm_dir}/error.log"
            elif [[ ${algo} == flow-iter ]]; then
                 { timeout 3d /usr/bin/time -v python cm_pipeline/scripts/run_cm.py --no-prune \
                    --input "${edgelist}" --existing-clustering "${out_dir}/com.csv" \
                    --working-directory "${out_cm_dir}" \
                    --output "${cm_com}" --threshold 1log10 --clusterer external \
                    --clusterer_args dsc_cm_cargs.json \
                    --clusterer_file cm_pipeline/hm01/clusterers/external_clusterers/dsc_wrapper.py; } 1> "${out_cm_dir}/output.log" 2> "${out_cm_dir}/error.log"
            fi
            
            [ -f "${cm_com}" ] && touch "${cm_done}"
        fi

        if [ -f "${cm_com}" ]; then
            run_stats "${inp_edge}" "${cm_com}" "${stats_cm_dir}"
            run_accuracy "${inp_edge}" "${inp_gt}" "${cm_com}" "${acc_cm_dir}"
        fi
    fi
done

echo "[$(timestamp)] Finished: ${network_id} ${weighting_strategy} ${threshold} ${unweight}" >> log.txt
