# --- Settings ---
IS_RUN_WCC=0

# The consensus wrapper script
CONSENSUS_SCRIPT="src/run_pamcon.py"

# List of algorithms to include in the consensus
input_algos=(
    "flow-iter"
    "leiden-mod"
    "RTRex"
    "ikc-5"
    "leiden-cpm-0.01+wcc"
)

# Base Output Directory
DSC_BASE="data/dsc"

# --- Arguments ---
network_id=$1
generator="ec-sbm"
clustering="sbm+wcc"
start=0
end=0

# --- Helper Functions ---
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# ==========================================
# ID Construction
# ==========================================
# Joins input algos with '-x-'
# e.g., "flow-iter-x-leiden-mod-x-RTRex-x-ikc-5"
joined_algos=$(printf "%s-x-" "${input_algos[@]}")
joined_algos=${joined_algos%-x-} # Remove trailing -x-

# Construct the Base ID
# e.g., "flow-iter-x-leiden-mod--pamcon"
base_id="${joined_algos}--pamcon"

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

# ==========================================
# Main Loop
# ==========================================
for seed in $(seq ${start} ${end})
do
    echo "============================"
    echo "[$(timestamp)] Starting Consensus: ${network_id} ${base_id} Seed:${seed}"

    # --- Input Paths ---
    # Ground Truth directory (contains edge.csv and com.csv)
    gt_dir="data/synthetic_networks/${generator}/${clustering}/${network_id}/${seed}/"
    inp_edge="${gt_dir}/edge.csv"
    inp_gt="${gt_dir}/com.csv"

    if [ ! -f "${inp_edge}" ]; then
        echo "[$(timestamp)] Input edge file ${inp_edge} missing. Skipping seed ${seed}."
        continue
    fi

    # Root directory where individual algorithm results are stored
    clustering_root="${DSC_BASE}/clusterings/${generator}/${clustering}/${network_id}/${seed}/"
    
    # --- Collect Input Clusterings ---
    cluster_files=()
    missing_input=false
    for alg in "${input_algos[@]}"; do
        cfile="${clustering_root}/${alg}/com.csv"
        if [ -f "$cfile" ]; then
            cluster_files+=("$cfile")
        else
            echo "[$(timestamp)] Warning: Missing input clustering for ${alg} at ${cfile}"
            missing_input=true
        fi
    done

    if [ "$missing_input" = true ] && [ ${#cluster_files[@]} -eq 0 ]; then
        echo "[$(timestamp)] No valid input clusterings found. Skipping."
        continue
    fi

    # ==========================================
    # 1. Run Pamcon Consensus
    # ==========================================
    # Output Directory: .../seed/algo1-x-algo2--pamcon/
    out_dir="${DSC_BASE}/clusterings/${generator}/${clustering}/${network_id}/${seed}/${base_id}/"
    stats_dir="${DSC_BASE}/stats/${generator}/${clustering}/${network_id}/${seed}/${base_id}/"
    acc_dir="${DSC_BASE}/acc/${generator}/${clustering}/${network_id}/${seed}/${base_id}/"

    mkdir -p "${out_dir}"

    # We now expect the Python script to output 'com.csv' directly
    pamcon_result="${out_dir}/com.csv"

    if [ ! -f "${out_dir}/done" ]; then
        echo "[$(timestamp)] Running Consensus Script..."
        
        { timeout 3d /usr/bin/time -v python "${CONSENSUS_SCRIPT}" \
            --output-dir "${out_dir}" \
            --graph "${inp_edge}" \
            --clusters "${cluster_files[@]}" \
            --out-prefix "com" \
            --stage-prefix "input_cluster"; } 1> "${out_dir}/output.log" 2> "${out_dir}/error.log"

        if [ -f "${pamcon_result}" ]; then
            touch "${out_dir}/done"
            echo "[$(timestamp)] Consensus completed."
        else
            echo "[$(timestamp)] Consensus failed. File ${pamcon_result} not found. Check ${out_dir}/error.log"
            continue
        fi
    else
        echo "[$(timestamp)] Consensus already done."
    fi

    # Run Eval on Raw Consensus
    if [ -f "${pamcon_result}" ]; then
        run_stats "${inp_edge}" "${pamcon_result}" "${stats_dir}"
        run_accuracy "${inp_edge}" "${inp_gt}" "${pamcon_result}" "${acc_dir}"
    fi

    # ==========================================
    # 2. Run WCC (Optional)
    # ==========================================
    if [ "${IS_RUN_WCC}" -eq 1 ]; then
        suffix="${base_id}+wcc"
        
        out_wcc_dir="${DSC_BASE}/clusterings/${generator}/${clustering}/${network_id}/${seed}/${suffix}/"
        stats_wcc_dir="${DSC_BASE}/stats/${generator}/${clustering}/${network_id}/${seed}/${suffix}/"
        acc_wcc_dir="${DSC_BASE}/acc/${generator}/${clustering}/${network_id}/${seed}/${suffix}/"
        
        mkdir -p "${out_wcc_dir}"
        if [ ! -f "${out_wcc_dir}/done" ]; then
            echo "[$(timestamp)] Running WCC..."
            # WCC Input: pamcon_result (com.csv)
            # WCC Output: com.csv in wcc dir
            { timeout 3d /usr/bin/time -v ./constrained-clustering/constrained_clustering \
                MincutOnly \
                --connectedness-criterion "1log_10(n)" \
                --edgelist "${inp_edge}" \
                --existing-clustering "${pamcon_result}" \
                --num-processors 1 \
                --output-file "${out_wcc_dir}/com.csv" \
                --log-file "${out_wcc_dir}/wcc.log" \
                --log-level 1; } 2> "${out_wcc_dir}/error.log"
            
            if [ -f "${out_wcc_dir}/com.csv" ]; then
                touch "${out_wcc_dir}/done"
            fi
        fi

        if [ -f "${out_wcc_dir}/com.csv" ]; then
            run_stats "${inp_edge}" "${out_wcc_dir}/com.csv" "${stats_wcc_dir}"
            run_accuracy "${inp_edge}" "${inp_gt}" "${out_wcc_dir}/com.csv" "${acc_wcc_dir}"
        fi
    fi
done

echo "[$(timestamp)] Finished: ${network_id} pamcon" >> log.txt
