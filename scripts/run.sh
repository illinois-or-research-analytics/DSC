#!/bin/bash

algo=$1
generator=ec-sbm
clustering=$2
network_id=$3

# Base Output Directory
DSC_BASE="data/dsc"

# Flags
IS_RUN_CC=1
IS_RUN_WCC=0
IS_RUN_CM=0

# Disable CC/WCC for specific algos
case ${algo} in
    leiden*|flow*|fista*|ikc*|RTRex) IS_RUN_CC=0 ;;
esac

# ==========================================
# Function: Run Statistics
# ==========================================
run_stats() {
    local edge_file=$1
    local com_file=$2
    local stats_dir=$3

    echo "Computing stats..."
    if [ ! -f "${stats_dir}/done" ]; then
        mkdir -p "${stats_dir}"
        { /usr/bin/time -v python network_evaluation/network_stats/compute_cluster_stats.py \
            --network "${edge_file}" \
            --community "${com_file}" \
            --outdir "${stats_dir}"; } 2> "${stats_dir}/error.log"
    else
        echo "Stats already done."
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

    echo "Computing accuracy..."
    if [ ! -f "${acc_dir}/done" ]; then
        mkdir -p "${acc_dir}"
        { /usr/bin/time -v python network_evaluation/commdet_acc/compute_cd_accuracy.py \
            --input-network "${edge_file}" \
            --gt-clustering "${gt_file}" \
            --est-clustering "${est_file}" \
            --output-prefix "${acc_dir}/result"; } 2> "${acc_dir}/error.log"
    else
        echo "Accuracy already done."
    fi
}

start=0
end=0

for seed in $(seq ${start} ${end})
do
    echo "============================"
    echo "${algo} ${generator} ${clustering} ${network_id} ${seed}"

    inp_dir="data/synthetic_networks/${generator}/${clustering}/${network_id}/${seed}/"
    inp_edge="${inp_dir}/edge.csv"
    inp_gt="${inp_dir}/com.csv"
    
    if [ ! -f "${inp_edge}" ]; then
        echo "Input file ${inp_edge} does not exist. Skipping."
        continue
    fi

    base_root_clusterings="${DSC_BASE}/clusterings/${generator}/${clustering}/${network_id}/${seed}"
    base_root_stats="${DSC_BASE}/stats/${generator}/${clustering}/${network_id}/${seed}"
    base_root_acc="${DSC_BASE}/acc/${generator}/${clustering}/${network_id}/${seed}"

    leiden_model=""
    leiden_res=""
    ikc_k=""
    sbm_model=""
    
    if [[ ${algo} == leiden* ]]; then
        leiden_model=$(echo ${algo} | cut -d'-' -f2)
        if [[ ${leiden_model} == cpm ]]; then
            leiden_res=$(echo ${algo} | cut -d'-' -f3)
        fi
    elif [[ ${algo} == ikc* ]]; then
        ikc_k=$(echo ${algo} | cut -d'-' -f2)
    elif [[ ${algo} == sbm* ]]; then
        sbm_model=$(echo ${algo} | cut -d'-' -f2)
    fi

    # ==========================================
    # 1. Run Base Clustering
    # ==========================================
    suffix="${algo}"
    out_dir="${base_root_clusterings}/${suffix}/"
    stats_dir="${base_root_stats}/${suffix}/"
    acc_dir="${base_root_acc}/${suffix}/"
    
    base_com="${out_dir}/com.csv"
    base_dens="${out_dir}/density.csv"
    base_done="${out_dir}/done"

    echo "Running clustering..."
    mkdir -p "${out_dir}"

    if [ ! -f "${base_done}" ]; then
        if [[ ${algo} == fista* ]]; then
            [ ! -f bin/${algo} ] && echo "Executable ${algo} missing" && exit 1
            { timeout 3d /usr/bin/time -v ./bin/${algo} 200 "${inp_edge}" "${base_com}" "${base_dens}"; } 1> "${out_dir}/run.log" 2> "${out_dir}/error.log"
        elif [[ ${algo} == flow-iter-* ]]; then
            flow_threshold=$(echo ${algo} | cut -d'-' -f3)
            [ ! -f bin/flow-iter ] && echo "Executable flow-iter missing" && exit 1
            { timeout 3d /usr/bin/time -v ./bin/flow-iter "${inp_edge}" "${base_com}" "${base_dens}" "${flow_threshold}"; } 1> "${out_dir}/run.log" 2> "${out_dir}/error.log"
        elif [[ ${algo} == flow-iter ]]; then
            [ ! -f bin/flow-iter ] && echo "Executable flow-iter missing" && exit 1
            { timeout 3d /usr/bin/time -v ./bin/flow-iter "${inp_edge}" "${base_com}" "${base_dens}"; } 1> "${out_dir}/run.log" 2> "${out_dir}/error.log"
        elif [[ ${algo} == flow ]]; then
            [ ! -f bin/flow ] && echo "Executable flow missing" && exit 1
            { timeout 3d /usr/bin/time -v ./bin/flow "${inp_edge}" "${base_com}" "${base_dens}"; } 1> "${out_dir}/run.log" 2> "${out_dir}/error.log"
        elif [[ ${algo} == leiden* ]]; then
            if [[ ${leiden_model} == cpm ]]; then
                { timeout 3d /usr/bin/time -v python src/leiden/run_leiden.py --edgelist "${inp_edge}" --output-directory "${out_dir}" --model cpm --resolution "${leiden_res}"; } 2> "${out_dir}/error.log"
            elif [[ ${leiden_model} == mod ]]; then
                { timeout 3d /usr/bin/time -v python src/leiden/run_leiden.py --edgelist "${inp_edge}" --output-directory "${out_dir}" --model mod; } 2> "${out_dir}/error.log"
            else
                echo "Unknown leiden_model: ${leiden_model}"; continue
            fi
        
        elif [[ ${algo} == infomap ]]; then
            { timeout 3d /usr/bin/time -v python src/infomap/run_infomap.py --edgelist "${inp_edge}" --output-directory "${out_dir}"; } 1> "${out_dir}/output.log" 2> "${out_dir}/error.log"
        
        elif [[ ${algo} == ikc* ]]; then
            { timeout 3d /usr/bin/time -v python src/ikc/run_ikc.py --edgelist "${inp_edge}" --output-directory "${out_dir}" --kvalue "${ikc_k}"; } 1> "${out_dir}/output.log" 2> "${out_dir}/error.log"
        
        elif [[ ${algo} == sbm* ]]; then
            { timeout 3d /usr/bin/time -v python src/sbm/run_sbm.py --edgelist "${inp_edge}" --output-directory "${out_dir}" --method "${sbm_model}"; } 1> "${out_dir}/output.log" 2> "${out_dir}/error.log"
        
        elif [[ ${algo} == RTRex ]]; then
            { timeout 3d /usr/bin/time -v python src/RTRex/run_RTRex.py --edgelist "${inp_edge}" --output-directory "${out_dir}"; } 1> "${out_dir}/output.log" 2> "${out_dir}/error.log"
        
        else
            echo "Unknown method: ${algo}"; continue
        fi

        if [ -f "${base_com}" ]; then 
            touch "${base_done}"
        fi
    fi

    if [ ! -f "${base_com}" ]; then
        echo "CRITICAL: Base clustering failed or timed out."
        continue
    fi

    run_stats "${inp_edge}" "${base_com}" "${stats_dir}"
    run_accuracy "${inp_edge}" "${inp_gt}" "${base_com}" "${acc_dir}"

    # ==========================================
    # 2. Run CC
    # ==========================================
    if [ "${IS_RUN_CC}" -eq 1 ]; then
        suffix="${algo}+cc"
        out_cc_dir="${base_root_clusterings}/${suffix}/"
        stats_cc_dir="${base_root_stats}/${suffix}/"
        acc_cc_dir="${base_root_acc}/${suffix}/"
        cc_com="${out_cc_dir}/com.csv"
        cc_done="${out_cc_dir}/done"
        
        mkdir -p "${out_cc_dir}"
        if [ ! -f "${cc_done}" ]; then
            { timeout 3d /usr/bin/time -v ./constrained-clustering/constrained_clustering \
                MincutOnly \
                --edgelist "${inp_edge}" \
                --existing-clustering "${base_com}" \
                --num-processors 1 \
                --output-file "${cc_com}" \
                --log-file "${out_cc_dir}/cc.log" \
                --log-level 1 \
                --connectedness-criterion 0; } 2> "${out_cc_dir}/error.log"
            [ -f "${cc_com}" ] && touch "${cc_done}"
        fi
        
        if [ -f "${cc_com}" ]; then
            run_stats "${inp_edge}" "${cc_com}" "${stats_cc_dir}"
            run_accuracy "${inp_edge}" "${inp_gt}" "${cc_com}" "${acc_cc_dir}"
        fi
    fi

    # ==========================================
    # 3. Run WCC 
    # ==========================================
    if [ "${IS_RUN_WCC}" -eq 1 ]; then
        suffix="${algo}+wcc"
        out_wcc_dir="${base_root_clusterings}/${suffix}/"
        stats_wcc_dir="${base_root_stats}/${suffix}/"
        acc_wcc_dir="${base_root_acc}/${suffix}/"
        wcc_com="${out_wcc_dir}/com.csv"
        wcc_done="${out_wcc_dir}/done"
        
        mkdir -p "${out_wcc_dir}"
        if [ ! -f "${wcc_done}" ]; then
            { timeout 3d /usr/bin/time -v ./constrained-clustering/constrained_clustering \
                MincutOnly \
                --connectedness-criterion "1log_10(n)" \
                --edgelist "${inp_edge}" \
                --existing-clustering "${base_com}" \
                --num-processors 1 \
                --output-file "${wcc_com}" \
                --log-file "${out_wcc_dir}/wcc.log" \
                --log-level 1; } 2> "${out_wcc_dir}/error.log"
            [ -f "${wcc_com}" ] && touch "${wcc_done}"
        fi

        if [ -f "${wcc_com}" ]; then
            run_stats "${inp_edge}" "${wcc_com}" "${stats_wcc_dir}"
            run_accuracy "${inp_edge}" "${inp_gt}" "${wcc_com}" "${acc_wcc_dir}"
        fi
    fi

    # ==========================================
    # 4. Run CM
    # ==========================================
    if [ "${IS_RUN_CM}" -eq 1 ]; then
        suffix="${algo}+cm"
        out_cm_dir="${base_root_clusterings}/${suffix}/"
        stats_cm_dir="${base_root_stats}/${suffix}/"
        acc_cm_dir="${base_root_acc}/${suffix}/"
        cm_com="${out_cm_dir}/com.csv"
        cm_done="${out_cm_dir}/done"
        
        if [ ! -f "${cm_done}" ]; then
            if [[ ${algo} == leiden* ]]; then
                if [[ ${leiden_model} == cpm ]]; then
                    mkdir -p "${out_cm_dir}"
                    { timeout 3d /usr/bin/time -v python cm_pipeline/scripts/run_cm.py --no-prune \
                        --input "${inp_edge}" --existing-clustering "${base_com}" --working-directory "${out_cm_dir}" \
                        --output "${cm_com}" --threshold 1log10 --clusterer leiden --resolution "${leiden_res}"; } 1> "${out_cm_dir}/output.log" 2> "${out_cm_dir}/error.log"
                elif [[ ${leiden_model} == mod ]]; then
                    mkdir -p "${out_cm_dir}"
                    { timeout 3d /usr/bin/time -v python cm_pipeline/scripts/run_cm.py --no-prune \
                        --input "${inp_edge}" --existing-clustering "${base_com}" --working-directory "${out_cm_dir}" \
                        --output "${cm_com}" --threshold 1log10 --clusterer leiden_mod; } 1> "${out_cm_dir}/output.log" 2> "${out_cm_dir}/error.log"
                fi
            elif [[ ${algo} == infomap ]]; then
                mkdir -p "${out_cm_dir}"
                { timeout 3d /usr/bin/time -v python cm_pipeline/scripts/run_cm.py --no-prune \
                    --input "${inp_edge}" --existing-clustering "${base_com}" --working-directory "${out_cm_dir}" \
                    --output "${cm_com}" --threshold 1log10 --clusterer external \
                    --clusterer_args infomap_cm_cargs.json --clusterer_file cm_pipeline/hm01/clusterers/external_clusterers/infomap_wrapper.py; } 1> "${out_cm_dir}/output.log" 2> "${out_cm_dir}/error.log"
            elif [[ ${algo} == ikc* ]]; then
                mkdir -p "${out_cm_dir}"
                { timeout 3d /usr/bin/time -v python cm_pipeline/scripts/run_cm.py --no-prune \
                    --input "${inp_edge}" --existing-clustering "${base_com}" --working-directory "${out_cm_dir}" \
                    --output "${cm_com}" --threshold 1log10 --clusterer ikc --k "${ikc_k}"; } 1> "${out_cm_dir}/output.log" 2> "${out_cm_dir}/error.log"
            elif [[ ${algo} == flow-iter ]]; then
                mkdir -p "${out_cm_dir}"
                { timeout 3d /usr/bin/time -v python cm_pipeline/scripts/run_cm.py --no-prune \
                --input "${inp_edge}" --existing-clustering "${base_com}" --working-directory "${out_cm_dir}" \
                --output "${cm_com}" --threshold 1log10 --clusterer external \
                --clusterer_args dsc_cm_cargs.json --clusterer_file cm_pipeline/hm01/clusterers/external_clusterers/dsc_wrapper.py; } 1> "${out_cm_dir}/output.log" 2> "${out_cm_dir}/error.log"
            else
                echo "CM not implemented for ${algo}"
            fi
            [ -f "${cm_com}" ] && touch "${cm_done}"
        fi

        if [ -f "${cm_com}" ]; then
            run_stats "${inp_edge}" "${cm_com}" "${stats_cm_dir}"
            run_accuracy "${inp_edge}" "${inp_gt}" "${cm_com}" "${acc_cm_dir}"
        fi
    fi

    echo "${algo} ${generator} ${clustering} ${network_id} ${seed}" >> complete.log
done