#!/bin/bash

set -euo pipefail

output_dir=bin
all_methods=(fista-int-iter fista-frac-iter fista-int flow-iter flow)

usage() {
    cat <<EOF
Usage: $0 [all|METHOD ...]

Methods:
  ${all_methods[*]}

Examples:
  $0
  $0 all
  $0 flow-iter
  $0 fista-int flow
EOF
}

is_method() {
    local method=$1
    local known

    for known in "${all_methods[@]}"; do
        [[ "${method}" == "${known}" ]] && return 0
    done
    return 1
}

compile_method() {
    local algo=$1

    if [[ "${algo}" == fista* ]]; then
        local fista_dir="src/${algo}"
        local fista_exec="${algo}"

        echo "Compiling ${fista_exec}"
        g++ -O3 -unroll-loops -fopenmp -std=c++17 "${fista_dir}/${fista_exec}.cpp" -o "${output_dir}/${fista_exec}"
        chmod +x "${output_dir}/${fista_exec}"
    elif [[ "${algo}" == flow* ]]; then
        local flow_dir="src/${algo}"
        local flow_exec="${algo}"

        echo "Compiling ${flow_exec}"
        make -C "${flow_dir}" clean
        make -C "${flow_dir}"
        chmod +x "${flow_dir}/${flow_exec}"
        mv "${flow_dir}/${flow_exec}" "${output_dir}/${flow_exec}"
    else
        echo "Unknown method: ${algo}" >&2
        exit 1
    fi
}

requested=("$@")
if [[ ${#requested[@]} -eq 0 ]]; then
    requested=(all)
fi

selected=()
for arg in "${requested[@]}"; do
    case "${arg}" in
        -h|--help)
            usage
            exit 0
            ;;
        all)
            selected=("${all_methods[@]}")
            break
            ;;
        *)
            if ! is_method "${arg}"; then
                echo "Unknown method: ${arg}" >&2
                usage >&2
                exit 1
            fi
            selected+=("${arg}")
            ;;
    esac
done

mkdir -p "${output_dir}"

for algo in "${selected[@]}"; do
    compile_method "${algo}"
done
