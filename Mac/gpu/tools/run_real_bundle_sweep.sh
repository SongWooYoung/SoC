#!/bin/zsh

set -euo pipefail

script_dir=${0:A:h}
gpu_dir=${script_dir:h}

manifest_path=${SOC_QWEN3_MANIFEST:-${gpu_dir:h:h}/models/cpp/qwen3-0.6b/manifest.json}
integration_bin=${SOC_GPU_INTEGRATION_BIN:-${gpu_dir}/build/bin/test_real_bundle_regression}
summary_path=${SOC_GPU_SWEEP_SUMMARY:-${gpu_dir}/build/reports/test_real_bundle_sweep_summary.md}
report_dir=${SOC_GPU_SWEEP_REPORT_DIR:-${gpu_dir}/build/reports/sweep_cases}
max_new_tokens=${SOC_QWEN_MAX_NEW_TOKENS:-8}

if [[ ! -f "${manifest_path}" ]]; then
    echo "missing manifest: ${manifest_path}" >&2
    exit 1
fi

if [[ ! -x "${integration_bin}" ]]; then
    echo "missing integration binary: ${integration_bin}" >&2
    exit 1
fi

mkdir -p "${report_dir}"
mkdir -p "${summary_path:h}"
rm -f -- "${report_dir}"/*.md(N)

case_names=(
    short
    medium
    long
)

case_prompts=(
    'Summarize CPU and GPU scheduling in one short paragraph.'
    'Summarize how CPU scheduling, GPU command-buffer execution, and memory bandwidth trade off during transformer prefill and decode. Keep the answer concise but concrete.'
    'Summarize how CPU scheduling, GPU command-buffer execution, KV-cache reuse, temporary memory pressure, and prompt-length scaling interact during transformer prefill and decode. Contrast short prompts with longer prompts, explain why prefill and decode stress different parts of the runtime, and keep the explanation focused on practical runtime behavior rather than theory.'
)

extract_metric() {
    local report_path=$1
    local section_title=$2
    local metric_name=$3

    awk -F'|' -v section="## ${section_title}" -v metric="${metric_name}" '
        $0 == section { in_section = 1; next }
        in_section && /^## / { in_section = 0 }
        in_section && $0 ~ /^\|/ {
            name = $2
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
            if (name == metric) {
                value = $5
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
                print value
                exit
            }
        }
    ' "${report_path}"
}

extract_bullet_metric() {
    local report_path=$1
    local section_title=$2
    local prefix=$3

    awk -v section="## ${section_title}" -v prefix="- ${prefix}" '
        $0 == section { in_section = 1; next }
        in_section && /^## / { in_section = 0 }
        in_section && index($0, prefix) == 1 {
            value = substr($0, length(prefix) + 1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            print value
            exit
        }
    ' "${report_path}"
}

{
    echo '# GPU Real-Bundle Prompt Sweep'
    echo
    echo "- Manifest: ${manifest_path}"
    echo "- Integration binary: ${integration_bin}"
    echo "- Max new tokens: ${max_new_tokens}"
    echo "- Per-case reports: ${report_dir}"
    echo
    echo '| Case | Raw prompt tokens | Chat prompt tokens | Raw CPU:GPU wall ratio | Chat CPU:GPU wall ratio | Raw GPU active ratio | Chat GPU active ratio | Raw report | Chat report |'
    echo '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |'
} > "${summary_path}"

for index in {1..${#case_names}}; do
    case_name=${case_names[index]}
    prompt=${case_prompts[index]}
    report_path="${report_dir}/${case_name}.md"

    echo "running sweep case ${case_name}" >&2
    SOC_QWEN3_MANIFEST="${manifest_path}" \
    SOC_QWEN_PROMPT="${prompt}" \
    SOC_QWEN_MAX_NEW_TOKENS="${max_new_tokens}" \
    SOC_GPU_REPORT_PATH="${report_path}" \
    "${integration_bin}"

    raw_tokens=$(extract_bullet_metric "${report_path}" 'Raw Prompt' 'Prompt tokens:')
    chat_tokens=$(extract_bullet_metric "${report_path}" 'Chat Template Prompt' 'Prompt tokens:')
    raw_ratio=$(extract_bullet_metric "${report_path}" 'Raw Prompt' 'CPU:GPU context wall ratio =')
    chat_ratio=$(extract_bullet_metric "${report_path}" 'Chat Template Prompt' 'CPU:GPU context wall ratio =')
    raw_gpu_active=$(extract_metric "${report_path}" 'Raw Prompt' 'GPU active ratio')
    chat_gpu_active=$(extract_metric "${report_path}" 'Chat Template Prompt' 'GPU active ratio')

    {
        echo "| ${case_name} | ${raw_tokens} | ${chat_tokens} | ${raw_ratio} | ${chat_ratio} | ${raw_gpu_active} | ${chat_gpu_active} | [raw/chat](${report_path}) | [raw/chat](${report_path}) |"
    } >> "${summary_path}"
done

echo >> "${summary_path}"
echo "Generated summary at ${summary_path}"
echo "Generated ${#case_names} per-case reports under ${report_dir}"