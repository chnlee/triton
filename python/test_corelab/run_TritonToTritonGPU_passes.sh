#!/bin/bash
triton_opt=/home/chan/triton/python/build/cmake.linux-x86_64-cpython-3.10/bin/triton-opt
working_dir=/home/chan/triton/liquid-kernel/compiled
# 첫 번째 인자로 입력된 MLIR 파일
ttir_file="$1"

if [ -z "$ttir_file" ]; then
    echo "❌ Error: No input file provided!" >&2
    exit 1
fi

if [ ! -f "$ttir_file" ]; then
    echo "❌ Error: Input file '$ttir_file' does not exist!" >&2
    exit 1
fi

# 파일명 및 확장자 제거
filename=$(basename -- "$ttir_file")  # 예: matmul.ttir.mlir
filename_noext="${filename%.*}"        # 예: matmul.ttir

passes=(
    "convert-triton-to-tritongpu{num-ctas=1 num-warps=4 target='cuda:86' threads-per-warp=32}"
    "tritongpu-coalesce"
    "tritongpu-F32DotTC"
    "triton-nvidia-gpu-plan-cta"
    "tritongpu-remove-layout-conversions"
    "tritongpu-optimize-thread-locality"
    "tritongpu-accelerate-matmul"
    "tritongpu-remove-layout-conversions"
    "tritongpu-optimize-dot-operands"
    "cse"
    "tritongpu-optimize-accumulator-init"
    "tritongpu-combine-tensor-select-and-if"
    "tritongpu-loop-scheduling"
    "tritongpu-pipeline"
    "tritongpu-prefetch"
    "tritongpu-optimize-dot-operands"
    "tritongpu-coalesce-async-copy"
    "tritongpu-remove-layout-conversions"
    "tritongpu-reduce-data-duplication"
    "tritongpu-reorder-instructions"
    "cse"
    "symbol-dce"
    "triton-nvidia-gpu-fence-insertion"
    "triton-nvidia-tma-lowering"
    "canonicalize"
)
# --convert-triton-to-tritongpu \
# --tritongpu-coalesce \
# --tritongpu-F32DotTC \
# --triton-nvidia-gpu-plan-cta \
# --tritongpu-remove-layout-conversions \
# --tritongpu-optimize-thread-locality \
# --tritongpu-accelerate-matmul \
# --tritongpu-remove-layout-conversions \
# --tritongpu-optimize-dot-operands \
# --tritongpu-optimize-accumulator-init \
# --tritongpu-combine-tensor-select-and-if \
# --tritongpu-loop-scheduling \
# --tritongpu-pipeline \
# --tritongpu-prefetch \
# --tritongpu-optimize-dot-operands \
# --tritongpu-coalesce-async-copy \
# --tritongpu-remove-layout-conversions \
# --tritongpu-reduce-data-duplication \
# --tritongpu-reorder-instructions \
# --cse \
# --symbol-dce \
# --triton-nvidia-gpu-fence-insertion \
# --triton-nvidia-tma-lowering \
# --canonicalize
        

# 초기 입력 파일 설정
input_filename="$ttir_file"
counter=0

for pass in "${passes[@]}"
do
    echo "===> Applying Pass: $pass"

    pass_name="${pass//\{*/}"  # `{}` 옵션 제거
    output_filename="$working_dir/${filename_noext}.${counter}.${pass_name}.ttgir.mlir"

    echo "Output File: $output_filename"

    # 특정 Pass에 디버깅 옵션 적용
    if [[ "$pass" == "tritongpu-loop-scheduling" ]]; then
        $triton_opt --debug-only="triton-loop-schedule" --mlir-print-ir-after=tritongpu-loop-scheduling --pass-pipeline="builtin.module($pass)" "$input_filename" -o "$output_filename"
    else
        $triton_opt --pass-pipeline="builtin.module($pass)" "$input_filename" -o "$output_filename"
    fi

    # 실행 실패 시 오류 출력 후 종료
    if [ $? -ne 0 ]; then
        echo "❌ $pass_name failed."
        echo "Command: $triton_opt --pass-pipeline=\"builtin.module($pass)\" \"$input_filename\" -o \"$output_filename\""
        exit 1
    fi

    # MLIR 파일이 생성되었는지 확인
    if [ ! -f "$output_filename" ]; then
        echo "❌ ERROR: Expected output file '$output_filename' was not created!" >&2
        exit 1
    fi

    # 다음 Pass의 입력 파일 업데이트
    input_filename="$output_filename"
    ((counter++))
done