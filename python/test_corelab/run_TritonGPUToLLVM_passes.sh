#!/bin/bash
triton_opt=/home/chan/triton/python/build/cmake.linux-x86_64-cpython-3.10/bin/triton-opt
llvm_opt=/home/chan/.triton/llvm/llvm-2fe947b4-ubuntu-x64/bin/opt
working_dir=/home/chan/triton/liquid-kernel/compiled
# 첫 번째 인자로 전달된 파일명 (절대 경로 처리 가능)
ttgir_file="$1"

# 입력 파일이 존재하는지 확인
if [ -z "$ttgir_file" ]; then
    echo "❌ Error: No input file provided!"
    exit 1
fi

if [ ! -f "$ttgir_file" ]; then
    echo "❌ Error: Input file '$ttgir_file' does not exist!"
    exit 1
fi

# 파일명 추출 (확장자 제거)
filename=$(basename -- "$ttgir_file")  # 예: matmul.ttgir.mlir
filename_noext="${filename%.*}"        # 예: matmul.ttgir


passes=(
#    "decompose-unsupported-nvidia-conversions"
   "tritongpu-combine-tensor-select-and-if"
   "convert-scf-to-cf"
   "convert-index-to-llvm"
   "allocate-shared-memory"
   "tritongpu-global-scratch-memory-allocation"
   "convert-triton-gpu-to-llvm{compute-capability=86 ptx-version=84}"
   "convert-nv-gpu-to-llvm"
   "convert-arith-to-llvm"
   "canonicalize"
   "cse"
   "symbol-dce"
)
# 초기 입력 파일 설정
input_filename="$ttgir_file"
counter=0

for pass in "${passes[@]}"
do
    echo "===> Applying Pass: $pass"

    # Pass 이름에서 `{}` 옵션 제거
    pass_name="${pass//\{*/}"

    # 출력 파일명 설정
    output_filename="$working_dir/${filename_noext}.${counter}.${pass_name}.llir.mlir"

    echo "Output File: $output_filename"

    # Triton Pass 실행
    $triton_opt --pass-pipeline="builtin.module($pass)" "$input_filename" -o "$output_filename"

    if [ $? -ne 0 ]; then
        echo "❌ $pass_name failed."
        echo "Command: $triton_opt --pass-pipeline=\"builtin.module($pass)\" \"$input_filename\" -o \"$output_filename\""
        exit 1
    fi

    # 다음 Pass의 입력 파일 업데이트
    input_filename="$output_filename"
    ((counter++))
done