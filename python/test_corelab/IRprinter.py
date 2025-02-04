import subprocess
import os

SCRIPT_PATH_TG2LL = "/home/chan/triton/python/test_corelab/run_TritonGPUToLLVM_passes.sh"
SCRIPT_PATH_T2TG = "/home/chan/triton/python/test_corelab/run_TritonToTritonGPU_passes.sh"

def run_TritonGPUToLLVM_passes(ttgir_file):
    """
    Bash 스크립트를 실행하여 Triton IR을 변환하는 함수
    """
    if not os.path.exists(SCRIPT_PATH_TG2LL):
        raise FileNotFoundError(f"❌ Script not found: {SCRIPT_PATH_TG2LL}")
    if not os.path.exists(ttgir_file):
        raise FileNotFoundError(f"❌ Input file not found: {ttgir_file}")
    
    print(f"🚀 Running Triton Passes on {ttgir_file}")
    
    # 결과 로그를 저장
    log_file = "/home/chan/triton/python/test_corelab/TritonGPUToLLVM_passes.log"
    
    try:
        with open(log_file, "w") as log:
            result = subprocess.run([SCRIPT_PATH_TG2LL, ttgir_file], check=True, stdout=log, stderr=log, text=True)
        print(f"✅ TritonGPU to LLVM Passes completed. Check log: {log_file}")
    except subprocess.CalledProcessError as e:
        print("❌ Error running script. Check log for details:", log_file)

def run_TritonToTritonGPU_passes(ttgir_file):
    """
    Bash 스크립트를 실행하여 Triton IR을 TritonGPU IR로 변환하는 함수
    """
    if not os.path.exists(SCRIPT_PATH_T2TG):
        raise FileNotFoundError(f"❌ Script not found: {SCRIPT_PATH_T2TG}")
    if not os.path.exists(ttgir_file):
        raise FileNotFoundError(f"❌ Input file not found: {ttgir_file}")
    
    print(f"🚀 Running TritonToTritonGPU Passes on {ttgir_file}")
    
    # 결과 로그 저장
    log_file = "/home/chan/triton/python/test_corelab/TritonToTritonGPU_passes.log"
    
    try:
        with open(log_file, "w") as log:
            result = subprocess.run([SCRIPT_PATH_T2TG, ttgir_file], check=True, stdout=log, stderr=log, text=True)
        print(f"✅ TritonToTritonGPU Passes completed. Check log: {log_file}")
    except subprocess.CalledProcessError as e:
        print("❌ Error running script. Check log for details:", log_file)

def print_ir(compiled_kernel):
    with open(f"/home/chan/triton/liquid-kernel/compiled/{compiled_kernel.name}.ttir.mlir", "w", encoding="utf-8") as f:
        f.write(compiled_kernel.asm.get("ttir", "No TTIR available"))
    with open(f"/home/chan/triton/liquid-kernel/compiled/{compiled_kernel.name}.ttgir.mlir", "w", encoding="utf-8") as f:
        f.write(compiled_kernel.asm.get("ttgir", "No TTGIR available"))
    with open(f"/home/chan/triton/liquid-kernel/compiled/{compiled_kernel.name}.ll", "w", encoding="utf-8") as f:
        f.write(compiled_kernel.asm.get("llir", "No LLIR available"))
    with open(f"/home/chan/triton/liquid-kernel/compiled/{compiled_kernel.name}.ptx", "w", encoding="utf-8") as f:
        f.write(compiled_kernel.asm.get("ptx", "No PTX available"))