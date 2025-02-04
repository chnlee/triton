def print_ir(compiled_kernel):
    with open(f"/home/chan/triton/liquid-kernel/compiled/{compiled_kernel.name}.ttir.mlir", "w", encoding="utf-8") as f:
        f.write(compiled_kernel.asm.get("ttir", "No TTIR available"))
    with open(f"/home/chan/triton/liquid-kernel/compiled/{compiled_kernel.name}.ttgir.mlir", "w", encoding="utf-8") as f:
        f.write(compiled_kernel.asm.get("ttgir", "No TTGIR available"))
    with open(f"/home/chan/triton/liquid-kernel/compiled/{compiled_kernel.name}.ll", "w", encoding="utf-8") as f:
        f.write(compiled_kernel.asm.get("llir", "No LLIR available"))
    with open(f"/home/chan/triton/liquid-kernel/compiled/{compiled_kernel.name}.ptx", "w", encoding="utf-8") as f:
        f.write(compiled_kernel.asm.get("ptx", "No PTX available"))