#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant dense<32> : tensor<128x32xi32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %5 : i32
    %11 = arith.remsi %10, %9 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.divsi %10, %9 : i32
    %14 = arith.muli %12, %c128_i32 : i32
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %18 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %19 = arith.addi %17, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %20 = arith.addi %18, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %21 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.remsi %19, %21 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %23 = arith.muli %13, %c64_i32 : i32
    %24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %25 = tt.splat %23 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %26 = arith.addi %25, %24 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %27 = tt.splat %arg4 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %28 = arith.remsi %26, %27 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %29 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %30 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked>
    %31 = arith.muli %29, %30 : tensor<128x1xi32, #blocked>
    %32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %34 = tt.broadcast %31 : tensor<128x1xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %35 = tt.broadcast %33 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %36 = arith.addi %34, %35 : tensor<128x32xi32, #blocked>
    %37 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blocked>
    %38 = tt.addptr %37, %36 : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi32, #blocked>
    %39 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %40 = tt.expand_dims %39 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %41 = tt.splat %arg7 : i32 -> tensor<32x1xi32, #blocked1>
    %42 = arith.muli %40, %41 : tensor<32x1xi32, #blocked1>
    %43 = tt.expand_dims %28 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %44 = tt.broadcast %42 : tensor<32x1xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
    %45 = tt.broadcast %43 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
    %46 = arith.addi %44, %45 : tensor<32x64xi32, #blocked1>
    %47 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked1>
    %48 = tt.addptr %47, %46 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi32, #blocked1>
    %49 = arith.addi %arg5, %c31_i32 : i32
    %50 = arith.divsi %49, %c32_i32 : i32
    %51 = arith.muli %arg7, %c32_i32 : i32
    %52 = tt.splat %51 : i32 -> tensor<32x64xi32, #blocked1>
    %53 = ttg.local_alloc  : () -> !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable>
    %54 = ttg.local_alloc  : () -> !ttg.memdesc<2x32x64xf32, #shared1, #smem, mutable>
    %55 = arith.cmpi sgt, %50, %c0_i32 : i32
    %56 = tt.splat %arg5 : i32 -> tensor<1x32xi32, #blocked>
    %57 = arith.cmpi slt, %33, %56 : tensor<1x32xi32, #blocked>
    %58 = tt.broadcast %57 : tensor<1x32xi1, #blocked> -> tensor<128x32xi1, #blocked>
    %59 = ttg.memdesc_subview %53[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable, 2x128x32>
    %60 = tt.splat %55 : i1 -> tensor<128x32xi1, #blocked>
    %61 = arith.andi %60, %58 : tensor<128x32xi1, #blocked>
    %62 = ttg.async_copy_global_to_local %38, %59 mask %61 other %cst_0 : tensor<128x32x!tt.ptr<f32>, #blocked> -> <128x32xf32, #shared, #smem, mutable, 2x128x32>
    %63 = ttg.async_commit_group %62
    %64 = tt.splat %arg5 : i32 -> tensor<32x1xi32, #blocked1>
    %65 = arith.cmpi slt, %40, %64 : tensor<32x1xi32, #blocked1>
    %66 = tt.broadcast %65 : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
    %67 = ttg.memdesc_subview %54[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 2x32x64>
    %68 = tt.splat %55 : i1 -> tensor<32x64xi1, #blocked1>
    %69 = arith.andi %68, %66 : tensor<32x64xi1, #blocked1>
    %70 = ttg.async_copy_global_to_local %48, %67 mask %69 other %cst_1 : tensor<32x64x!tt.ptr<f32>, #blocked1> -> <32x64xf32, #shared1, #smem, mutable, 2x32x64>
    %71 = ttg.async_commit_group %70
    %72 = arith.cmpi sgt, %50, %c1_i32 : i32
    %73 = tt.addptr %38, %cst : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi32, #blocked>
    %74 = tt.addptr %48, %52 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi32, #blocked1>
    %75 = arith.subi %arg5, %c32_i32 : i32
    %76 = tt.splat %75 : i32 -> tensor<1x32xi32, #blocked>
    %77 = arith.cmpi slt, %33, %76 : tensor<1x32xi32, #blocked>
    %78 = tt.broadcast %77 : tensor<1x32xi1, #blocked> -> tensor<128x32xi1, #blocked>
    %79 = ttg.memdesc_subview %53[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable, 2x128x32>
    %80 = tt.splat %72 : i1 -> tensor<128x32xi1, #blocked>
    %81 = arith.andi %80, %78 : tensor<128x32xi1, #blocked>
    %82 = ttg.async_copy_global_to_local %73, %79 mask %81 other %cst_0 : tensor<128x32x!tt.ptr<f32>, #blocked> -> <128x32xf32, #shared, #smem, mutable, 2x128x32>
    %83 = ttg.async_commit_group %82
    %84 = tt.splat %75 : i32 -> tensor<32x1xi32, #blocked1>
    %85 = arith.cmpi slt, %40, %84 : tensor<32x1xi32, #blocked1>
    %86 = tt.broadcast %85 : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
    %87 = ttg.memdesc_subview %54[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 2x32x64>
    %88 = tt.splat %72 : i1 -> tensor<32x64xi1, #blocked1>
    %89 = arith.andi %88, %86 : tensor<32x64xi1, #blocked1>
    %90 = ttg.async_copy_global_to_local %74, %87 mask %89 other %cst_1 : tensor<32x64x!tt.ptr<f32>, #blocked1> -> <32x64xf32, #shared1, #smem, mutable, 2x32x64>
    %91 = ttg.async_commit_group %90
    %92 = ttg.memdesc_subview %53[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable, 2x128x32>
    %93 = ttg.async_wait %71 {num = 2 : i32}
    %94 = ttg.memdesc_subview %54[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 2x32x64>
    %95:11 = scf.for %arg9 = %c0_i32 to %50 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %73, %arg12 = %74, %arg13 = %c1_i32, %arg14 = %c0_i32, %arg15 = %92, %arg16 = %93, %arg17 = %94, %arg18 = %93, %arg19 = %83, %arg20 = %91) -> (tensor<128x64xf32, #mma>, tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<32x64x!tt.ptr<f32>, #blocked1>, i32, i32, !ttg.memdesc<128x32xf32, #shared, #smem, mutable, 2x128x32>, !ttg.async.token, !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 2x32x64>, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %114 = arith.subi %50, %c2_i32 : i32
      %115 = arith.cmpi slt, %arg9, %114 : i32
      %116 = ttg.local_load %arg15 token %arg16 : !ttg.memdesc<128x32xf32, #shared, #smem, mutable, 2x128x32> -> tensor<128x32xf32, #blocked>
      %117 = ttg.local_load %arg17 token %arg18 : !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 2x32x64> -> tensor<32x64xf32, #blocked1>
      %118 = ttg.convert_layout %116 : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %119 = ttg.convert_layout %117 : tensor<32x64xf32, #blocked1> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %120 = tt.dot %118, %119, %arg10, inputPrecision = tf32 : tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<128x64xf32, #mma>
      %121 = tt.addptr %arg11, %cst : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi32, #blocked>
      %122 = tt.addptr %arg12, %52 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi32, #blocked1>
      %123 = arith.addi %arg13, %c1_i32 : i32
      %124 = arith.cmpi slt, %123, %c2_i32 : i32
      %125 = arith.select %124, %123, %c0_i32 : i32
      %126 = arith.addi %arg9, %c2_i32 : i32
      %127 = arith.muli %126, %c32_i32 : i32
      %128 = arith.subi %arg5, %127 : i32
      %129 = tt.splat %128 : i32 -> tensor<1x32xi32, #blocked>
      %130 = arith.cmpi slt, %33, %129 : tensor<1x32xi32, #blocked>
      %131 = tt.broadcast %130 : tensor<1x32xi1, #blocked> -> tensor<128x32xi1, #blocked>
      %132 = ttg.memdesc_subview %53[%125, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable, 2x128x32>
      %133 = tt.splat %115 : i1 -> tensor<128x32xi1, #blocked>
      %134 = arith.andi %133, %131 : tensor<128x32xi1, #blocked>
      %135 = ttg.async_copy_global_to_local %121, %132 mask %134 other %cst_0 : tensor<128x32x!tt.ptr<f32>, #blocked> -> <128x32xf32, #shared, #smem, mutable, 2x128x32>
      %136 = ttg.async_commit_group %135
      %137 = tt.splat %128 : i32 -> tensor<32x1xi32, #blocked1>
      %138 = arith.cmpi slt, %40, %137 : tensor<32x1xi32, #blocked1>
      %139 = tt.broadcast %138 : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
      %140 = ttg.memdesc_subview %54[%125, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 2x32x64>
      %141 = tt.splat %115 : i1 -> tensor<32x64xi1, #blocked1>
      %142 = arith.andi %141, %139 : tensor<32x64xi1, #blocked1>
      %143 = ttg.async_copy_global_to_local %122, %140 mask %142 other %cst_1 : tensor<32x64x!tt.ptr<f32>, #blocked1> -> <32x64xf32, #shared1, #smem, mutable, 2x32x64>
      %144 = ttg.async_commit_group %143
      %145 = arith.addi %arg14, %c1_i32 : i32
      %146 = arith.cmpi slt, %145, %c2_i32 : i32
      %147 = arith.select %146, %145, %c0_i32 : i32
      %148 = ttg.memdesc_subview %53[%147, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable, 2x128x32>
      %149 = ttg.async_wait %arg20 {num = 2 : i32}
      %150 = ttg.memdesc_subview %54[%147, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 2x32x64>
      scf.yield %120, %121, %122, %125, %147, %148, %149, %150, %149, %136, %144 : tensor<128x64xf32, #mma>, tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<32x64x!tt.ptr<f32>, #blocked1>, i32, i32, !ttg.memdesc<128x32xf32, #shared, #smem, mutable, 2x128x32>, !ttg.async.token, !ttg.memdesc<32x64xf32, #shared1, #smem, mutable, 2x32x64>, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    %96 = ttg.async_wait  {num = 0 : i32}
    ttg.local_dealloc %53 : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable>
    ttg.local_dealloc %54 : !ttg.memdesc<2x32x64xf32, #shared1, #smem, mutable>
    %97 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %98 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked1>
    %99 = arith.muli %98, %97 : tensor<128x1xi32, #blocked1>
    %100 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    %101 = tt.addptr %100, %99 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
    %102 = tt.expand_dims %26 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %103 = tt.broadcast %101 : tensor<128x1x!tt.ptr<f32>, #blocked1> -> tensor<128x64x!tt.ptr<f32>, #blocked1>
    %104 = tt.broadcast %102 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %105 = tt.addptr %103, %104 : tensor<128x64x!tt.ptr<f32>, #blocked1>, tensor<128x64xi32, #blocked1>
    %106 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1>
    %107 = arith.cmpi slt, %97, %106 : tensor<128x1xi32, #blocked1>
    %108 = tt.splat %arg4 : i32 -> tensor<1x64xi32, #blocked1>
    %109 = arith.cmpi slt, %102, %108 : tensor<1x64xi32, #blocked1>
    %110 = tt.broadcast %107 : tensor<128x1xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %111 = tt.broadcast %109 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %112 = arith.andi %110, %111 : tensor<128x64xi1, #blocked1>
    %113 = ttg.convert_layout %95#0 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked1>
    tt.store %105, %113, %112 : tensor<128x64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

