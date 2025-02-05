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
    %59 = ttg.memdesc_subview %53[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable>
    %60 = tt.splat %55 : i1 -> tensor<128x32xi1, #blocked>
    %61 = arith.andi %60, %58 : tensor<128x32xi1, #blocked>
    %62 = ttg.async_copy_global_to_local %38, %59 mask %61 other %cst_0 : tensor<128x32x!tt.ptr<f32>, #blocked> -> <128x32xf32, #shared, #smem, mutable>
    %63 = ttg.async_commit_group %62
    %64 = tt.splat %arg5 : i32 -> tensor<32x1xi32, #blocked1>
    %65 = arith.cmpi slt, %40, %64 : tensor<32x1xi32, #blocked1>
    %66 = tt.broadcast %65 : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
    %67 = ttg.memdesc_subview %54[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable>
    %68 = tt.splat %55 : i1 -> tensor<32x64xi1, #blocked1>
    %69 = arith.andi %68, %66 : tensor<32x64xi1, #blocked1>
    %70 = ttg.async_copy_global_to_local %48, %67 mask %69 other %cst_1 : tensor<32x64x!tt.ptr<f32>, #blocked1> -> <32x64xf32, #shared1, #smem, mutable>
    %71 = ttg.async_commit_group %70
    %72 = arith.cmpi sgt, %50, %c1_i32 : i32
    %73 = tt.addptr %38, %cst : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi32, #blocked>
    %74 = tt.addptr %48, %52 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi32, #blocked1>
    %75 = arith.subi %arg5, %c32_i32 : i32
    %76 = tt.splat %75 : i32 -> tensor<1x32xi32, #blocked>
    %77 = arith.cmpi slt, %33, %76 : tensor<1x32xi32, #blocked>
    %78 = tt.broadcast %77 : tensor<1x32xi1, #blocked> -> tensor<128x32xi1, #blocked>
    %79 = ttg.memdesc_subview %53[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable>
    %80 = tt.splat %72 : i1 -> tensor<128x32xi1, #blocked>
    %81 = arith.andi %80, %78 : tensor<128x32xi1, #blocked>
    %82 = ttg.async_copy_global_to_local %73, %79 mask %81 other %cst_0 : tensor<128x32x!tt.ptr<f32>, #blocked> -> <128x32xf32, #shared, #smem, mutable>
    %83 = ttg.async_commit_group %82
    %84 = tt.splat %75 : i32 -> tensor<32x1xi32, #blocked1>
    %85 = arith.cmpi slt, %40, %84 : tensor<32x1xi32, #blocked1>
    %86 = tt.broadcast %85 : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
    %87 = ttg.memdesc_subview %54[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable>
    %88 = tt.splat %72 : i1 -> tensor<32x64xi1, #blocked1>
    %89 = arith.andi %88, %86 : tensor<32x64xi1, #blocked1>
    %90 = ttg.async_copy_global_to_local %74, %87 mask %89 other %cst_1 : tensor<32x64x!tt.ptr<f32>, #blocked1> -> <32x64xf32, #shared1, #smem, mutable>
    %91 = ttg.async_commit_group %90
    %92 = ttg.memdesc_subview %53[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable>
    %93 = ttg.async_wait %71 {num = 2 : i32}
    %94 = ttg.memdesc_subview %54[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable>
    %c0_i32_3 = arith.constant 0 : i32
    %c0_i32_4 = arith.constant 0 : i32
    %95 = ttg.memdesc_subview %92[%c0_i32_3, %c0_i32_4] : !ttg.memdesc<128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x8xf32, #shared, #smem, mutable, 128x32>
    %96 = ttg.local_load %95 : !ttg.memdesc<128x8xf32, #shared, #smem, mutable, 128x32> -> tensor<128x8xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %c0_i32_5 = arith.constant 0 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %97 = ttg.memdesc_subview %94[%c0_i32_5, %c0_i32_6] : !ttg.memdesc<32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<8x64xf32, #shared1, #smem, mutable, 32x64>
    %98 = ttg.local_load %97 : !ttg.memdesc<8x64xf32, #shared1, #smem, mutable, 32x64> -> tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %99:13 = scf.for %arg9 = %c0_i32 to %50 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %73, %arg12 = %74, %arg13 = %c1_i32, %arg14 = %c0_i32, %arg15 = %92, %arg16 = %93, %arg17 = %94, %arg18 = %93, %arg19 = %83, %arg20 = %91, %arg21 = %96, %arg22 = %98) -> (tensor<128x64xf32, #mma>, tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<32x64x!tt.ptr<f32>, #blocked1>, i32, i32, !ttg.memdesc<128x32xf32, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<32x64xf32, #shared1, #smem, mutable>, !ttg.async.token, !ttg.async.token, !ttg.async.token, tensor<128x8xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>, tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>)  : i32 {
      %118 = arith.subi %50, %c2_i32 : i32
      %119 = arith.cmpi slt, %arg9, %118 : i32
      %120 = ttg.local_load %arg15 : !ttg.memdesc<128x32xf32, #shared, #smem, mutable> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %121 = ttg.local_load %arg17 : !ttg.memdesc<32x64xf32, #shared1, #smem, mutable> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %122 = tt.dot %120, %121, %arg10, inputPrecision = tf32 : tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<128x64xf32, #mma>
      %c0_i32_7 = arith.constant 0 : i32
      %c8_i32_8 = arith.constant 8 : i32
      %123 = ttg.memdesc_subview %arg15[%c0_i32_7, %c8_i32_8] : !ttg.memdesc<128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x8xf32, #shared, #smem, mutable, 128x32>
      %124 = ttg.local_load %123 : !ttg.memdesc<128x8xf32, #shared, #smem, mutable, 128x32> -> tensor<128x8xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %c8_i32_9 = arith.constant 8 : i32
      %c0_i32_10 = arith.constant 0 : i32
      %125 = ttg.memdesc_subview %arg17[%c8_i32_9, %c0_i32_10] : !ttg.memdesc<32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<8x64xf32, #shared1, #smem, mutable, 32x64>
      %126 = ttg.local_load %125 : !ttg.memdesc<8x64xf32, #shared1, #smem, mutable, 32x64> -> tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %127 = tt.dot %arg21, %arg22, %arg10, inputPrecision = tf32 : tensor<128x8xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<128x64xf32, #mma>
      %c0_i32_11 = arith.constant 0 : i32
      %c16_i32 = arith.constant 16 : i32
      %128 = ttg.memdesc_subview %arg15[%c0_i32_11, %c16_i32] : !ttg.memdesc<128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x8xf32, #shared, #smem, mutable, 128x32>
      %129 = ttg.local_load %128 : !ttg.memdesc<128x8xf32, #shared, #smem, mutable, 128x32> -> tensor<128x8xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %c16_i32_12 = arith.constant 16 : i32
      %c0_i32_13 = arith.constant 0 : i32
      %130 = ttg.memdesc_subview %arg17[%c16_i32_12, %c0_i32_13] : !ttg.memdesc<32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<8x64xf32, #shared1, #smem, mutable, 32x64>
      %131 = ttg.local_load %130 : !ttg.memdesc<8x64xf32, #shared1, #smem, mutable, 32x64> -> tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %132 = tt.dot %124, %126, %127, inputPrecision = tf32 : tensor<128x8xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<128x64xf32, #mma>
      %c0_i32_14 = arith.constant 0 : i32
      %c24_i32 = arith.constant 24 : i32
      %133 = ttg.memdesc_subview %arg15[%c0_i32_14, %c24_i32] : !ttg.memdesc<128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x8xf32, #shared, #smem, mutable, 128x32>
      %134 = ttg.local_load %133 : !ttg.memdesc<128x8xf32, #shared, #smem, mutable, 128x32> -> tensor<128x8xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %c24_i32_15 = arith.constant 24 : i32
      %c0_i32_16 = arith.constant 0 : i32
      %135 = ttg.memdesc_subview %arg17[%c24_i32_15, %c0_i32_16] : !ttg.memdesc<32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<8x64xf32, #shared1, #smem, mutable, 32x64>
      %136 = ttg.local_load %135 : !ttg.memdesc<8x64xf32, #shared1, #smem, mutable, 32x64> -> tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %137 = tt.dot %129, %131, %132, inputPrecision = tf32 : tensor<128x8xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<128x64xf32, #mma>
      %138 = tt.addptr %arg11, %cst : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi32, #blocked>
      %139 = tt.addptr %arg12, %52 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi32, #blocked1>
      %140 = arith.addi %arg13, %c1_i32 : i32
      %141 = arith.cmpi slt, %140, %c2_i32 : i32
      %142 = arith.select %141, %140, %c0_i32 : i32
      %143 = arith.addi %arg9, %c2_i32 : i32
      %144 = arith.muli %143, %c32_i32 : i32
      %145 = arith.subi %arg5, %144 : i32
      %146 = tt.splat %145 : i32 -> tensor<1x32xi32, #blocked>
      %147 = arith.cmpi slt, %33, %146 : tensor<1x32xi32, #blocked>
      %148 = tt.broadcast %147 : tensor<1x32xi1, #blocked> -> tensor<128x32xi1, #blocked>
      %149 = ttg.memdesc_subview %53[%142, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable>
      %150 = tt.splat %119 : i1 -> tensor<128x32xi1, #blocked>
      %151 = arith.andi %150, %148 : tensor<128x32xi1, #blocked>
      %152 = ttg.async_copy_global_to_local %138, %149 mask %151 other %cst_0 : tensor<128x32x!tt.ptr<f32>, #blocked> -> <128x32xf32, #shared, #smem, mutable>
      %153 = ttg.async_commit_group %152
      %154 = tt.splat %145 : i32 -> tensor<32x1xi32, #blocked1>
      %155 = arith.cmpi slt, %40, %154 : tensor<32x1xi32, #blocked1>
      %156 = tt.broadcast %155 : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
      %157 = ttg.memdesc_subview %54[%142, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable>
      %158 = tt.splat %119 : i1 -> tensor<32x64xi1, #blocked1>
      %159 = arith.andi %158, %156 : tensor<32x64xi1, #blocked1>
      %160 = ttg.async_copy_global_to_local %139, %157 mask %159 other %cst_1 : tensor<32x64x!tt.ptr<f32>, #blocked1> -> <32x64xf32, #shared1, #smem, mutable>
      %161 = ttg.async_commit_group %160
      %162 = arith.addi %arg14, %c1_i32 : i32
      %163 = arith.cmpi slt, %162, %c2_i32 : i32
      %164 = arith.select %163, %162, %c0_i32 : i32
      %165 = ttg.memdesc_subview %53[%164, %c0_i32, %c0_i32] : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf32, #shared, #smem, mutable>
      %166 = ttg.async_wait %arg20 {num = 2 : i32}
      %167 = ttg.memdesc_subview %54[%164, %c0_i32, %c0_i32] : !ttg.memdesc<2x32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<32x64xf32, #shared1, #smem, mutable>
      %c0_i32_17 = arith.constant 0 : i32
      %c0_i32_18 = arith.constant 0 : i32
      %168 = ttg.memdesc_subview %165[%c0_i32_17, %c0_i32_18] : !ttg.memdesc<128x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<128x8xf32, #shared, #smem, mutable, 128x32>
      %169 = ttg.local_load %168 : !ttg.memdesc<128x8xf32, #shared, #smem, mutable, 128x32> -> tensor<128x8xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %c0_i32_19 = arith.constant 0 : i32
      %c0_i32_20 = arith.constant 0 : i32
      %170 = ttg.memdesc_subview %167[%c0_i32_19, %c0_i32_20] : !ttg.memdesc<32x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<8x64xf32, #shared1, #smem, mutable, 32x64>
      %171 = ttg.local_load %170 : !ttg.memdesc<8x64xf32, #shared1, #smem, mutable, 32x64> -> tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %172 = tt.dot %134, %136, %137, inputPrecision = tf32 : tensor<128x8xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<128x64xf32, #mma>
      scf.yield %172, %138, %139, %142, %164, %165, %166, %167, %166, %153, %161, %169, %171 : tensor<128x64xf32, #mma>, tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<32x64x!tt.ptr<f32>, #blocked1>, i32, i32, !ttg.memdesc<128x32xf32, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<32x64xf32, #shared1, #smem, mutable>, !ttg.async.token, !ttg.async.token, !ttg.async.token, tensor<128x8xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>, tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    }
    %100 = ttg.async_wait  {num = 0 : i32}
    ttg.local_dealloc %53 : !ttg.memdesc<2x128x32xf32, #shared, #smem, mutable>
    ttg.local_dealloc %54 : !ttg.memdesc<2x32x64xf32, #shared1, #smem, mutable>
    %101 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %102 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked1>
    %103 = arith.muli %102, %101 : tensor<128x1xi32, #blocked1>
    %104 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    %105 = tt.addptr %104, %103 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
    %106 = tt.expand_dims %26 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %107 = tt.broadcast %105 : tensor<128x1x!tt.ptr<f32>, #blocked1> -> tensor<128x64x!tt.ptr<f32>, #blocked1>
    %108 = tt.broadcast %106 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %109 = tt.addptr %107, %108 : tensor<128x64x!tt.ptr<f32>, #blocked1>, tensor<128x64xi32, #blocked1>
    %110 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1>
    %111 = arith.cmpi slt, %101, %110 : tensor<128x1xi32, #blocked1>
    %112 = tt.splat %arg4 : i32 -> tensor<1x64xi32, #blocked1>
    %113 = arith.cmpi slt, %106, %112 : tensor<1x64xi32, #blocked1>
    %114 = tt.broadcast %111 : tensor<128x1xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %115 = tt.broadcast %113 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %116 = arith.andi %114, %115 : tensor<128x64xi1, #blocked1>
    %117 = ttg.convert_layout %99#0 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked1>
    tt.store %109, %117, %116 : tensor<128x64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

