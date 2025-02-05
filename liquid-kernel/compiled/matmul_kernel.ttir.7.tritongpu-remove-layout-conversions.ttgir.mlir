#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    %40 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %41 = tt.expand_dims %39 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %42 = tt.expand_dims %40 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %43 = tt.splat %arg7 : i32 -> tensor<32x1xi32, #blocked1>
    %44 = arith.muli %41, %43 : tensor<32x1xi32, #blocked1>
    %45 = tt.expand_dims %28 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %46 = tt.broadcast %44 : tensor<32x1xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
    %47 = tt.broadcast %45 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
    %48 = arith.addi %46, %47 : tensor<32x64xi32, #blocked1>
    %49 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked1>
    %50 = tt.addptr %49, %48 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi32, #blocked1>
    %51 = arith.addi %arg5, %c31_i32 : i32
    %52 = arith.divsi %51, %c32_i32 : i32
    %53 = arith.muli %arg7, %c32_i32 : i32
    %54 = tt.splat %53 : i32 -> tensor<32x64xi32, #blocked1>
    %55:3 = scf.for %arg9 = %c0_i32 to %52 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %38, %arg12 = %50) -> (tensor<128x64xf32, #mma>, tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<32x64x!tt.ptr<f32>, #blocked1>)  : i32 {
      %73 = arith.muli %arg9, %c32_i32 : i32
      %74 = arith.subi %arg5, %73 : i32
      %75 = tt.splat %74 : i32 -> tensor<1x32xi32, #blocked>
      %76 = arith.cmpi slt, %33, %75 : tensor<1x32xi32, #blocked>
      %77 = tt.broadcast %76 : tensor<1x32xi1, #blocked> -> tensor<128x32xi1, #blocked>
      %78 = tt.load %arg11, %77, %cst_0 : tensor<128x32x!tt.ptr<f32>, #blocked>
      %79 = tt.splat %74 : i32 -> tensor<32x1xi32, #blocked1>
      %80 = arith.cmpi slt, %42, %79 : tensor<32x1xi32, #blocked1>
      %81 = tt.broadcast %80 : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
      %82 = tt.load %arg12, %81, %cst_1 : tensor<32x64x!tt.ptr<f32>, #blocked1>
      %83 = ttg.convert_layout %78 : tensor<128x32xf32, #blocked> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %84 = ttg.convert_layout %82 : tensor<32x64xf32, #blocked1> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %85 = tt.dot %83, %84, %arg10, inputPrecision = tf32 : tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<128x64xf32, #mma>
      %86 = tt.addptr %arg11, %cst : tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<128x32xi32, #blocked>
      %87 = tt.addptr %arg12, %54 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi32, #blocked1>
      scf.yield %85, %86, %87 : tensor<128x64xf32, #mma>, tensor<128x32x!tt.ptr<f32>, #blocked>, tensor<32x64x!tt.ptr<f32>, #blocked1>
    }
    %56 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %57 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked1>
    %58 = arith.muli %57, %56 : tensor<128x1xi32, #blocked1>
    %59 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    %60 = tt.addptr %59, %58 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
    %61 = tt.expand_dims %26 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %62 = tt.broadcast %60 : tensor<128x1x!tt.ptr<f32>, #blocked1> -> tensor<128x64x!tt.ptr<f32>, #blocked1>
    %63 = tt.broadcast %61 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %64 = tt.addptr %62, %63 : tensor<128x64x!tt.ptr<f32>, #blocked1>, tensor<128x64xi32, #blocked1>
    %65 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1>
    %66 = arith.cmpi slt, %56, %65 : tensor<128x1xi32, #blocked1>
    %67 = tt.splat %arg4 : i32 -> tensor<1x64xi32, #blocked1>
    %68 = arith.cmpi slt, %61, %67 : tensor<1x64xi32, #blocked1>
    %69 = tt.broadcast %66 : tensor<128x1xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %70 = tt.broadcast %68 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %71 = arith.andi %69, %70 : tensor<128x64xi1, #blocked1>
    %72 = ttg.convert_layout %55#0 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked1>
    tt.store %64, %72, %71 : tensor<128x64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

