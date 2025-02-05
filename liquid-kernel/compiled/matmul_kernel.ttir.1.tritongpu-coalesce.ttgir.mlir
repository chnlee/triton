#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c31_i32 = arith.constant 31 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %c63_i32 = arith.constant 63 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<32> : tensor<128x32xi32, #blocked1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #blocked1>
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
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
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %16 = tt.splat %14 : i32 -> tensor<128xi32, #blocked2>
    %17 = arith.addi %16, %15 : tensor<128xi32, #blocked2>
    %18 = tt.splat %arg3 : i32 -> tensor<128xi32, #blocked2>
    %19 = arith.remsi %17, %18 : tensor<128xi32, #blocked2>
    %20 = arith.muli %13, %c64_i32 : i32
    %21 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked2>
    %22 = tt.splat %20 : i32 -> tensor<64xi32, #blocked2>
    %23 = arith.addi %22, %21 : tensor<64xi32, #blocked2>
    %24 = tt.splat %arg4 : i32 -> tensor<64xi32, #blocked2>
    %25 = arith.remsi %23, %24 : tensor<64xi32, #blocked2>
    %26 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked2>
    %27 = ttg.convert_layout %19 : tensor<128xi32, #blocked2> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %28 = tt.expand_dims %27 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xi32, #blocked3>
    %29 = ttg.convert_layout %28 : tensor<128x1xi32, #blocked3> -> tensor<128x1xi32, #blocked4>
    %30 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked4>
    %31 = arith.muli %29, %30 : tensor<128x1xi32, #blocked4>
    %32 = ttg.convert_layout %26 : tensor<32xi32, #blocked2> -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x32xi32, #blocked5>
    %34 = ttg.convert_layout %33 : tensor<1x32xi32, #blocked5> -> tensor<1x32xi32, #blocked1>
    %35 = tt.broadcast %31 : tensor<128x1xi32, #blocked4> -> tensor<128x32xi32, #blocked4>
    %36 = ttg.convert_layout %35 : tensor<128x32xi32, #blocked4> -> tensor<128x32xi32, #blocked1>
    %37 = tt.broadcast %34 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %38 = arith.addi %36, %37 : tensor<128x32xi32, #blocked1>
    %39 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blocked1>
    %40 = tt.addptr %39, %38 : tensor<128x32x!tt.ptr<f32>, #blocked1>, tensor<128x32xi32, #blocked1>
    %41 = ttg.convert_layout %26 : tensor<32xi32, #blocked2> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %42 = tt.expand_dims %41 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1xi32, #blocked3>
    %43 = ttg.convert_layout %42 : tensor<32x1xi32, #blocked3> -> tensor<32x1xi32, #blocked4>
    %44 = tt.splat %arg7 : i32 -> tensor<32x1xi32, #blocked4>
    %45 = arith.muli %43, %44 : tensor<32x1xi32, #blocked4>
    %46 = ttg.convert_layout %25 : tensor<64xi32, #blocked2> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %47 = tt.expand_dims %46 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x64xi32, #blocked5>
    %48 = ttg.convert_layout %47 : tensor<1x64xi32, #blocked5> -> tensor<1x64xi32, #blocked>
    %49 = tt.broadcast %45 : tensor<32x1xi32, #blocked4> -> tensor<32x64xi32, #blocked4>
    %50 = ttg.convert_layout %49 : tensor<32x64xi32, #blocked4> -> tensor<32x64xi32, #blocked>
    %51 = tt.broadcast %48 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %52 = arith.addi %50, %51 : tensor<32x64xi32, #blocked>
    %53 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %54 = tt.addptr %53, %52 : tensor<32x64x!tt.ptr<f32>, #blocked>, tensor<32x64xi32, #blocked>
    %55 = arith.addi %arg5, %c31_i32 : i32
    %56 = arith.divsi %55, %c32_i32 : i32
    %57 = arith.muli %arg7, %c32_i32 : i32
    %58 = tt.splat %57 : i32 -> tensor<32x64xi32, #blocked>
    %59:3 = scf.for %arg9 = %c0_i32 to %56 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %40, %arg12 = %54) -> (tensor<128x64xf32, #blocked>, tensor<128x32x!tt.ptr<f32>, #blocked1>, tensor<32x64x!tt.ptr<f32>, #blocked>)  : i32 {
      %85 = arith.muli %arg9, %c32_i32 : i32
      %86 = arith.subi %arg5, %85 : i32
      %87 = tt.splat %86 : i32 -> tensor<1x32xi32, #blocked1>
      %88 = arith.cmpi slt, %34, %87 : tensor<1x32xi32, #blocked1>
      %89 = tt.broadcast %88 : tensor<1x32xi1, #blocked1> -> tensor<128x32xi1, #blocked1>
      %90 = ttg.convert_layout %arg11 : tensor<128x32x!tt.ptr<f32>, #blocked1> -> tensor<128x32x!tt.ptr<f32>, #blocked6>
      %91 = ttg.convert_layout %89 : tensor<128x32xi1, #blocked1> -> tensor<128x32xi1, #blocked6>
      %92 = ttg.convert_layout %cst_2 : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #blocked6>
      %93 = tt.load %90, %91, %92 : tensor<128x32x!tt.ptr<f32>, #blocked6>
      %94 = ttg.convert_layout %93 : tensor<128x32xf32, #blocked6> -> tensor<128x32xf32, #blocked1>
      %95 = tt.splat %86 : i32 -> tensor<32x1xi32, #blocked4>
      %96 = arith.cmpi slt, %43, %95 : tensor<32x1xi32, #blocked4>
      %97 = tt.broadcast %96 : tensor<32x1xi1, #blocked4> -> tensor<32x64xi1, #blocked4>
      %98 = ttg.convert_layout %97 : tensor<32x64xi1, #blocked4> -> tensor<32x64xi1, #blocked>
      %99 = ttg.convert_layout %arg12 : tensor<32x64x!tt.ptr<f32>, #blocked> -> tensor<32x64x!tt.ptr<f32>, #blocked7>
      %100 = ttg.convert_layout %98 : tensor<32x64xi1, #blocked> -> tensor<32x64xi1, #blocked7>
      %101 = ttg.convert_layout %cst_1 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #blocked7>
      %102 = tt.load %99, %100, %101 : tensor<32x64x!tt.ptr<f32>, #blocked7>
      %103 = ttg.convert_layout %102 : tensor<32x64xf32, #blocked7> -> tensor<32x64xf32, #blocked>
      %104 = ttg.convert_layout %94 : tensor<128x32xf32, #blocked1> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked8}>>
      %105 = ttg.convert_layout %103 : tensor<32x64xf32, #blocked> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked8}>>
      %106 = ttg.convert_layout %arg10 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked8>
      %107 = tt.dot %104, %105, %106, inputPrecision = tf32 : tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked8}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked8}>> -> tensor<128x64xf32, #blocked8>
      %108 = ttg.convert_layout %107 : tensor<128x64xf32, #blocked8> -> tensor<128x64xf32, #blocked>
      %109 = tt.addptr %arg11, %cst_0 : tensor<128x32x!tt.ptr<f32>, #blocked1>, tensor<128x32xi32, #blocked1>
      %110 = tt.addptr %arg12, %58 : tensor<32x64x!tt.ptr<f32>, #blocked>, tensor<32x64xi32, #blocked>
      scf.yield %108, %109, %110 : tensor<128x64xf32, #blocked>, tensor<128x32x!tt.ptr<f32>, #blocked1>, tensor<32x64x!tt.ptr<f32>, #blocked>
    }
    %60 = ttg.convert_layout %17 : tensor<128xi32, #blocked2> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %61 = tt.expand_dims %60 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xi32, #blocked3>
    %62 = ttg.convert_layout %61 : tensor<128x1xi32, #blocked3> -> tensor<128x1xi32, #blocked4>
    %63 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked4>
    %64 = arith.muli %63, %62 : tensor<128x1xi32, #blocked4>
    %65 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked4>
    %66 = tt.addptr %65, %64 : tensor<128x1x!tt.ptr<f32>, #blocked4>, tensor<128x1xi32, #blocked4>
    %67 = ttg.convert_layout %23 : tensor<64xi32, #blocked2> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %68 = tt.expand_dims %67 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x64xi32, #blocked5>
    %69 = ttg.convert_layout %68 : tensor<1x64xi32, #blocked5> -> tensor<1x64xi32, #blocked>
    %70 = tt.broadcast %66 : tensor<128x1x!tt.ptr<f32>, #blocked4> -> tensor<128x64x!tt.ptr<f32>, #blocked4>
    %71 = ttg.convert_layout %70 : tensor<128x64x!tt.ptr<f32>, #blocked4> -> tensor<128x64x!tt.ptr<f32>, #blocked>
    %72 = tt.broadcast %69 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %73 = tt.addptr %71, %72 : tensor<128x64x!tt.ptr<f32>, #blocked>, tensor<128x64xi32, #blocked>
    %74 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked4>
    %75 = arith.cmpi slt, %62, %74 : tensor<128x1xi32, #blocked4>
    %76 = tt.splat %arg4 : i32 -> tensor<1x64xi32, #blocked>
    %77 = arith.cmpi slt, %69, %76 : tensor<1x64xi32, #blocked>
    %78 = tt.broadcast %75 : tensor<128x1xi1, #blocked4> -> tensor<128x64xi1, #blocked4>
    %79 = ttg.convert_layout %78 : tensor<128x64xi1, #blocked4> -> tensor<128x64xi1, #blocked>
    %80 = tt.broadcast %77 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
    %81 = arith.andi %79, %80 : tensor<128x64xi1, #blocked>
    %82 = ttg.convert_layout %73 : tensor<128x64x!tt.ptr<f32>, #blocked> -> tensor<128x64x!tt.ptr<f32>, #blocked7>
    %83 = ttg.convert_layout %59#0 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked7>
    %84 = ttg.convert_layout %81 : tensor<128x64xi1, #blocked> -> tensor<128x64xi1, #blocked7>
    tt.store %82, %83, %84 : tensor<128x64x!tt.ptr<f32>, #blocked7>
    tt.return
  }
}

