#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 24576 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %c31_i32 = arith.constant 31 : i32
    %c63_i32 = arith.constant 63 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #blocked1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #blocked2>
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst_2 = arith.constant dense<32> : tensor<128x32xi32, #blocked2>
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
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %18 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %19 = arith.addi %17, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %20 = arith.addi %18, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %21 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %22 = arith.remsi %19, %21 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %23 = arith.muli %13, %c64_i32 : i32
    %24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %25 = tt.splat %23 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %26 = arith.addi %25, %24 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %27 = tt.splat %arg4 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %28 = arith.remsi %26, %27 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %29 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %30 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked2>
    %31 = arith.muli %29, %30 : tensor<128x1xi32, #blocked2>
    %32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x32xi32, #blocked2>
    %34 = tt.broadcast %31 : tensor<128x1xi32, #blocked2> -> tensor<128x32xi32, #blocked2>
    %35 = tt.broadcast %33 : tensor<1x32xi32, #blocked2> -> tensor<128x32xi32, #blocked2>
    %36 = arith.addi %34, %35 : tensor<128x32xi32, #blocked2>
    %37 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blocked2>
    %38 = tt.addptr %37, %36 : tensor<128x32x!tt.ptr<f32>, #blocked2>, tensor<128x32xi32, #blocked2>
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
    cf.br ^bb1(%c0_i32, %cst, %38, %48 : i32, tensor<128x64xf32, #blocked>, tensor<128x32x!tt.ptr<f32>, #blocked2>, tensor<32x64x!tt.ptr<f32>, #blocked1>)
  ^bb1(%53: i32, %54: tensor<128x64xf32, #blocked>, %55: tensor<128x32x!tt.ptr<f32>, #blocked2>, %56: tensor<32x64x!tt.ptr<f32>, #blocked1>):  // 2 preds: ^bb0, ^bb2
    %57 = arith.cmpi slt, %53, %50 : i32
    cf.cond_br %57, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %58 = arith.muli %53, %c32_i32 : i32
    %59 = arith.subi %arg5, %58 : i32
    %60 = tt.splat %59 : i32 -> tensor<1x32xi32, #blocked2>
    %61 = arith.cmpi slt, %33, %60 : tensor<1x32xi32, #blocked2>
    %62 = tt.broadcast %61 : tensor<1x32xi1, #blocked2> -> tensor<128x32xi1, #blocked2>
    %63 = tt.load %55, %62, %cst_1 : tensor<128x32x!tt.ptr<f32>, #blocked2>
    %64 = ttg.local_alloc %63 {allocation.offset = 0 : i32} : (tensor<128x32xf32, #blocked2>) -> !ttg.memdesc<128x32xf32, #shared, #smem>
    %65 = tt.splat %59 : i32 -> tensor<32x1xi32, #blocked1>
    %66 = arith.cmpi slt, %40, %65 : tensor<32x1xi32, #blocked1>
    %67 = tt.broadcast %66 : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
    %68 = tt.load %56, %67, %cst_0 : tensor<32x64x!tt.ptr<f32>, #blocked1>
    %69 = ttg.local_alloc %68 {allocation.offset = 16384 : i32} : (tensor<32x64xf32, #blocked1>) -> !ttg.memdesc<32x64xf32, #shared, #smem>
    %70 = ttg.local_load %64 : !ttg.memdesc<128x32xf32, #shared, #smem> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %71 = ttg.local_load %69 : !ttg.memdesc<32x64xf32, #shared, #smem> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %72 = tt.dot %70, %71, %54, inputPrecision = tf32 : tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x64xf32, #blocked>
    %73 = tt.addptr %55, %cst_2 : tensor<128x32x!tt.ptr<f32>, #blocked2>, tensor<128x32xi32, #blocked2>
    %74 = tt.addptr %56, %52 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi32, #blocked1>
    %75 = arith.addi %53, %c1_i32 : i32
    cf.br ^bb1(%75, %72, %73, %74 : i32, tensor<128x64xf32, #blocked>, tensor<128x32x!tt.ptr<f32>, #blocked2>, tensor<32x64x!tt.ptr<f32>, #blocked1>)
  ^bb3:  // pred: ^bb1
    %76 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %77 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked1>
    %78 = arith.muli %77, %76 : tensor<128x1xi32, #blocked1>
    %79 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    %80 = tt.addptr %79, %78 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1>
    %81 = tt.expand_dims %26 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %82 = tt.broadcast %80 : tensor<128x1x!tt.ptr<f32>, #blocked1> -> tensor<128x64x!tt.ptr<f32>, #blocked1>
    %83 = tt.broadcast %81 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %84 = tt.addptr %82, %83 : tensor<128x64x!tt.ptr<f32>, #blocked1>, tensor<128x64xi32, #blocked1>
    %85 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1>
    %86 = arith.cmpi slt, %76, %85 : tensor<128x1xi32, #blocked1>
    %87 = tt.splat %arg4 : i32 -> tensor<1x64xi32, #blocked1>
    %88 = arith.cmpi slt, %81, %87 : tensor<1x64xi32, #blocked1>
    %89 = tt.broadcast %86 : tensor<128x1xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %90 = tt.broadcast %88 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %91 = arith.andi %89, %90 : tensor<128x64xi1, #blocked1>
    %92 = ttg.convert_layout %54 {allocation.offset = 0 : i32} : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked1>
    tt.store %84, %92, %91 : tensor<128x64x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

