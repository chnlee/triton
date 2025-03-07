#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#loc = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":249:0)
#shared = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":249:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":249:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":249:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":249:0), %arg4: i32 {tt.divisibility = 16 : i32} loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":249:0), %arg5: i32 {tt.divisibility = 16 : i32} loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":249:0), %arg6: i32 {tt.divisibility = 16 : i32} loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":249:0), %arg7: i32 {tt.divisibility = 16 : i32} loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":249:0), %arg8: i32 {tt.divisibility = 16 : i32} loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":249:0)) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked> loc(#loc1)
    %c31_i32 = arith.constant 31 : i32 loc(#loc1)
    %c63_i32 = arith.constant 63 : i32 loc(#loc1)
    %c127_i32 = arith.constant 127 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #blocked1> loc(#loc1)
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #blocked2> loc(#loc1)
    %c8_i32 = arith.constant 8 : i32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %cst_2 = arith.constant dense<32> : tensor<128x32xi32, #blocked2> loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.addi %arg3, %c127_i32 : i32 loc(#loc57)
    %2 = arith.divsi %1, %c128_i32 : i32 loc(#loc58)
    %3 = arith.addi %arg4, %c63_i32 : i32 loc(#loc59)
    %4 = arith.divsi %3, %c64_i32 : i32 loc(#loc60)
    %5 = arith.muli %4, %c8_i32 : i32 loc(#loc7)
    %6 = arith.divsi %0, %5 : i32 loc(#loc8)
    %7 = arith.muli %6, %c8_i32 : i32 loc(#loc9)
    %8 = arith.subi %2, %7 : i32 loc(#loc10)
    %9 = arith.minsi %8, %c8_i32 : i32 loc(#loc11)
    %10 = arith.remsi %0, %5 : i32 loc(#loc12)
    %11 = arith.remsi %10, %9 : i32 loc(#loc13)
    %12 = arith.addi %7, %11 : i32 loc(#loc14)
    %13 = arith.divsi %10, %9 : i32 loc(#loc15)
    %14 = arith.muli %12, %c128_i32 : i32 loc(#loc16)
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc17)
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc17)
    %17 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc18)
    %18 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc18)
    %19 = arith.addi %17, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc18)
    %20 = arith.addi %18, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc18)
    %21 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc19)
    %22 = arith.remsi %19, %21 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc19)
    %23 = arith.muli %13, %c64_i32 : i32 loc(#loc20)
    %24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc21)
    %25 = tt.splat %23 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc22)
    %26 = arith.addi %25, %24 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc22)
    %27 = tt.splat %arg4 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc23)
    %28 = arith.remsi %26, %27 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc23)
    %29 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2> loc(#loc24)
    %30 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked2> loc(#loc25)
    %31 = arith.muli %29, %30 : tensor<128x1xi32, #blocked2> loc(#loc25)
    %32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> loc(#loc26)
    %33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x32xi32, #blocked2> loc(#loc26)
    %34 = tt.broadcast %31 : tensor<128x1xi32, #blocked2> -> tensor<128x32xi32, #blocked2> loc(#loc27)
    %35 = tt.broadcast %33 : tensor<1x32xi32, #blocked2> -> tensor<128x32xi32, #blocked2> loc(#loc27)
    %36 = arith.addi %34, %35 : tensor<128x32xi32, #blocked2> loc(#loc27)
    %37 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blocked2> loc(#loc28)
    %38 = tt.addptr %37, %36 : tensor<128x32x!tt.ptr<f32>, #blocked2>, tensor<128x32xi32, #blocked2> loc(#loc28)
    %39 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc29)
    %40 = tt.expand_dims %39 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1> loc(#loc29)
    %41 = tt.splat %arg7 : i32 -> tensor<32x1xi32, #blocked1> loc(#loc30)
    %42 = arith.muli %40, %41 : tensor<32x1xi32, #blocked1> loc(#loc30)
    %43 = tt.expand_dims %28 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1> loc(#loc31)
    %44 = tt.broadcast %42 : tensor<32x1xi32, #blocked1> -> tensor<32x64xi32, #blocked1> loc(#loc32)
    %45 = tt.broadcast %43 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1> loc(#loc32)
    %46 = arith.addi %44, %45 : tensor<32x64xi32, #blocked1> loc(#loc32)
    %47 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked1> loc(#loc33)
    %48 = tt.addptr %47, %46 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi32, #blocked1> loc(#loc33)
    %49 = arith.addi %arg5, %c31_i32 : i32 loc(#loc61)
    %50 = arith.divsi %49, %c32_i32 : i32 loc(#loc62)
    %51 = arith.muli %arg7, %c32_i32 : i32 loc(#loc35)
    %52 = tt.splat %51 : i32 -> tensor<32x64xi32, #blocked1> loc(#loc36)
    %53:3 = scf.for %arg9 = %c0_i32 to %50 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %38, %arg12 = %48) -> (tensor<128x64xf32, #blocked>, tensor<128x32x!tt.ptr<f32>, #blocked2>, tensor<32x64x!tt.ptr<f32>, #blocked1>)  : i32 {
      %71 = arith.muli %arg9, %c32_i32 : i32 loc(#loc38)
      %72 = arith.subi %arg5, %71 : i32 loc(#loc39)
      %73 = tt.splat %72 : i32 -> tensor<1x32xi32, #blocked2> loc(#loc40)
      %74 = arith.cmpi slt, %33, %73 : tensor<1x32xi32, #blocked2> loc(#loc40)
      %75 = tt.broadcast %74 : tensor<1x32xi1, #blocked2> -> tensor<128x32xi1, #blocked2> loc(#loc41)
      %76 = tt.load %arg11, %75, %cst_1 : tensor<128x32x!tt.ptr<f32>, #blocked2> loc(#loc41)
      %77 = ttg.local_alloc %76 : (tensor<128x32xf32, #blocked2>) -> !ttg.memdesc<128x32xf32, #shared, #smem> loc(#loc41)
      %78 = tt.splat %72 : i32 -> tensor<32x1xi32, #blocked1> loc(#loc42)
      %79 = arith.cmpi slt, %40, %78 : tensor<32x1xi32, #blocked1> loc(#loc42)
      %80 = tt.broadcast %79 : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1> loc(#loc43)
      %81 = tt.load %arg12, %80, %cst_0 : tensor<32x64x!tt.ptr<f32>, #blocked1> loc(#loc43)
      %82 = ttg.local_alloc %81 : (tensor<32x64xf32, #blocked1>) -> !ttg.memdesc<32x64xf32, #shared, #smem> loc(#loc43)
      %83 = ttg.local_load %77 : !ttg.memdesc<128x32xf32, #shared, #smem> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> loc(#loc41)
      %84 = ttg.local_load %82 : !ttg.memdesc<32x64xf32, #shared, #smem> -> tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> loc(#loc43)
      %85 = tt.dot %83, %84, %arg10, inputPrecision = tf32 : tensor<128x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x64xf32, #blocked> loc(#loc44)
      %86 = tt.addptr %arg11, %cst_2 : tensor<128x32x!tt.ptr<f32>, #blocked2>, tensor<128x32xi32, #blocked2> loc(#loc45)
      %87 = tt.addptr %arg12, %52 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi32, #blocked1> loc(#loc36)
      scf.yield %85, %86, %87 : tensor<128x64xf32, #blocked>, tensor<128x32x!tt.ptr<f32>, #blocked2>, tensor<32x64x!tt.ptr<f32>, #blocked1> loc(#loc46)
    } loc(#loc37)
    %54 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1> loc(#loc47)
    %55 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked1> loc(#loc48)
    %56 = arith.muli %55, %54 : tensor<128x1xi32, #blocked1> loc(#loc48)
    %57 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1> loc(#loc49)
    %58 = tt.addptr %57, %56 : tensor<128x1x!tt.ptr<f32>, #blocked1>, tensor<128x1xi32, #blocked1> loc(#loc49)
    %59 = tt.expand_dims %26 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1> loc(#loc50)
    %60 = tt.broadcast %58 : tensor<128x1x!tt.ptr<f32>, #blocked1> -> tensor<128x64x!tt.ptr<f32>, #blocked1> loc(#loc51)
    %61 = tt.broadcast %59 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc51)
    %62 = tt.addptr %60, %61 : tensor<128x64x!tt.ptr<f32>, #blocked1>, tensor<128x64xi32, #blocked1> loc(#loc51)
    %63 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1> loc(#loc52)
    %64 = arith.cmpi slt, %54, %63 : tensor<128x1xi32, #blocked1> loc(#loc52)
    %65 = tt.splat %arg4 : i32 -> tensor<1x64xi32, #blocked1> loc(#loc53)
    %66 = arith.cmpi slt, %59, %65 : tensor<1x64xi32, #blocked1> loc(#loc53)
    %67 = tt.broadcast %64 : tensor<128x1xi1, #blocked1> -> tensor<128x64xi1, #blocked1> loc(#loc54)
    %68 = tt.broadcast %66 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1> loc(#loc54)
    %69 = arith.andi %67, %68 : tensor<128x64xi1, #blocked1> loc(#loc54)
    %70 = ttg.convert_layout %53#0 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked1> loc(#loc55)
    tt.store %62, %70, %69 : tensor<128x64x!tt.ptr<f32>, #blocked1> loc(#loc55)
    tt.return loc(#loc56)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":265:24)
#loc3 = loc("/home/chan/triton/python/triton/language/standard.py":40:22)
#loc4 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":266:27)
#loc5 = loc("/home/chan/triton/python/triton/language/standard.py":40:28)
#loc6 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":267:27)
#loc7 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":268:38)
#loc8 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":269:22)
#loc9 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":270:29)
#loc10 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":271:35)
#loc11 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":271:48)
#loc12 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":272:34)
#loc13 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":272:54)
#loc14 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":272:27)
#loc15 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":273:40)
#loc16 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":275:23)
#loc17 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":275:51)
#loc18 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":275:38)
#loc19 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":275:68)
#loc20 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":276:23)
#loc21 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":276:51)
#loc22 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":276:38)
#loc23 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":276:68)
#loc24 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":278:30)
#loc25 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":278:41)
#loc26 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":278:60)
#loc27 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":278:53)
#loc28 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":278:22)
#loc29 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":279:29)
#loc30 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":279:40)
#loc31 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":279:60)
#loc32 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":279:52)
#loc33 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":279:22)
#loc34 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":281:33)
#loc35 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":290:33)
#loc36 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":290:18)
#loc37 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":281:22)
#loc38 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":284:59)
#loc39 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":284:55)
#loc40 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":284:51)
#loc41 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":284:20)
#loc42 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":285:51)
#loc43 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":285:20)
#loc44 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":287:35)
#loc45 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":289:18)
#loc46 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":290:8)
#loc47 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":298:41)
#loc48 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":298:33)
#loc49 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":298:21)
#loc50 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":298:72)
#loc51 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":298:52)
#loc52 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":299:33)
#loc53 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":299:58)
#loc54 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":299:39)
#loc55 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":300:21)
#loc56 = loc("/home/chan/triton/python/tutorials/03-matrix-multiplication.py":300:4)
#loc57 = loc(callsite(#loc3 at #loc4))
#loc58 = loc(callsite(#loc5 at #loc4))
#loc59 = loc(callsite(#loc3 at #loc6))
#loc60 = loc(callsite(#loc5 at #loc6))
#loc61 = loc(callsite(#loc3 at #loc34))
#loc62 = loc(callsite(#loc5 at #loc34))
