; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"

@global_smem = external addrspace(3) global [0 x i8], align 16

define void @bmm_dot_kernel(i32 %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, ptr addrspace(1) nocapture readnone %10) local_unnamed_addr !dbg !7 {
  %12 = tail call i32 asm "mov.u32 $0, %ctaid.x;", "=r"() #3, !dbg !10
  %13 = tail call i32 asm "mov.u32 $0, %ctaid.y;", "=r"() #3, !dbg !11
  %14 = tail call i32 asm "mov.u32 $0, %ctaid.z;", "=r"() #3, !dbg !12
  %15 = shl i32 %12, 4, !dbg !13
  %16 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !14
  %17 = and i32 %16, 64, !dbg !14
  %18 = lshr i32 %16, 3, !dbg !14
  %19 = and i32 %18, 15, !dbg !14
  %20 = shl i32 %16, 2, !dbg !14
  %21 = and i32 %20, 12, !dbg !14
  %22 = or disjoint i32 %15, %19, !dbg !15
  %23 = shl i32 %13, 4, !dbg !16
  %24 = or disjoint i32 %23, %21, !dbg !17
  %25 = shl i32 %14, 1, !dbg !18
  %26 = and i32 %16, 4, !dbg !19
  %.lobit = lshr exact i32 %26, 2, !dbg !19
  %27 = or disjoint i32 %25, %.lobit, !dbg !20
  %28 = sext i32 %24 to i64, !dbg !21
  %29 = icmp sgt i32 %0, 0, !dbg !22
  br i1 %29, label %.lr.ph, label %.._crit_edge_crit_edge, !dbg !22

.._crit_edge_crit_edge:                           ; preds = %11
  %.pre = shl i32 %16, 1, !dbg !23
  %.pre28 = and i32 %.pre, 14, !dbg !23
  br label %._crit_edge, !dbg !22

.lr.ph:                                           ; preds = %11
  %30 = shl i32 %7, 4, !dbg !24
  %31 = mul i32 %27, %6, !dbg !25
  %32 = sext i32 %31 to i64, !dbg !26
  %33 = getelementptr float, ptr addrspace(1) %2, i64 %32, !dbg !26
  %34 = mul i32 %7, %19, !dbg !27
  %35 = sext i32 %34 to i64, !dbg !28
  %36 = getelementptr float, ptr addrspace(1) %33, i64 %35, !dbg !28
  %37 = getelementptr float, ptr addrspace(1) %36, i64 %28, !dbg !21
  %38 = mul i32 %27, %4, !dbg !29
  %39 = sext i32 %38 to i64, !dbg !30
  %40 = getelementptr float, ptr addrspace(1) %1, i64 %39, !dbg !30
  %41 = mul i32 %22, %5, !dbg !31
  %42 = sext i32 %41 to i64, !dbg !32
  %43 = getelementptr float, ptr addrspace(1) %40, i64 %42, !dbg !32
  %44 = zext nneg i32 %21 to i64, !dbg !33
  %45 = getelementptr float, ptr addrspace(1) %43, i64 %44, !dbg !33
  %46 = shl nuw nsw i32 %19, 4
  %47 = shl nuw nsw i32 %26, 6
  %48 = or disjoint i32 %46, %47
  %49 = or disjoint i32 %48, %21
  %50 = zext nneg i32 %49 to i64
  %51 = getelementptr inbounds float, ptr addrspace(3) @global_smem, i64 %50
  %52 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %50
  %53 = and i32 %20, 224
  %54 = shl nuw nsw i32 %17, 2
  %55 = or disjoint i32 %53, %54
  %56 = zext nneg i32 %55 to i64
  %57 = getelementptr inbounds float, ptr addrspace(3) @global_smem, i64 %56
  %58 = or disjoint i32 %55, 16
  %59 = zext nneg i32 %58 to i64
  %60 = getelementptr inbounds float, ptr addrspace(3) @global_smem, i64 %59
  %61 = shl i32 %16, 1
  %62 = and i32 %61, 14
  %63 = or disjoint i32 %62, %54
  %64 = zext nneg i32 %63 to i64
  %65 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %64
  %66 = or disjoint i32 %63, 16
  %67 = zext nneg i32 %66 to i64
  %68 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %67
  %69 = or disjoint i32 %63, 32
  %70 = zext nneg i32 %69 to i64
  %71 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %70
  %72 = or disjoint i32 %63, 48
  %73 = zext nneg i32 %72 to i64
  %74 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %73
  %75 = or disjoint i32 %63, 64
  %76 = zext nneg i32 %75 to i64
  %77 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %76
  %78 = or disjoint i32 %63, 80
  %79 = zext nneg i32 %78 to i64
  %80 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %79
  %81 = or disjoint i32 %63, 96
  %82 = zext nneg i32 %81 to i64
  %83 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %82
  %84 = or disjoint i32 %63, 112
  %85 = zext nneg i32 %84 to i64
  %86 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %85
  %87 = or disjoint i32 %63, 128
  %88 = zext nneg i32 %87 to i64
  %89 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %88
  %90 = or disjoint i32 %63, 144
  %91 = zext nneg i32 %90 to i64
  %92 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %91
  %93 = or disjoint i32 %63, 160
  %94 = zext nneg i32 %93 to i64
  %95 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %94
  %96 = or disjoint i32 %63, 176
  %97 = zext nneg i32 %96 to i64
  %98 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %97
  %99 = or disjoint i32 %63, 192
  %100 = zext nneg i32 %99 to i64
  %101 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %100
  %102 = or disjoint i32 %63, 208
  %103 = zext nneg i32 %102 to i64
  %104 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %103
  %105 = or disjoint i32 %63, 224
  %106 = zext nneg i32 %105 to i64
  %107 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %106
  %108 = or disjoint i32 %63, 240
  %109 = zext nneg i32 %108 to i64
  %110 = getelementptr inbounds float, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 2048), i64 %109
  %111 = sext i32 %30 to i64
  %112 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 8
  %113 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 16
  %114 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 24
  %115 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 32
  %116 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 40
  %117 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 48
  %118 = getelementptr inbounds i8, ptr addrspace(3) %57, i64 56
  %119 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 8
  %120 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 16
  %121 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 24
  %122 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 32
  %123 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 40
  %124 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 48
  %125 = getelementptr inbounds i8, ptr addrspace(3) %60, i64 56
  br label %126, !dbg !22

126:                                              ; preds = %.lr.ph, %126
  %.pn1726 = phi ptr addrspace(1) [ %37, %.lr.ph ], [ %228, %126 ]
  %.pn925 = phi ptr addrspace(1) [ %45, %.lr.ph ], [ %227, %126 ]
  %127 = phi i32 [ 0, %.lr.ph ], [ %229, %126 ]
  %128 = phi <4 x float> [ zeroinitializer, %.lr.ph ], [ %226, %126 ]
  %129 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0;\0A\09mov.u32 $1, 0x0;\0A\09mov.u32 $2, 0x0;\0A\09mov.u32 $3, 0x0;\0A\09ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %.pn925) #3, !dbg !34
  %130 = extractvalue { i32, i32, i32, i32 } %129, 0, !dbg !34
  %131 = extractvalue { i32, i32, i32, i32 } %129, 1, !dbg !34
  %132 = extractvalue { i32, i32, i32, i32 } %129, 2, !dbg !34
  %133 = extractvalue { i32, i32, i32, i32 } %129, 3, !dbg !34
  tail call void @llvm.nvvm.barrier0(), !dbg !34
  %134 = insertelement <4 x i32> poison, i32 %130, i64 0, !dbg !34
  %135 = insertelement <4 x i32> %134, i32 %131, i64 1, !dbg !34
  %136 = insertelement <4 x i32> %135, i32 %132, i64 2, !dbg !34
  %137 = insertelement <4 x i32> %136, i32 %133, i64 3, !dbg !34
  store <4 x i32> %137, ptr addrspace(3) %51, align 16, !dbg !34
  %138 = tail call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, 0x0;\0A\09mov.u32 $1, 0x0;\0A\09mov.u32 $2, 0x0;\0A\09mov.u32 $3, 0x0;\0A\09ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 + 0 ];", "=r,=r,=r,=r,l"(ptr addrspace(1) %.pn1726) #3, !dbg !35
  %139 = extractvalue { i32, i32, i32, i32 } %138, 0, !dbg !35
  %140 = extractvalue { i32, i32, i32, i32 } %138, 1, !dbg !35
  %141 = extractvalue { i32, i32, i32, i32 } %138, 2, !dbg !35
  %142 = extractvalue { i32, i32, i32, i32 } %138, 3, !dbg !35
  %143 = insertelement <4 x i32> poison, i32 %139, i64 0, !dbg !35
  %144 = insertelement <4 x i32> %143, i32 %140, i64 1, !dbg !35
  %145 = insertelement <4 x i32> %144, i32 %141, i64 2, !dbg !35
  %146 = insertelement <4 x i32> %145, i32 %142, i64 3, !dbg !35
  store <4 x i32> %146, ptr addrspace(3) %52, align 16, !dbg !35
  tail call void @llvm.nvvm.barrier0(), !dbg !34
  %147 = load <2 x float>, ptr addrspace(3) %57, align 64, !dbg !34
  %148 = load <2 x float>, ptr addrspace(3) %112, align 8, !dbg !34
  %149 = load <2 x float>, ptr addrspace(3) %113, align 16, !dbg !34
  %150 = load <2 x float>, ptr addrspace(3) %60, align 64, !dbg !34
  %151 = load <2 x float>, ptr addrspace(3) %119, align 8, !dbg !34
  %152 = load <2 x float>, ptr addrspace(3) %120, align 16, !dbg !34
  %153 = load <2 x float>, ptr addrspace(3) %65, align 8, !dbg !35
  %154 = shufflevector <2 x float> %153, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %155 = load <2 x float>, ptr addrspace(3) %68, align 8, !dbg !35
  %156 = shufflevector <2 x float> %155, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %157 = load <2 x float>, ptr addrspace(3) %71, align 8, !dbg !35
  %158 = shufflevector <2 x float> %157, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %159 = load <2 x float>, ptr addrspace(3) %74, align 8, !dbg !35
  %160 = shufflevector <2 x float> %159, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %161 = load <2 x float>, ptr addrspace(3) %77, align 8, !dbg !35
  %162 = shufflevector <2 x float> %161, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %163 = load <2 x float>, ptr addrspace(3) %80, align 8, !dbg !35
  %164 = shufflevector <2 x float> %163, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %165 = shufflevector <2 x float> %147, <2 x float> %150, <4 x i32> <i32 0, i32 0, i32 2, i32 2>, !dbg !36
  %166 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %165, <4 x float> %154, <4 x float> %128), !dbg !36
  %167 = shufflevector <2 x float> %147, <2 x float> %150, <4 x i32> <i32 1, i32 1, i32 3, i32 3>, !dbg !36
  %168 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %167, <4 x float> %156, <4 x float> %166), !dbg !36
  %169 = shufflevector <2 x float> %148, <2 x float> %151, <4 x i32> <i32 0, i32 0, i32 2, i32 2>, !dbg !36
  %170 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %169, <4 x float> %158, <4 x float> %168), !dbg !36
  %171 = shufflevector <2 x float> %148, <2 x float> %151, <4 x i32> <i32 1, i32 1, i32 3, i32 3>, !dbg !36
  %172 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %171, <4 x float> %160, <4 x float> %170), !dbg !36
  %173 = shufflevector <2 x float> %149, <2 x float> %152, <4 x i32> <i32 0, i32 0, i32 2, i32 2>, !dbg !36
  %174 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %173, <4 x float> %162, <4 x float> %172), !dbg !36
  %175 = shufflevector <2 x float> %149, <2 x float> %152, <4 x i32> <i32 1, i32 1, i32 3, i32 3>, !dbg !36
  %176 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %175, <4 x float> %164, <4 x float> %174), !dbg !36
  %177 = load <2 x float>, ptr addrspace(3) %114, align 8, !dbg !34
  %178 = load <2 x float>, ptr addrspace(3) %115, align 32, !dbg !34
  %179 = load <2 x float>, ptr addrspace(3) %116, align 8, !dbg !34
  %180 = load <2 x float>, ptr addrspace(3) %117, align 16, !dbg !34
  %181 = load <2 x float>, ptr addrspace(3) %118, align 8, !dbg !34
  %182 = load <2 x float>, ptr addrspace(3) %121, align 8, !dbg !34
  %183 = load <2 x float>, ptr addrspace(3) %122, align 32, !dbg !34
  %184 = load <2 x float>, ptr addrspace(3) %123, align 8, !dbg !34
  %185 = load <2 x float>, ptr addrspace(3) %124, align 16, !dbg !34
  %186 = load <2 x float>, ptr addrspace(3) %125, align 8, !dbg !34
  %187 = load <2 x float>, ptr addrspace(3) %83, align 8, !dbg !35
  %188 = shufflevector <2 x float> %187, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %189 = load <2 x float>, ptr addrspace(3) %86, align 8, !dbg !35
  %190 = shufflevector <2 x float> %189, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %191 = load <2 x float>, ptr addrspace(3) %89, align 8, !dbg !35
  %192 = shufflevector <2 x float> %191, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %193 = load <2 x float>, ptr addrspace(3) %92, align 8, !dbg !35
  %194 = shufflevector <2 x float> %193, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %195 = load <2 x float>, ptr addrspace(3) %95, align 8, !dbg !35
  %196 = shufflevector <2 x float> %195, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %197 = load <2 x float>, ptr addrspace(3) %98, align 8, !dbg !35
  %198 = shufflevector <2 x float> %197, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %199 = load <2 x float>, ptr addrspace(3) %101, align 8, !dbg !35
  %200 = shufflevector <2 x float> %199, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %201 = load <2 x float>, ptr addrspace(3) %104, align 8, !dbg !35
  %202 = shufflevector <2 x float> %201, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %203 = load <2 x float>, ptr addrspace(3) %107, align 8, !dbg !35
  %204 = shufflevector <2 x float> %203, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %205 = load <2 x float>, ptr addrspace(3) %110, align 8, !dbg !35
  %206 = shufflevector <2 x float> %205, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>, !dbg !35
  %207 = shufflevector <2 x float> %177, <2 x float> %182, <4 x i32> <i32 0, i32 0, i32 2, i32 2>, !dbg !36
  %208 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %207, <4 x float> %188, <4 x float> %176), !dbg !36
  %209 = shufflevector <2 x float> %177, <2 x float> %182, <4 x i32> <i32 1, i32 1, i32 3, i32 3>, !dbg !36
  %210 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %209, <4 x float> %190, <4 x float> %208), !dbg !36
  %211 = shufflevector <2 x float> %178, <2 x float> %183, <4 x i32> <i32 0, i32 0, i32 2, i32 2>, !dbg !36
  %212 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %211, <4 x float> %192, <4 x float> %210), !dbg !36
  %213 = shufflevector <2 x float> %178, <2 x float> %183, <4 x i32> <i32 1, i32 1, i32 3, i32 3>, !dbg !36
  %214 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %213, <4 x float> %194, <4 x float> %212), !dbg !36
  %215 = shufflevector <2 x float> %179, <2 x float> %184, <4 x i32> <i32 0, i32 0, i32 2, i32 2>, !dbg !36
  %216 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %215, <4 x float> %196, <4 x float> %214), !dbg !36
  %217 = shufflevector <2 x float> %179, <2 x float> %184, <4 x i32> <i32 1, i32 1, i32 3, i32 3>, !dbg !36
  %218 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %217, <4 x float> %198, <4 x float> %216), !dbg !36
  %219 = shufflevector <2 x float> %180, <2 x float> %185, <4 x i32> <i32 0, i32 0, i32 2, i32 2>, !dbg !36
  %220 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %219, <4 x float> %200, <4 x float> %218), !dbg !36
  %221 = shufflevector <2 x float> %180, <2 x float> %185, <4 x i32> <i32 1, i32 1, i32 3, i32 3>, !dbg !36
  %222 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %221, <4 x float> %202, <4 x float> %220), !dbg !36
  %223 = shufflevector <2 x float> %181, <2 x float> %186, <4 x i32> <i32 0, i32 0, i32 2, i32 2>, !dbg !36
  %224 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %223, <4 x float> %204, <4 x float> %222), !dbg !36
  %225 = shufflevector <2 x float> %181, <2 x float> %186, <4 x i32> <i32 1, i32 1, i32 3, i32 3>, !dbg !36
  %226 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %225, <4 x float> %206, <4 x float> %224), !dbg !36
  %227 = getelementptr i8, ptr addrspace(1) %.pn925, i64 64, !dbg !37
  %228 = getelementptr float, ptr addrspace(1) %.pn1726, i64 %111, !dbg !38
  %229 = add i32 %127, 16, !dbg !22
  %230 = icmp slt i32 %229, %0, !dbg !22
  br i1 %230, label %126, label %._crit_edge.loopexit, !dbg !22

._crit_edge.loopexit:                             ; preds = %126
  %231 = bitcast <4 x float> %226 to <4 x i32>, !dbg !23
  br label %._crit_edge, !dbg !39

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %.._crit_edge_crit_edge
  %.pre-phi29 = phi i32 [ %.pre28, %.._crit_edge_crit_edge ], [ %62, %._crit_edge.loopexit ], !dbg !23
  %232 = phi <4 x i32> [ zeroinitializer, %.._crit_edge_crit_edge ], [ %231, %._crit_edge.loopexit ]
  %233 = mul i32 %27, %8, !dbg !39
  %234 = sext i32 %233 to i64, !dbg !40
  %235 = getelementptr float, ptr addrspace(1) %3, i64 %234, !dbg !40
  %236 = mul i32 %22, %9, !dbg !41
  %237 = sext i32 %236 to i64, !dbg !42
  %238 = getelementptr float, ptr addrspace(1) %235, i64 %237, !dbg !42
  %239 = getelementptr float, ptr addrspace(1) %238, i64 %28, !dbg !43
  tail call void @llvm.nvvm.barrier0(), !dbg !23
  %240 = shl i32 %16, 3, !dbg !23
  %241 = and i32 %240, 448, !dbg !23
  %242 = or disjoint i32 %.pre-phi29, %241, !dbg !23
  %243 = lshr exact i32 %17, 2, !dbg !23
  %244 = or disjoint i32 %242, %243, !dbg !23
  %245 = and i32 %20, 508, !dbg !23
  %246 = lshr i32 %244, 2, !dbg !23
  %247 = and i32 %246, 116, !dbg !23
  %248 = add nuw nsw i32 %247, %244, !dbg !23
  %249 = zext nneg i32 %248 to i64, !dbg !23
  %250 = getelementptr inbounds float, ptr addrspace(3) @global_smem, i64 %249, !dbg !23
  %251 = extractelement <4 x i32> %232, i64 0, !dbg !23
  %252 = extractelement <4 x i32> %232, i64 1, !dbg !23
  tail call void asm sideeffect "@$3 st.shared.v2.b32 [ $0 + 0 ], { $1, $2 };", "r,r,r,b"(ptr addrspace(3) %250, i32 %251, i32 %252, i1 true) #3, !dbg !23
  %253 = or disjoint i32 %244, 32, !dbg !23
  %254 = lshr i32 %253, 2, !dbg !23
  %255 = and i32 %254, 1073741820, !dbg !23
  %256 = add nuw nsw i32 %255, %253, !dbg !23
  %257 = zext nneg i32 %256 to i64, !dbg !23
  %258 = getelementptr inbounds float, ptr addrspace(3) @global_smem, i64 %257, !dbg !23
  %259 = extractelement <4 x i32> %232, i64 2, !dbg !23
  %260 = extractelement <4 x i32> %232, i64 3, !dbg !23
  tail call void asm sideeffect "@$3 st.shared.v2.b32 [ $0 + 0 ], { $1, $2 };", "r,r,r,b"(ptr addrspace(3) %258, i32 %259, i32 %260, i1 true) #3, !dbg !23
  tail call void @llvm.nvvm.barrier0(), !dbg !23
  %261 = and i32 %16, 124, !dbg !23
  %262 = add nuw nsw i32 %245, %261, !dbg !23
  %263 = zext nneg i32 %262 to i64, !dbg !23
  %264 = getelementptr inbounds float, ptr addrspace(3) @global_smem, i64 %263, !dbg !23
  %.extract = load i32, ptr addrspace(3) %264, align 16, !dbg !23
  %265 = getelementptr inbounds i8, ptr addrspace(3) %264, i64 4, !dbg !23
  %.extract21 = load i32, ptr addrspace(3) %265, align 4, !dbg !23
  %266 = getelementptr inbounds i8, ptr addrspace(3) %264, i64 8, !dbg !23
  %.extract22 = load i32, ptr addrspace(3) %266, align 8, !dbg !23
  %267 = getelementptr inbounds i8, ptr addrspace(3) %264, i64 12, !dbg !23
  %.extract23 = load i32, ptr addrspace(3) %267, align 4, !dbg !23
  tail call void asm sideeffect "st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l"(i32 %.extract, i32 %.extract21, i32 %.extract22, i32 %.extract23, ptr addrspace(1) %239) #3, !dbg !23
  ret void, !dbg !44
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #2

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nocallback nounwind }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!nvvm.annotations = !{!4, !5}
!llvm.ident = !{!6}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!3 = !DIFile(filename: "test_3D_conv.py", directory: "/home/chan/triton/python/test_corelab")
!4 = !{ptr @bmm_dot_kernel, !"kernel", i32 1}
!5 = !{ptr @bmm_dot_kernel, !"reqntidx", i32 128}
!6 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!7 = distinct !DISubprogram(name: "bmm_dot_kernel", linkageName: "bmm_dot_kernel", scope: !3, file: !3, line: 13, type: !8, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!8 = !DISubroutineType(cc: DW_CC_normal, types: !9)
!9 = !{}
!10 = !DILocation(line: 22, column: 44, scope: !7)
!11 = !DILocation(line: 22, column: 62, scope: !7)
!12 = !DILocation(line: 22, column: 80, scope: !7)
!13 = !DILocation(line: 24, column: 28, scope: !7)
!14 = !DILocation(line: 24, column: 54, scope: !7)
!15 = !DILocation(line: 24, column: 41, scope: !7)
!16 = !DILocation(line: 25, column: 28, scope: !7)
!17 = !DILocation(line: 25, column: 41, scope: !7)
!18 = !DILocation(line: 28, column: 32, scope: !7)
!19 = !DILocation(line: 28, column: 64, scope: !7)
!20 = !DILocation(line: 28, column: 51, scope: !7)
!21 = !DILocation(line: 32, column: 88, scope: !7)
!22 = !DILocation(line: 35, column: 30, scope: !7)
!23 = !DILocation(line: 45, column: 16, scope: !7)
!24 = !DILocation(line: 41, column: 35, scope: !7)
!25 = !DILocation(line: 32, column: 38, scope: !7)
!26 = !DILocation(line: 32, column: 25, scope: !7)
!27 = !DILocation(line: 32, column: 76, scope: !7)
!28 = !DILocation(line: 32, column: 50, scope: !7)
!29 = !DILocation(line: 31, column: 38, scope: !7)
!30 = !DILocation(line: 31, column: 25, scope: !7)
!31 = !DILocation(line: 31, column: 76, scope: !7)
!32 = !DILocation(line: 31, column: 50, scope: !7)
!33 = !DILocation(line: 31, column: 88, scope: !7)
!34 = !DILocation(line: 36, column: 20, scope: !7)
!35 = !DILocation(line: 37, column: 20, scope: !7)
!36 = !DILocation(line: 38, column: 27, scope: !7)
!37 = !DILocation(line: 40, column: 22, scope: !7)
!38 = !DILocation(line: 41, column: 22, scope: !7)
!39 = !DILocation(line: 43, column: 29, scope: !7)
!40 = !DILocation(line: 43, column: 16, scope: !7)
!41 = !DILocation(line: 43, column: 67, scope: !7)
!42 = !DILocation(line: 43, column: 41, scope: !7)
!43 = !DILocation(line: 43, column: 79, scope: !7)
!44 = !DILocation(line: 45, column: 4, scope: !7)
