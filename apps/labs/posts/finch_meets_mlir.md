---
title: "Finch meets MLIR"
authors: [yu-sheng-aow]
published: August 26, 2026
description: 'A recap of my Quansight internship working on Finch: MLIR and Asarray implementation'
category: [Beyond PyData, Internship]
featuredImage:
  src: /posts/finch-meets-mlir/finch-hero.png
  alt: 'Pixel-art finch represented as a grid of numbered cells.'
hero:
  imageSrc: /posts/finch-meets-mlir/finch-hero.png
  imageAlt: 'Pixel-art finch represented as a grid of numbered cells.'
---

## Introduction

This summer, I had the opportunity to work on [finch-tensor](https://github.com/finch-tensor/finch-tensor) for my internship. I was given a task to implement a brand new backend to the compiler, namely [MLIR](https://mlir.llvm.org/) or Multi-Level IR Compiler Framework, and implement some technical work that bridges Scipy arrays to Finch tensors. Before I dive deep into the technical jargon, here is a short story about an undergraduate student finding his footing in the open-source community.

During my freshman year, I took my first Python introductory course. It didnt leave a big impression on me. The class revolves around writing code by hand, and that was a miserable experience. It was only when I took an algorithm course the very next semester, I started realizing the complexity yet beautiful nature of the computer science field. During my sophomore year, I start cold-emailing professors regarding my interest to contribute to their research preferably algorithms. One of my professors reached back out to me telling me that there was a new faculty coming into the college that is doing interesting work in sparse tensors. I was thrilled to be a part of this research. I didnt know it at that time but that decision was going to change my life as that was my first exposure to the open-source community.

Being part of an open-source community felt very new and eye-opening to me. Having so many bright minds working on the same project collaboratively while joining from different time zones. Contributing to this community also introduced new challenges I have never faced previously from filing bug issues on github that are read by the community and making detailed pull request for review. I had to learn to communcate my ideas and contributions clearly on the development meetings. That was where I met [Mateusz Sokół](https://github.com/mtsokol), who became my mentor for the Quansight Internship Program, thus leading to the creation of this blog post.

For my first task in the internship, I am tasked to determine the feasibility of lowering Finch Assembly IR (Intermediate Representation) to MLIR. Lets get right into it!

## Understanding MLIR

Coming into this project, I had no clue what MLIR was. I had to dig through every MLIR documentation and understand throughly what a dialect was. To my best understanding, a dialect is a specialized IR that allows MLIR to express control over every layer of execution. There were plenty of dialects in MLIR, but this were some of the unique ones that I have identified:

- [gpu](https://mlir.llvm.org/docs/Dialects/GPU/) dialect provides a middle level abstraction for launching GPU kernels.
- [Tensor Operator Set Architecture(TOSA)](https://mlir.llvm.org/docs/Dialects/TOSA/#tosa-and-tensor-level-expressiveness) dialect represent tensor computations common in machine learning.
-  [XeGPU](https://mlir.llvm.org/docs/Dialects/XeGPU/) dialect provides GPU abstractions for Intel Xe GPUs.

The accessibiltiy of MLIR is limitless, allowing programs to extend to GPU and even the most poignant hardware like Intel Xe GPUs, and each dialects are easily utilised with easy, and composable features.

To circle back to my project, I had to determine the feasibility of lowering Finch Assembly IR to MLIR. Finch Assembly IR is a backend-agnostic IR in finch-tensor that allows code generation lowering to any specific backend. Currrently, finch-tensor has two backends: C and Numba. MLIR would be vastly different from the two other backends, as Numba was written in Python while C was written in C. Implementing MLIR would take a great understanding of the MLIR language and its capability. Does MLIR allows the lowering of Finch Assembly IR considering its unique semantics and assumption?

My first task in the internship would be drafting up a hand-written MLIR kernel with a minimal set of dialects usage to match the Finch Assembly IR in finch-tensor. We deemed that Dense Matrix Multiplication, Sparse Matrix Multiplication and Sampled Dense-Dense Matrix Multiplication(SDDMM) would be the kernels we try to lower to MLIR.

These were the dialects that were utilised in our lowering:

- [arith](https://mlir.llvm.org/docs/Dialects/ArithOps/) dialect for the mathematical operations (e.g add and mul)
- [memref](https://mlir.llvm.org/docs/Dialects/MemRef/) dialect for buffers
- [scf](https://mlir.llvm.org/docs/Dialects/SCFDialect/) dialect to represent control flow constructs. (e.g for and while)
- [llvm](https://mlir.llvm.org/docs/Dialects/LLVM/) dialect to represent ctypes struct passed into the kernel
- [builtin](https://mlir.llvm.org/docs/Dialects/Builtin/) dialect to convert llvm.struct into memref
- [func](https://mlir.llvm.org/docs/Dialects/Func/) dialect to define mlir functions

## Arith dialect

Tackling this dialect was tougher than it initially seemed. In MLIR, all values within an operation must share a common type, and that type determines which operation is selected. An example of a function can be seen here:

```python
    case ffuncs.min:
        # Floating-point minimum.
        if MLIROperator.is_float(arg):
            return "arith.minimumf"

        # Unsigned-integer minimum.
        if MLIROperator.is_unsigned(arg):
            return "arith.minui"

        # Signed-integer minimum.
        return "arith.minsi"
```

This is an example of a failed-compiled code:
```text
// Create a floating-point constant.
%0 = arith.constant 0.0 : float

// Create an index constant.
%1 = arith.constant 1 : index

// Fails due to typing inconsistencies.
%mul = arith.minsi %0, %1 : index
```

This is the right compiled code:
```text
// Create a floating-point constant.
%0 = arith.constant 0.0 : float

// Create a floating-point constant.
%1 = arith.constant 1.0 : float

// Compiles.
%mul = arith.minimumf %0, %1 : float
```

or we could use index_cast to convert the typing of the SSA:
```text
// Create a floating-point constant.
%0 = arith.constant 0.0 : float

%1 = arith.constant 1 : index
%new_1 = arith.index_cast %1 : index to float

// Performs min operation.
%mul = arith.minimumf %0, %new_1 : float
```


MLIR express values in the form of SSA (Static Single Assignment), where every value is assigned once and remains immutable. Chain operations was prevalent in Finch Assembly IR as seen here:

```text
#i#41__pos: int64 = add(0, mul(#_A_3#38_stride_0, #i#41))
```
In MLIR, we would have to introduce a unique SSA for every operation and constant, which would look like this:

```text
// Create a unique SSA for each individual value.
%i_41 = arith.constant 1 : i64
%A_3_38_stride_0 = arith.constant 1 : i64
%0 = arith.constant 0 : i64

// Create a new SSA for multiplication.
%mul = arith.muli %_i_41, %A_3_38_stride_0 : i64

// Create another SSA for the result.
%result = arith.addi %mul, %0 : i64
```

## Buffer Handling

<p align="center">
  <img src="/posts/finch-meets-mlir/iceberg.png" width="300" alt="tip of the iceberg meme">
</p>


Operation handling was just the tip of the iceberg; buffer management gets even more technical and complicated. It requires an in-depth compiler theory knowledge: ABI lowering, memory management and memory ownership. Passing buffers from an IR to another language requires complicated serializing and deserializing logic to pass buffers' metadata (i.e strides and shapes) otherwise the target language would not be able to extract any metadata from the struct passed into the kernel.

First and foremost, I would like to talk about how buffers work in finch-tensor. Currently, buffers in finch-tensor are one-dimensional, resizable block of memory and appears as leaf fields at the ElementLevel. Level are nested into chains, with each level representing the format of one dimension. Dense levels provides their dimension's size and strides, while sparse levels does the same but adds integer buffers (ptr and idx), that record the tensor's nonzero entries. FiberTensor wraps the outermost level, accompanying it with the tensor's shape, pos and dirty-bit.

```text
Dense Matrix Construction:

FiberTensor
└── DenseLevel
    └── DenseLevel
        └── ElementLevel


Sparse CSR Matrix construction:

FiberTensor
└── DenseLevel
    └── SparseListLevel
        └── ElementLevel
```

To sucessfully lower Finch Assembly IR buffers, the MLIR backend recursively lowers each StructFtype to a nested llvm struct, and unpack it in the body with extractvalue indexing. The llvm.structs are type casted to memref with builtin.unrealized_conversion_cast and integer scalars are type casted with arith.index_cast.

- This is an example of a dense matrix being typecasted to memref:

    ```text
    %v = llvm.extractvalue %_A_15[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    ```
    We could see that the dense matrix is being indexed to the ElementLevel with "%_A_15[0, 0, 0, 0]" to extract the buffer. The buffer is then typecasted to "memref<?xf64>" as memref.store and memref.load can only be performed on memref buffers as seen here:

    ```text
    %v_52 = memref.load %v_2[%v_37] : memref<?xf64>
    ```

On top of that, the function parameters and the return of the structs have to be redefined in the lowering.

Every func.func MLIR creates are tagged with llvm.emit_c_interface so that MLIR generates a C-ABI wrapper. The kernel builds a ctypes.Structure from the StructFType, and passes a pointer to that struct as an argument to the compiled kernel.

As structs cannot be returned through a normal return register, the caller allocates space for the return struct, and pass a pointer to the kernel as an extra argument. The kernel write the finished result struct into %_ret using llvm.store. Afterwards, the kernel reconstructs the struct into Finch tensor objects, which the compute function returns. Currently, buffer cannot be allocated or resized in MLIR; it only reads and writes to them, so %_ret just carries back a copy of a descriptor the caller already has. Once buffer allocation buffer is supported, the kernel would be able to emit memref.alloc and memref.realloc, and ptr / idx / val buffer would come back through the struct stored to %_ret.

- This is an example of func.func for a dense matmul kernel

    ```text
    func.func @main(%_A_15: ..., %_A_16: ..., %_A_7: ..., %_ret: !llvm.ptr) attributes {llvm.emit_c_interface}
    ```
   The main function takes in four arguments: two Dense(Dense(Element)) matrices (%_A_15, %_A_16), one BufferizedNDArray dense output matrix (%_A_7) and %_ret, where the result struct is stored into.

   ```text
    %v_56 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_57 = llvm.insertvalue %_A_7, %v_56[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_57, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
   ```
   This is the method used to return the result structs at the end of the kernel. We create an llvm struct type with llvm.mlir.undef of a tuple containing one BufferizedNDArray. Next, we store the newly-created struct in the %_ret pointer with llvm.store, and then end the kernel with func.return.