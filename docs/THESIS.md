# ExactKV

Lossy KV-cache compression. Exact full-KV outputs.

## Vision

ExactKV is a compressor-agnostic verification framework for KV-cache compression.

Modern LLM inference is increasingly bottlenecked by KV-cache memory usage. Numerous compression techniques have emerged, including quantization, token dropping, eviction, and reconstruction-based methods.

While these methods improve throughput and memory efficiency, they introduce output divergence from full-KV decoding.

ExactKV aims to solve this problem.

Instead of trusting compressed KV outputs directly, ExactKV treats compressed inference as a draft generation process and verifies drafted tokens against full-KV decoding before committing them.

The result is:

* Reduced active KV memory
* Increased throughput
* Exact output equivalence to full-KV decoding

## Core Insight

Existing systems force users to choose between:

* Full KV

  * Accurate
  * Expensive

* Compressed KV

  * Fast
  * Potentially incorrect

ExactKV introduces a third option:

* Compressed KV
* Verification Layer
* Exact Outputs

## Primary Goal

Create an open-source runtime that allows any KV-cache compression strategy to be evaluated and deployed with correctness guarantees.

## Secondary Goal

Create the first compressor-agnostic benchmark suite for KV-cache compression quality.

## Inspiration

The project is heavily inspired by the VeriCache paper:

"VeriCache: Turning Lossy KV Cache into Lossless LLM Inference"

However, ExactKV is not a paper reimplementation.

ExactKV aims to become a platform capable of evaluating and verifying multiple KV compression strategies.

## Long-Term Vision

ExactKV should become:

"PyTorch Lightning for KV Compression"

Users should be able to swap compressors while retaining identical output behavior.

## Success Criteria

* Exact output equivalence to full-KV decoding
* Support for multiple compressors
* Hugging Face integration
* Benchmark suite
* Reproducible performance results
* Clean API
