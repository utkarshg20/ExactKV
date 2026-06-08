# Research Background

## Problem

KV-cache memory is a major bottleneck in LLM serving.

As context length increases:

Memory ∝ Context Length

Large context windows can consume significant GPU memory.

## Existing Solutions

### Quantization

Examples:

* KIVI
* KVQuant
* TurboQuant

Advantages:

* Significant memory reduction

Disadvantages:

* Output drift

### Token Dropping

Examples:

* SnapKV
* H2O
* FastGen

Advantages:

* Lower memory

Disadvantages:

* Potential information loss

### Offloading

Examples:

* LMCache
* CPU KV
* SSD KV

Advantages:

* Increased capacity

Disadvantages:

* Transfer latency

## VeriCache

Key idea:

Compressed KV performs draft generation.

Full KV verifies drafted tokens.

Only verified tokens are committed.

Result:

Lossless outputs with compressed inference.

## ExactKV Positioning

ExactKV extends this idea by:

* Supporting multiple compressors
* Providing benchmarking
* Providing a unified interface
* Providing developer tooling
