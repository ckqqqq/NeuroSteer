# Recovered demo task statistics

Source commit: `fb0a0c9`; removed by `a5bac2d`. Files restored byte-for-byte, without recomputing statistics.

| Task | Bytes | SHA-256 |
| --- | ---: | --- |
| debate | 394554 | `3c9ec89374e8ef7bdbaf4f8fcfadafc701adf63446752c5cbf43ddbd5eb28368` |
| polite | 394554 | `37d1c4b0372c9b3252b9066bcc60529d540e60b961b2a0d7950a5b0d5b408241` |
| sentiment | 394554 | `c567d6671958a67c6482cd3efba2823ea63ae26e4a3fa99b150ef170fea39abd` |
| toxicity | 394554 | `9f86b4ddbe3a88df1865b9914961b9aea14ddb3e2478918b3bb37ee8bbf4c82a` |

These are activation-statistics caches, not full GPT-2/SAE checkpoints. They contain PyTorch tensor serializations recorded on CUDA device 0. On DGX Spark, all four files were deserialized successfully on CUDA; all frequency/mean vectors had shape (24576,) and finite values. Baseline, eight directions and a combined real-generation smoke test passed.
