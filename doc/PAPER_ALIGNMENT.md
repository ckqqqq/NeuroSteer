# Paper and implementation alignment

## Scope

The maintained path is SAE-based language-model feature intervention and interpretability analysis. Feature-level interventions are useful for circuit-oriented investigation but are not a complete circuit-discovery algorithm. Supported behavioral tasks are sentiment, toxicity, stance (`debate` in code) and politeness.

## Recovered artifacts

The four `results/demo/demo_v1/**/…neuron_info_cache_of_gpt2-small_l6.pkl` files were added in commit `fb0a0c9` and deleted in `a5bac2d`. They were recovered byte-for-byte from `fb0a0c9`, not recomputed. Each contains `pos`/`neg` feature frequency and mean-activation statistics. Full GPT-2 and SAE weights are loaded separately using the original public model identifiers.

## Important implementation distinctions

- `src/denoising.py` applies ReLU to target-minus-source activation frequency and target-minus-source mean, then takes Top-K frequency indices. Its means are differences, not an unspecified positive-class mean.
- `SAE_decoding` constructs an additive direction by summing selected decoder rows times the selected feature values. It does not add a decoder bias.
- The demo passes `is_norm=0`, sums decoded task directions and applies slider coefficients in residual space. Its sliders are not the paper's normalized latent coefficient. Do not silently reinterpret historical values as the updated paper formula.
- `steering_hook` currently returns without intervention when the sequence length is one. The `last` option edits `[:, :-1, :]`, while `all` edits the current sequence. These historical semantics are preserved rather than changed beneath old experiment labels.
- Baseline and steered requests use stochastic generation. A side-by-side difference alone is not evidence of a causal effect; controlled seeds and repeated sampling are needed for stronger comparisons.
- The loader follows each SAE's recorded model-preprocessing options via `from_pretrained_no_processing`, avoiding unrelated TransformerLens folding defaults. This fixes a compatibility issue in the historical demo.
- Gemma uses a different SAE release and matching tokenizer. Recovered GPT-2 task caches cannot be reused for Gemma.

## Remaining reproducibility checks

The original training dataset splits, SAE training-loss weight interpretation, PPL evaluation model, and timing denominator still require original configurations/logs. Existing evaluation scripts may require external data or APIs. No claim is made that all paper tables were reproduced by this maintenance work.

## Removed unfinished material

Removed `src/multi_media/`, the multimodal sentiment notebook, the empty failed-exploration note, CoT launchers targeting missing `test_llama.py`, the alpha-search launcher targeting missing `debate_test_0119.py`, the print-only path script, and the placeholder `test_unit.py`. Removed unreachable Llama and Gemma-1 loader branches. Retained historical results, layer-analysis notebooks and the linear-combination notebook.
