# assets/

Figures from both open-weight teacher pipelines (Llama-3.1-8B and Qwen2.5-7B+aug).

## Llama-3.1-8B pipeline (35-tool, 9,903 samples)

| File | Description |
|---|---|
| `training_curves_llama.png` | Training & validation loss + accuracy over 80 epochs |
| `latency_comparison_llama.png` | Routing latency: KAN-Route vs MLP vs BART-MNLI vs Phi-2 (Kaggle T4) |
| `spline_analysis_object_counting.png` | Spline activations — "Count how many bananas are on the dining table" → `object_counting` ✓ (96.0%) |
| `spline_analysis_action_recognition.png` | Spline activations — "What is the person in the image doing?" → `action_recognition` ✓ (96.8%) |
| `spline_analysis_background_removal.png` | Spline activations — "Remove the background and isolate the subject" → `background_removal` ✓ (95.0%) |

## Qwen2.5-7B + paraphrase augmentation (35-tool, 39,400 samples)

| File | Description |
|---|---|
| `training_curves_qwen_aug.png` | Training & validation loss + accuracy over 80 epochs (KAN final loss 0.551 vs MLP 0.646) |
| `latency_comparison_qwen_aug.png` | Routing latency: KAN-Route (0.64ms) vs MLP (0.48ms) vs Qwen2.5-7B (213.9ms, 334× slower) |
| `spline_analysis_qwen_object_counting.png` | Spline activations — "Count how many bananas are on the dining table" → `object_counting` ✓ (89.6%) |
| `spline_analysis_qwen_action_recognition.png` | Spline activations — "What is the person in the image doing?" → `action_recognition` ✓ (89.3%) |
| `spline_analysis_qwen_background_removal_error.png` | **Failure case** — "Remove the background and isolate the subject" → predicted `instance_segmentation` ✗ (31.4%), expected `background_removal`. Illustrates semantic proximity between segmentation and removal tools. |

To regenerate programmatically, use `scripts/plot_spline_activations.py` and `scripts/benchmark_latency.py`.
