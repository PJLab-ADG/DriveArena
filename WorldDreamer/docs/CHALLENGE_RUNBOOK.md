# Challenge Prototype Runbook

Example paths for this workstation:

```bash
NUSC_ROOT=/media/yuhang/35d64c58-50a6-4088-bb4e-6a2cea99c35b/Nuscenes-data/nuscenes
PRETRAINED_ROOT=/media/yuhang/35d64c58-50a6-4088-bb4e-6a2cea99c35b/Nuscenes-data/base-pretrained
CKPT=$PRETRAINED_ROOT/dreamer_pretrained/SDv1.5_mv_single_ref_nus/weight-S200000
```

Extract one 20s case:

```bash
conda run -n dreamer python WorldDreamer/tools/extract_challenge_case.py \
  --nusc-root "$NUSC_ROOT" \
  --case-id case_000001 \
  --history-seconds 5 \
  --future-seconds 15 \
  --fps 12 \
  --out-dir ./challenge_data
```

Simulate future rollout and conditions:

```bash
conda run -n dreamer python WorldDreamer/tools/simulate_future_limsim.py \
  --case-dir ./challenge_data/case_000001 \
  --future-seconds 15 \
  --fps 12 \
  --bridge-seconds 1.5 \
  --use-limsim true \
  --fallback-kinematic-if-needed true
```

Render predictions:

```bash
conda run -n dreamer python WorldDreamer/tools/challenge_infer.py \
  --case-dir ./challenge_data/case_000001 \
  --mode no_leakage_limsim \
  --resume-from-checkpoint "$CKPT" \
  --pretrained-root "$PRETRAINED_ROOT" \
  --fps 12 \
  --num-future-frames 180 \
  --output-dir-name basedreamer_limsim_no_leakage
```

For fast smoke tests, add `--backend prototype --debug-num-frames 12`.

Check leakage, evaluate, and visualize:

```bash
conda run -n dreamer python WorldDreamer/tools/check_no_future_leakage.py \
  --case-dir ./challenge_data/case_000001 \
  --pred-dir ./challenge_data/case_000001/outputs/basedreamer_limsim_no_leakage

conda run -n dreamer python WorldDreamer/tools/evaluate_challenge_case.py \
  --gt-dir ./challenge_data/case_000001/gt_future_for_eval_only/images \
  --pred-dir ./challenge_data/case_000001/outputs/basedreamer_limsim_no_leakage \
  --num-frames 180 \
  --camera-first true \
  --out-json ./challenge_data/case_000001/diagnostics/metrics.json

conda run -n dreamer python WorldDreamer/tools/visualize_history_sim_gap.py \
  --case-dir ./challenge_data/case_000001

conda run -n dreamer python WorldDreamer/tools/visualize_challenge_case.py \
  --case-dir ./challenge_data/case_000001 \
  --pred-dir ./challenge_data/case_000001/outputs/basedreamer_limsim_no_leakage \
  --num-frames 180 \
  --camera-first true
```
