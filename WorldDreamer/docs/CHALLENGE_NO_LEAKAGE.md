# Challenge No-Leakage Rules

Inference may read:

- `input/history_images`
- `input/history_ego_states.json`
- `input/history_boxes.json`
- `input/camera_params.json`
- `input/static_map.json`
- `input/map_bev.png`
- `input/scene_description.txt`
- `sim_future/limsim_rollout_bridged.json`
- `sim_future/worlddreamer_conditions`

Inference must not read:

- `gt_future_for_eval_only/images`
- `gt_future_for_eval_only/ego_states.json`
- `gt_future_for_eval_only/boxes.json`
- `gt_future_for_eval_only/can_bus.json`

Run `WorldDreamer/tools/check_no_future_leakage.py` after simulation and
inference. It validates condition sources and reference image paths and writes
`sim_future/no_leakage_report.json`.
