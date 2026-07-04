# Challenge Data Format

This prototype uses a camera-first layout everywhere:

```text
input/history_images/{camera}/{frame_id:06d}.jpg
gt_future_for_eval_only/images/{camera}/{frame_id:06d}.jpg
outputs/basedreamer_limsim_no_leakage/{camera}/{frame_id:06d}.jpg
```

History frames are `000000` to `000059`; future frames are `000000` to
`000179`. Future GT is stored only under `gt_future_for_eval_only` and is not an
allowed inference source.

`input/manifest.json` records the case id, source scene, frame counts, camera
order, path patterns, allowed inference inputs, and forbidden GT future paths.
