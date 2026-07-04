# Future Scene Generation Challenge Prototype

The prototype adds a one-case, camera-first challenge pipeline without changing
the original WorldDreamer training or test entrypoints.

Implemented stages:

- Extract 60 history frames and 180 future GT frames from 12Hz nuScenes infos.
- Keep future GT under `gt_future_for_eval_only`.
- Estimate initial ego and agent states from the last 1-2 seconds of history.
- Run a LimSim adapter with a kinematic fallback when no stable LimSim API is available.
- Bridge history and rollout states over the first 1.5 seconds.
- Convert bridged rollout states into per-frame WorldDreamer condition JSON.
- Run no-leakage inference into camera-first prediction folders.
- Evaluate PSNR/SSIM and create diagnostic grids/BEV images.

Current limitations:

- The default inference backend is `auto`: it tries the WorldDreamer single-frame
  autoregressive backend when dependencies and checkpoint paths are usable, then
  falls back to a deterministic prototype renderer if needed.
- The fallback rollout is kinematic and history-only; it does not consume future GT.
- The map export is a placeholder BEV file. WorldDreamer map cache generation can
  replace `input/map_bev.png` and `input/static_map.json`.
- The prototype is scoped to one case, but all scripts accept case paths and can
  be wrapped for train/val/test splits.
- A true 5s history-conditioned video model is not trained here; the baseline is
  single-frame autoregressive reference conditioning.
