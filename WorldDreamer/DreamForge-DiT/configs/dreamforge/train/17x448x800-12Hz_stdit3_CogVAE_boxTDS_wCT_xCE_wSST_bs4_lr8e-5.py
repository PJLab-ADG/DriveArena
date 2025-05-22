# Dataset settings
num_frames = 17
data_cfg_name = "Nuscenes_vectormap_t"
bbox_mode = 'all-xyz'
img_collate_param_train = dict(
    # template added by code.
    bbox_mode = bbox_mode,
    is_train = True,
    
)

dataset_cfg_overrides = (
    # key, value
    ("dataset.dataset_process_root", "./data/nuscenes_mmdet3d-12Hz_description/"),
    ("dataset.data.train.ann_file", "./data/nuscenes_mmdet3d-12Hz_description/nuscenes_interp_12Hz_updated_description_train.pickle"),
    ("dataset.data.val.ann_file", "./data/nuscenes_mmdet3d-12Hz_description/nuscenes_interp_12Hz_updated_description_val.pickle"),
)

# Runner
dtype = "bf16"
sp_size = 1
plugin = "zero2-seq" if sp_size > 1 else "zero2"
grad_checkpoint = True
batch_size = 1
drop_cond_ratio = 0.15

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16

# Model settings
mv_order_map = {
    0: [5, 1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 0],
}
t_order_map = None

global_flash_attn = True
global_layernorm = True
global_xformers = True
micro_frame_size = None

vae_out_channels = 16

model = dict(
    type="DreamForgeSTDiT3-XL/2",
    qk_norm=True,
    pred_sigma=False,  # CHANGED
    enable_flash_attn=True and global_flash_attn,
    enable_layernorm_kernel=True and global_layernorm,
    enable_sequence_parallelism=sp_size > 1,
    freeze_y_embedder=True,  # CHANGED
    with_temp_block=True,  # CHANGED
    use_x_control_embedder=True,
    enable_xformers = False and global_xformers,
    sequence_parallelism_temporal=False,
    use_st_cross_attn=False,
    uncond_cam_in_dim=(3, 7),
    cam_encoder_cls="dreamforgedit.models.dreamforge.embedder.CamEmbedder",
    cam_encoder_param=dict(
        input_dim=3,
        # out_dim=1152,  # no need to set this.
        num=7,
        after_proj=True,
    ),
    bbox_embedder_cls="dreamforgedit.models.dreamforge.embedder.ContinuousBBoxWithTextTempEmbedding",
    bbox_embedder_param=dict(
        n_classes=10,
        class_token_dim=1152,
        trainable_class_token=False,
        embedder_num_freq=4,
        proj_dims=[1152, 512, 512, 1152],
        mode=bbox_mode,
        minmax_normalize=False,
        use_text_encoder_init=True, 
        after_proj=True,
        sample_id=True,  # CHANGED
        # new
        num_heads=8,
        mlp_ratio=4.0,
        qk_norm=True,
        enable_flash_attn=False and global_flash_attn,
        enable_xformers=True and global_xformers,
        enable_layernorm_kernel=True and global_layernorm,
        use_scale_shift_table=True,
        time_downsample_factor=4.5,
    ),
    map_embedder_cls="dreamforgedit.models.dreamforge.embedder.MapControlEmbedding",
    map_embedder_param=dict(
        conditioning_size=[3, 200, 200],
        block_out_channels=[16, 32, 96, 256],
        # conditioning_embedding_channels=1152,  # no need to set this.
    ),
    map_embedder_downsample_rate=4.5,  # CHANGED
    layout_embedder_cls="dreamforgedit.models.dreamforge.embedder.LayoutControlEmbedding",
    layout_embedder_param=dict(
        conditioning_embedding_channels=1152,
        conditioning_channels=13,
        block_out_channels=[16, 32, 96, 256],
    ),
    layout_embedder_downsample_rate=4.5,  # CHANGED
    # add ope embedder
    ope_embedder_cls="dreamforgedit.models.dreamforge.ope.ObjectPositionEmbedding",
    ope_embedder_param=dict(
        embed_dims=1152,
        LID=False,
        block_out_channels=(256,),
        scale=8,
        output_size=[56, 100],
    ),
    ope_embedder_downsample_rate=4.5,  # CHANGED
    micro_frame_size=micro_frame_size,
    frame_emb_cls="dreamforgedit.models.dreamforge.embedder.CamEmbedderTemp",
    frame_emb_param=dict(
        input_dim=3,
        # out_dim=1152,  # no need to set this.
        num=4,
        after_proj=True,
        # new
        num_heads=8,
        mlp_ratio=4.0,
        qk_norm=True,
        enable_flash_attn=False and global_flash_attn,
        enable_xformers=True and global_xformers,
        enable_layernorm_kernel=True and global_layernorm,
        use_scale_shift_table=True,
        time_downsample_factor=4.5,
    ),
    control_skip_cross_view=True,
    control_skip_temporal=False,
    use_ope = True,
    use_lm_attn = True,
    use_tpe = True,
)


partial_load=""

vae = dict(
    type="VideoAutoencoderKLCogVideoX",
    from_pretrained="./pretrained/CogVideoX-2b",
    subfolder="vae",
    micro_frame_size=micro_frame_size,
    micro_batch_size=1,
)
text_encoder = dict(
    type="t5",
    from_pretrained= "./pretrained/t5-v1_1-xxl",
    model_max_length=300,
    shardformer=True,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    cog_style_trans=True,  # NOTE: trigger error with 9-frame, should change in all cases when frame > 1.
    sample_method="logit-normal",
)

val = dict(
    validation_index=[1828, 4467, 5543],
    batch_size=1,
    verbose=2,
    num_sample=1,
    save_fps=12,
    seed=1024,
    scheduler = dict(
        **scheduler,
        num_sampling_steps=30,
        cfg_scale=2.0,  # base value 1, 0 is uncond
    ),
)

# Mask settings
# 25%
mask_ratios = {
    "random": 0.01,
    "intepolate": 0.002,
    "quarter_random": 0.002,
    "quarter_head": 0.002,
    "quarter_tail": 0.002,
    "quarter_head_tail": 0.002,
    "image_random": 0.0,
    "image_head": 0.22,
    "image_tail": 0.005,
    "image_head_tail": 0.005,
}

# add mask settings
mask_shift = 0.5

# Log settings
seed = 42
outputs = "outputs"
wandb = False
epochs = 10
log_every = 1
ckpt_every = 250 * 5
report_every = ckpt_every

# optimization settings
load = None
grad_clip = 1.0
lr = 8e-5
ema_decay = 0.99
adam_eps = 1e-15
weight_decay = 1e-2
warmup_steps = 3000

#torchrun --nproc-per-node=1--nnode=1 --node_rank=0 --master_addr localhost --master_port 29500 scripts/train_magicdrive.py configs/magicdrive/train/stage2_17x448x800-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr8e-5.py --cfg-options num_workers=2 prefetch_factor=2

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
# torchrun --nproc-per-node=8 --nnode=1 --node_rank=0 --master_addr localhost --master_port 29501 scripts/train_dreamforge_t.py configs/dreamforge/train/17x448x800-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr8e-5.py --cfg-options num_workers=2 prefetch_factor=2 

# load=outputs/MagicDriveSTDiT3-XL-2_stage2_17x448x800-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr8e-5_20250219-1640/ load=/cpfs01/user/yangxuemeng/hutao/DreamForgeDiT/outputs/MagicDriveSTDiT3-XL-2_stage2_17x448x800-12Hz_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_bs4_lr8e-5_20250218-1101/epoch1-global_step3750/