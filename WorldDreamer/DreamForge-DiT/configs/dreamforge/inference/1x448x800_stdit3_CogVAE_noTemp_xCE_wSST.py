frame_interval = 1
validation_index = [204, 912, 1828, 2253, 4467, 5543]
num_sample = 1

batch_size = 1
dtype = "bf16"

scheduler = dict(
    type="rflow-slice",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=2.0,  # base value 1, 0 is uncond
)


# Dataset settings
num_frames = 1
data_cfg_name = "Nuscenes_vectormap_t_val"
bbox_mode = 'all-xyz'
img_collate_param_train = dict(
    # template added by code.
    bbox_mode = bbox_mode,
    with_temporal_dim = True # add temporal axis for single-frame version
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
grad_checkpoint = False
# batch_size = 2
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
    freeze_y_embedder=False,  # CHANGED
    # dreamforge
    with_temp_block=False,  # CHANGED
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
        # new
        num_heads=8,
        mlp_ratio=4.0,
        qk_norm=True,
        enable_flash_attn=False and global_flash_attn,
        enable_xformers=True and global_xformers,
        enable_layernorm_kernel=True and global_layernorm,
        use_scale_shift_table=True,
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
    ope_embedder_cls="dreamforgedit.models.dreamforge.ope.ObjectPositionEmbedding",
    ope_embedder_param=dict(
        embed_dims=1152,
        LID=False,
        ori_shape=[1600, 900],
        gen_shape=[400, 224],
        block_out_channels=(256, 256),
        scale=4,
        output_size=[56, 100],
    ),
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
    ),
    control_skip_cross_view=True,
    control_skip_temporal=True,
    use_ope=True,
    # load pretrained
    # from_pretrained="./pretrained/hpcai-tech/OpenSora-STDiT-v3",
    # force_huggingface=True,  # if `from_pretrained` is a repo from hf, use this.
)
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
    # shardformer=True,
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

# Log settings
seed = 123
outputs = "outputs/test/CogVAE"
wandb = False
epochs = 150
log_every = 1
ckpt_every = 500 * 5

# optimization settings
load = None
grad_clip = 1.0
lr = 2e-5
ema_decay = 0.99
adam_eps = 1e-15
weight_decay = 1e-2
warmup_steps = 500
