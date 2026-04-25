from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Kaiqing/VIP-Guard",
    local_dir="./checkpoints/checkpoints_attr_Stage2_merge",
    local_dir_use_symlinks=False
)