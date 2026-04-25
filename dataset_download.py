from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Kaiqing/VIPGuard_Stage3_DATA",
    local_dir="FaceDATA/Training_Img"
)