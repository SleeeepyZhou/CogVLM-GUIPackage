from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="THUDM/cogagent-vqa-hf",
    local_dir="./models/cogagent-vqa-hf",
    max_workers=8
)
