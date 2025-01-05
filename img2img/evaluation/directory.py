def get_last_version(output_dir):
    existing_versions = [int(d.name.split("_")[-1]) for d in output_dir.glob("version_*")]
    return max(existing_versions)

def get_last_epoch(checkpoint_dir):
    existing_epochs = [
        int(f.name.split(".")[0].split("=")[1]) 
        for f in checkpoint_dir.glob("*.ckpt") 
        if f.name.split(".")[0].split("=")[0]=='epoch'
    ]
    return max(existing_epochs)