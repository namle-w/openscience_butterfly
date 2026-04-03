import torch
from pathlib import Path

def convert_ckpt_to_pt(ckpt_path: str, output_path: str, model_prefix: str = "model."):
    """
    Load a Lightning checkpoint, strip the LightningModule prefix,
    and save the bare model state_dict to a .pt file.

    Args:
        ckpt_path (str): đường dẫn tới file .ckpt
        output_path (str): nơi lưu file .pt đầu ra
        model_prefix (str): prefix key của model trong checkpoint (thường là "model.")
    """
    # load full checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        raise KeyError(f"No 'state_dict' in checkpoint {ckpt_path}")

    # strip LightningModule name prefix
    new_sd = {}
    for k, v in sd.items():
        if k.startswith(model_prefix):
            new_key = k[len(model_prefix):]
        else:
            new_key = k
        new_sd[new_key] = v

    # đảm bảo folder đầu ra tồn tại
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # save
    torch.save(new_sd, output_path)
    print(f"Saved stripped state_dict to {output_path}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Lightning .ckpt to bare model .pt")
    parser.add_argument("ckpt", help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("out", help="Output path for the .pt file")
    parser.add_argument("--prefix", default="model.",
                        help="Prefix to strip from keys (default: 'model.')")
    args = parser.parse_args()

    convert_ckpt_to_pt(args.ckpt, args.out, model_prefix=args.prefix)
