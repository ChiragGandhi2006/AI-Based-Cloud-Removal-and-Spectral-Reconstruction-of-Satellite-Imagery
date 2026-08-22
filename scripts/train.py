"""Train the cloud-removal / spectral reconstruction model."""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import set_seed, get_device
from src.models import build_model
from src.data import SEN12MSCRDataset
from src.training.trainer import Trainer


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train cloud-removal model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--checkpoint-path", default=None, help="Checkpoint output path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    set_seed(args.seed)

    cfg = load_config(os.path.join(root, args.config))
    model_cfg = cfg.get("model", {})
    train_cfg = dict(cfg.get("training", {}))
    dataset_cfg = cfg.get("dataset", {})

    train_cfg["device"] = get_device(args.device)
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
    if args.checkpoint_path:
        train_cfg["checkpoint_path"] = args.checkpoint_path

    if "learning_rate" not in train_cfg:
        train_cfg["learning_rate"] = cfg.get("training", {}).get("learning_rate", 0.001)

    device = get_device(args.device)
    print(f"Device: {device}")

    model = build_model(
        model_cfg.get("name", "attention_unet"),
        in_channels=model_cfg.get("in_channels", 13),
        out_channels=model_cfg.get("out_channels", 13),
        base_filters=model_cfg.get("base_filters", 32),
        use_sar=model_cfg.get("use_sar", True),
        use_mask=model_cfg.get("use_mask", False),
    )
    model.model_config = {
        "name": model_cfg.get("name", "attention_unet"),
        "in_channels": model_cfg.get("in_channels", 13),
        "out_channels": model_cfg.get("out_channels", 13),
        "base_filters": model_cfg.get("base_filters", 32),
        "use_sar": model_cfg.get("use_sar", True),
        "use_mask": model_cfg.get("use_mask", False),
    }
    print(f"Model: {type(model).__name__}")

    train_path = os.path.join(root, dataset_cfg.get("train", "data/processed/train"))
    val_path = os.path.join(root, dataset_cfg.get("val", "data/processed/val"))
    test_path = os.path.join(root, dataset_cfg.get("test", "data/processed/test"))

    train_ds = SEN12MSCRDataset(root, split="train") if os.path.isdir(train_path) else None
    val_ds = SEN12MSCRDataset(root, split="val") if os.path.isdir(val_path) else None
    test_ds = SEN12MSCRDataset(root, split="test") if os.path.isdir(test_path) else None

    if train_ds is None or len(train_ds) == 0:
        print("No training data found. Run scripts/preprocess_dataset.py first "
              "(or place processed patches under data/processed/).")
        sys.exit(1)

    print(f"Train samples: {len(train_ds)}")
    if val_ds is not None:
        print(f"Val samples:   {len(val_ds)}")

    trainer = Trainer(model, train_ds, val_ds, test_ds, train_cfg)
    trainer.fit(epochs=train_cfg.get("epochs", 100))


if __name__ == "__main__":
    main()