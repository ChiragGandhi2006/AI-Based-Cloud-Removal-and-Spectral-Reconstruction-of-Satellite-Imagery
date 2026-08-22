"""Download the SEN12MS-CR dataset (placeholder - dataset must be downloaded manually)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import SEN12MSDownloader


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    target_dir = os.path.join("data", "raw", "SEN12MS-CR")
    os.makedirs(target_dir, exist_ok=True)

    SEN12MSDownloader.download()
    print(f"\nExpected dataset location: {os.path.abspath(target_dir)}")
    print("The SEN12MS-CR dataset is large (~500 GB split subsets) and requires")
    print("registration on https://mediatum.ub.tum.de/1474000 before it can be")
    print("downloaded programmatically. See docs/DATASETS.md for details.")


if __name__ == "__main__":
    main()