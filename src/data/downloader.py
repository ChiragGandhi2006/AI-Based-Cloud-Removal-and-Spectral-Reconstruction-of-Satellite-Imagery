import os
import urllib.request
import zipfile


class SEN12MSDownloader:
    dataset_url = "https://example.com/SEN12MS-CR.zip"
    extract_dir = "data/raw/SEN12MS-CR"

    @staticmethod
    def download():
        print("Downloading SEN12MS-CR dataset...")
        # urllib.request.urlretrieve(SEN12MSDownloader.dataset_url, "SEN12MS-CR.zip")
        print("Download complete. Please extract to data/raw/SEN12MS-CR/")

    @staticmethod
    def extract():
        zip_path = "SEN12MS-CR.zip"
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(SEN12MSDownloader.extract_dir)
            print("Extraction complete.")
        else:
            print("Zip file not found. Please download manually.")