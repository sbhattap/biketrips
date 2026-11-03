"""Download the public data folder and unzip trips and POIs in place."""
import zipfile

import gdown

from biketrips.config import load_config


def unzip(path, dest):
    with zipfile.ZipFile(path) as z:
        # macOS resource forks ship inside the archives and are not data.
        members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
        z.extractall(dest, members=members)


def main():
    d = load_config()["data"]
    if not d["trips_zip"].exists():
        gdown.download_folder(d["drive_folder"], output=str(d["raw_dir"]), quiet=False)
    if not d["trips_csv"].exists():
        unzip(d["trips_zip"], d["trips_zip"].parent)
    if not d["poi_dir"].exists():
        unzip(d["poi_zip"], d["poi_zip"].parent)
    print("Data ready under", d["raw_dir"])


if __name__ == "__main__":
    main()
