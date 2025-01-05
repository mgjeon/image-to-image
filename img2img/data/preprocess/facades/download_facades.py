from pathlib import Path
from urllib.request import urlretrieve
import tarfile
from tqdm import tqdm
import argparse

# https://github.com/tqdm/tqdm?tab=readme-ov-file#hooks-and-callbacks
class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="./datasets")
    args = parser.parse_args()

    DATAROOT = Path(args.dataroot)
    DATAROOT.mkdir(exist_ok=True, parents=True)

    filename = DATAROOT / "facades.tar.gz"

    if not filename.exists():
        url = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                    desc=url.split('/')[-1]) as t:  # all optional kwargs
            urlretrieve(url, filename=filename,
                        reporthook=t.update_to, data=None)
            t.total = t.n

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(DATAROOT)

    DATASET = DATAROOT / "facades"

    TRAIN_PATH = DATASET / "train"
    VAL_PATH = DATASET / "val"
    TEST_PATH = DATASET / "test"

    train_files = sorted(TRAIN_PATH.glob("*.jpg"))
    val_files = sorted(VAL_PATH.glob("*.jpg"))
    test_files = sorted(TEST_PATH.glob("*.jpg"))

    print("TRAIN:", len(train_files))
    print("VAL  :", len(val_files))
    print("TEST :", len(test_files))

    # data_split_root = Path("./datasets/facades")
    data_split_root = DATAROOT / "facades_split"
    data_split_root.mkdir(exist_ok=True, parents=True)

    train_split_A = data_split_root / "train/A"
    train_split_B = data_split_root / "train/B"
    train_split_A.mkdir(exist_ok=True, parents=True)
    train_split_B.mkdir(exist_ok=True, parents=True)

    val_split_A = data_split_root / "val/A"
    val_split_B = data_split_root / "val/B"
    val_split_A.mkdir(exist_ok=True, parents=True)
    val_split_B.mkdir(exist_ok=True, parents=True)

    test_split_A = data_split_root / "test/A"
    test_split_B = data_split_root / "test/B"
    test_split_A.mkdir(exist_ok=True, parents=True)
    test_split_B.mkdir(exist_ok=True, parents=True)

    from skimage.io import imread, imsave

    for file in tqdm(train_files):
        img = imread(file)
        photo = img[:, :256]
        label = img[:, 256:]
        
        imsave(train_split_A / file.name, label)
        imsave(train_split_B / file.name, photo)

    for file in tqdm(val_files):
        img = imread(file)
        photo = img[:, :256]
        label = img[:, 256:]
        
        imsave(val_split_A / file.name, label)
        imsave(val_split_B / file.name, photo)

    for file in tqdm(test_files):
        img = imread(file)
        photo = img[:, :256]
        label = img[:, 256:]
        
        imsave(test_split_A / file.name, label)
        imsave(test_split_B / file.name, photo)

    # import shutil
    # shutil.rmtree(DATASET)
