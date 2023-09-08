import argparse
import os
import os.path
from pathlib import Path

import joblib
import torchvision
import webdataset as wds
from tqdm import tqdm

from image_loader import MyDataset

# memory explosion??
ALL_KEYS = set()


TRANSFORMS = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToPILImage()]
)


def ensure_dir(path) -> None:
    path = Path(path)
    if path.is_file():
        path = path.parent
    path.mkdir(exist_ok=True, parents=True)


def parse_args():
    parser = argparse.ArgumentParser("""Generate sharded dataset from dataframe.""")
    parser.add_argument("--splits", default="train, val", help="which splits to write")
    parser.add_argument("--filekey", action="store_true", help="use file as key (default: index)")
    parser.add_argument("--maxsize", type=float, default=1e9, help="Maximum size per shard")
    parser.add_argument(
        "--maxcount", type=float, default=100000, help="Maximum number of records per shard"
    )
    parser.add_argument(
        "--shards", type=Path, required=True, help="directory where shards are written"
    )
    parser.add_argument("--dataframe", type=Path, required=True, help="path to dataset dataframe")
    args = parser.parse_args()

    assert args.maxsize > 10000000
    assert args.maxcount < 1000000

    return args


def readfile(fname):
    "Read a binary file from disk."
    with open(fname, "rb") as stream:
        return stream.read()


def write_dataset(dataframe_path, output_path="./shards", split="train", maxsize=1e9, maxcount=100000):
    dataset = MyDataset(
        dataframe_path=dataframe_path, split=split, transform=TRANSFORMS, include_path=True
    )

    nimages = len(dataset)
    print("# nimages", nimages)

    # This is the output pattern under which we write shards.
    output_pattern = os.path.join(output_path, f"inspection-{split}-%06d.tar")

    with wds.ShardWriter(output_pattern, maxsize=int(maxsize), maxcount=int(maxcount)) as sink:
        count = 0
        for image, targets, img_path in tqdm(dataset, total=len(dataset)):
            count += 1

            # just write first N
            if count > 20000:
                break

            # Construct a unique key from the filename.
            xkey = os.path.join(*img_path.split("/")[-2:]).replace(".jpg", "")
            # Useful check.
            assert xkey not in ALL_KEYS, xkey
            ALL_KEYS.add(xkey)

            # NOTE: don't include .jpg" in xkey, as it infers the encoder type from it

            # Construct a sample.
            sample = {"__key__": xkey, "jpg": image, "npy": targets}

            # Write the sample to the sharded tar archives.
            sink.write(sample)


if __name__ == "__main__":
    args = parse_args()

    splits = args.splits.split(",")

    ensure_dir(args.shards)

    with joblib.parallel_backend("multiprocessing", n_jobs=len(splits)):
        joblib.Parallel(verbose=100)(
            joblib.delayed(write_dataset)(
                args.dataframe,
                output_path=args.shards,
                split=split,
                maxsize=args.maxsize,
                maxcount=args.maxcount,
            )
            for split in splits
        )