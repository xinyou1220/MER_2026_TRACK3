"""Zip a submission CSV for upload."""

import argparse
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/submission.csv")
    parser.add_argument("--out", default="outputs/submission.zip")
    args = parser.parse_args()
    with ZipFile(args.out, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(args.input, arcname="submission.csv")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
