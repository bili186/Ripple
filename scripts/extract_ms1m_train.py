#!/usr/bin/env python3
"""
Extract face images and labels from MXNet IndexedRecord format.

Based on the official InsightFace rec2image.py approach.
Reference:
- https://github.com/deepinsight/insightface/blob/master/recognition/common/rec2image.py
- https://huggingface.co/datasets/gaunernst/ms1mv3-recordio#usage

Usage:
    python3 scripts/extract_ms1m_train.py \
        --rec data/train/faces_emore/train.rec \
        --idx data/train/faces_emore/train.idx \
        --output_dir data/train/ms1m_arcface/ms1m_arcface_112x112 \
        --lst_path data/train/ms1m_arcface/ms1m_arcface.lst
"""

import argparse
import os
import os.path as osp
import io
from PIL import Image
import numpy as np

# Fix for older mxnet versions with numpy.bool issue
np.bool = bool

import mxnet as mx
from mxnet.recordio import MXIndexedRecordIO, unpack


def extract_dataset(rec_path, idx_path, output_dir, lst_path):
    """
    Extract all images from MXNet IndexedRecord to individual files.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f'Opening record: {rec_path}')
    print(f'Using index:    {idx_path}')

    # Open the indexed record IO[reference:4]
    record = MXIndexedRecordIO(idx_path, rec_path, 'r')

    # --- 1. Read header from key 0 to get image count ---
    # InsightFace stores the number of images in the header of record 0[reference:7]
    raw_data0 = record.read_idx(0)
    if raw_data0 is None:
        raise ValueError('Key 0 not found in record. Invalid format?')

    header0, _ = unpack(raw_data0)
    # header0.label contains [num_images, ...] or a single value[reference:9]
    if hasattr(header0.label, '__len__'):
        num_images = int(header0.label[0])
    else:
        num_images = int(header0.label)

    print(f'Total images in record: {num_images}')

    # --- 2. Read property file for number of classes (optional but useful) ---
    # The property file is usually located alongside the rec/idx files
    prop_dir = osp.dirname(rec_path)
    prop_path = osp.join(prop_dir, 'property')
    n_classes = None
    if osp.exists(prop_path):
        with open(prop_path, 'r') as f:
            parts = f.read().strip().split(',')
            n_classes = int(parts[0])
            print(f'Number of classes from property file: {n_classes}')
    else:
        print('Warning: property file not found, will infer classes from labels')

    # --- 3. Extract all images (keys 1 to num_images) ---
    # Images are stored at keys 1, 2, ..., num_images[reference:11]
    print(f'Extracting {num_images} images...')
    lst_lines = []
    identity_labels = set()

    for idx in range(1, num_images + 1):
        raw_data = record.read_idx(idx)
        if raw_data is None:
            print(f'Warning: key {idx} not found, skipping')
            continue

        try:
            header, img_bytes = unpack(raw_data)
        except Exception as e:
            print(f'Warning: failed to unpack key {idx}: {e}')
            continue

        # Extract label[reference:12]
        if hasattr(header.label, '__len__'):
            label = int(header.label[0])
        else:
            label = int(header.label)

        identity_labels.add(label)

        # Decode image[reference:13]
        try:
            img = Image.open(io.BytesIO(img_bytes))
            # Convert to RGB if necessary (PIL opens as RGB by default for JPEG)
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception as e:
            print(f'Warning: failed to decode image for key {idx}: {e}')
            continue

        # Save as: output_dir/{identity_label}/{key}.jpg
        subdir = osp.join(output_dir, str(label))
        os.makedirs(subdir, exist_ok=True)
        img_path = osp.join(subdir, f'{idx}.jpg')
        img.save(img_path, quality=95)  # [reference:15]

        # Build lst line: [label] [relative_path][reference:16]
        rel_path = osp.join(str(label), f'{idx}.jpg')
        lst_lines.append(f'{label} {rel_path}\n')

        if idx % 100000 == 0:
            print(f'  extracted {idx} / {num_images} images')

    # --- 4. Write lst file ---
    print(f'Writing lst file to: {lst_path}')
    with open(lst_path, 'w') as f_lst:
        f_lst.writelines(lst_lines)

    # --- 5. Write property file (if not exists) ---
    # Some training scripts require a property file[reference:17]
    if not osp.exists(prop_path):
        # Try to infer image size from the first image
        img_size = 112  # default for MS1M
        # Write property: [num_classes],[height],[width]
        with open(prop_path, 'w') as f_prop:
            f_prop.write(f'{len(identity_labels)},{img_size},{img_size}\n')
        print(f'Created property file: {prop_path}')
    else:
        print(f'Property file already exists: {prop_path}')

    record.close()
    print('Done!')
    print(f'  Images extracted to: {output_dir}')
    print(f'  Lst file: {lst_path}')
    print(f'  Total images: {len(lst_lines)}')
    print(f'  Total identities: {len(identity_labels)}')


def main():
    parser = argparse.ArgumentParser(
        description='Extract MS1M training data from MXNet IndexedRecord'
    )
    parser.add_argument('--rec_path', type=str, required=True,
                        help='Path to train.rec file')
    parser.add_argument('--idx_path', type=str, required=True,
                        help='Path to train.idx file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for extracted images')
    parser.add_argument('--lst_path', type=str, required=True,
                        help='Output path for .lst file')
    args = parser.parse_args()

    extract_dataset(args.rec_path, args.idx_path, args.output_dir, args.lst_path)


if __name__ == '__main__':
    main()
    
# Usage:
    # python3 scripts/extract_ms1m_train.py \
    #     --rec data/train/ms1m-retinaface-t1/train.rec \
    #     --idx data/train/ms1m-retinaface-t1/train.idx \
    #     --output_dir data/train/ms1m_retinaface/ms1m_retinaface_112x112 \
    #     --lst_path data/train/ms1m_retinaface/ms1m_retinaface.lst

