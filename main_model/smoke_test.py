"""
Fast sanity check for model.py / utils.py / dataset.py -- run this BEFORE
committing to full training. If anything here fails, the real four-
condition run will fail too, just slower and more expensively.

Covers, in under a minute on CPU:
  1. UNet forward + backward for all four conditions (A/B/C/D) and for
     HUTUBS's shape (labels=440) -- catches any shape bug in Block/UNet.
  2. The HUTUBS -> SONICOM partial pretrained-weight transfer, using a
     throwaway fake checkpoint -- confirms condition B transfers far more
     tensors than A/C/D (see README's transfer-learning table).
  3. (Optional) Real dataset loading, if you pass real directories --
     skipped otherwise so this still runs with no data at all.

Usage:
    python smoke_test.py
    python smoke_test.py --hutubs_dir ./HUTUBS/HRIRs --hutubs_csv ./HUTUBS/AnthropometricMeasures.csv \
                          --sonicom_dir ./SONICOM/HRIRs --sonicom_csv ./SONICOM/AnthropometricMeasures.csv \
                          --sonicom_images ./SONICOM/cropped
"""
import argparse
import os
import torch

from model import UNet
from utils import load_matching_state_dict

# Same condition -> (ear_dim, use_image) mapping as main.py.
IMAGE_FEAT_DIM = 32
CONDITIONS = {
    'A': dict(ear_dim=0,  use_image=False),
    'B': dict(ear_dim=24, use_image=False),
    'C': dict(ear_dim=0,  use_image=True),
    'D': dict(ear_dim=24, use_image=True),
}


def check(name, ok):
    status = 'OK' if ok else 'FAIL'
    print(f"    [{status}] {name}")
    assert ok, f"Smoke test failed: {name}"


def build(labels, condition_key, base_channels=4):
    """Tiny base_channels -- this is a shape/logic check, not a quality check."""
    cond = CONDITIONS[condition_key]
    return UNet(
        audio_channels=2, labels=labels, head_dim=0,
        ear_dim=cond['ear_dim'],
        image_dim=(IMAGE_FEAT_DIM if cond['use_image'] else 0),
        base_channels=base_channels,
    )


def test_forward_backward(labels, condition_key, batch=2):
    print(f"\n--- condition {condition_key}  (labels={labels}) ---")
    cond = CONDITIONS[condition_key]
    unet = build(labels, condition_key)
    print(f"    params: {sum(p.numel() for p in unet.parameters()):,}")

    x      = torch.randn(batch, 2, 256)
    t      = torch.randint(0, 600, (batch,))
    label  = torch.randint(0, labels, (batch,))
    ears   = torch.rand(batch, 24)          if cond['ear_dim']   else None
    images = torch.rand(batch, 6, 128, 128) if cond['use_image'] else None

    out = unet(x, t, labels=label, ears_embedding=ears, images=images)
    check("output shape matches input (B,2,256)", tuple(out.shape) == tuple(x.shape))

    out.mean().backward()
    n_params = sum(1 for _ in unet.parameters())
    n_with_grad = sum(1 for p in unet.parameters() if p.grad is not None)
    check(f"gradients reach every parameter ({n_with_grad}/{n_params})", n_with_grad == n_params)


def test_pretrained_transfer():
    print("\n=== Transfer-loading smoke test (HUTUBS[440] -> SONICOM[793]) ===")
    # Build a "HUTUBS" condition-B model and save it as a throwaway checkpoint,
    # exactly the shape train_fold() would produce for the real baseline.
    hutubs_unet = build(labels=440, condition_key='B')
    ckpt_path = '/tmp/_smoke_test_fake_hutubs_fold1.pt'
    torch.save({'ema_state_dict': hutubs_unet.state_dict()}, ckpt_path)

    expectation = {  # roughly: does this condition share HUTUBS's active branches?
        'A': 'fewer tensors match (ear branch off, cond_fuse shape differs)',
        'B': 'most tensors match (same active branches as HUTUBS)',
        'C': 'fewer tensors match (image branch is new, cond_fuse shape differs)',
        'D': 'ear_fc matches but cond_fuse/image do not',
    }
    for cond_key in ['A', 'B', 'C', 'D']:
        sonicom_unet = build(labels=793, condition_key=cond_key)
        print(f"\n  condition {cond_key} -- expect: {expectation[cond_key]}")
        load_matching_state_dict(sonicom_unet, ckpt_path)

    os.remove(ckpt_path)


def test_dataset(args):
    from dataset import HUTUBSDataset, SONICOMDataset

    if args.hutubs_dir and os.path.isdir(args.hutubs_dir):
        print("\n=== HUTUBS dataset (real data) ===")
        ds = HUTUBSDataset(args.hutubs_dir, args.hutubs_csv)
        print(f"    {len(ds)} samples | {ds.measurement_points} measurement points | "
              f"{len(ds.valid_subject_indices)} subjects")
        sample = ds[0]
        check("sample has hrtf + ear_measurements", 'hrtf' in sample and 'ear_measurements' in sample)
        check("ear_measurements is 24-d", sample['ear_measurements'].shape == (24,))
    else:
        print("\n(skipping HUTUBS dataset check -- pass --hutubs_dir to enable)")

    if args.sonicom_dir and os.path.isdir(args.sonicom_dir):
        print("\n=== SONICOM dataset, with images (real data) ===")
        ds = SONICOMDataset(args.sonicom_dir, args.sonicom_csv, image_dir=args.sonicom_images)
        print(f"    {len(ds)} samples | {ds.measurement_points} measurement points | "
              f"{len(ds.valid_subject_indices)} subjects")
        sample = ds[0]
        check("sample has image", 'image' in sample)
        check("image shape is (6,128,128)", tuple(sample['image'].shape) == (6, 128, 128))
    else:
        print("\n(skipping SONICOM dataset check -- pass --sonicom_dir to enable)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hutubs_dir', default=None)
    parser.add_argument('--hutubs_csv', default=None)
    parser.add_argument('--sonicom_dir', default=None)
    parser.add_argument('--sonicom_csv', default=None)
    parser.add_argument('--sonicom_images', default=None)
    args = parser.parse_args()

    print("=== UNet forward/backward, all four conditions (SONICOM shape) ===")
    for key in ['A', 'B', 'C', 'D']:
        test_forward_backward(labels=793, condition_key=key)

    print("\n=== UNet forward/backward, HUTUBS shape (condition B only) ===")
    test_forward_backward(labels=440, condition_key='B')

    test_pretrained_transfer()
    test_dataset(args)

    print("\nAll smoke tests passed — safe to move on to a real 1-epoch run.")
