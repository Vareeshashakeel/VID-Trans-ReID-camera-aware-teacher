
# Camera-aware VID-Trans-ReID Teacher (cleaned)

This repo is the cleaned camera-aware training code to produce the **teacher checkpoint** needed for teacher→student distillation.

## Main fixes
- dataset root is passed from CLI (`--dataset_root`)
- modern Python/PyTorch compatibility fix for `torch._six`
- clean checkpoint saving to `--output_dir`
- `center_w=0.0` is safe
- validation/test path is cleaned

## Train on MARS
```bash
python VID_Trans_ReID.py   --Dataset_name Mars   --dataset_root /absolute/path/to/MARS   --model_path /absolute/path/to/jx_vit_base_p16_224-80ecf9dd.pth   --output_dir ./output_camera_aware_teacher   --epochs 120   --eval_every 10   --batch_size 64   --num_workers 4   --seq_len 4   --num_instances 4   --center_w 0.0005   --attn_w 1.0
```

Best teacher checkpoint will be saved as:
- `Mars_camera_aware_best.pth`
