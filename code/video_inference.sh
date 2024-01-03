python video_inference.py --patch_size 240 \
    --temp_patch 12 \
    --resize_ratio 1.0 \
    --input_path video_22_all.avi \
    --out_path video_22_out.mp4 \
    --model_path ./model_zoo/shuffle_MS_video.pth \
    --save_video \
    --concatenate_input
