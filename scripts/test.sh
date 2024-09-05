for noise in $(seq 0.05 0.05 1.0); do
python -m torch.distributed.launch --nproc_per_node=2 \
        metro/tools/run_metro_bodymesh.py \
        --val_yaml 3dpw/test_has_gender.yaml \
        --arch hrnet-w64 \
        --num_workers 2 \
        --per_gpu_eval_batch_size 30 \
        --num_hidden_layers 4 \
        --num_attention_heads 4 \
        --input_feat_dim 2051,512,128 \
        --hidden_feat_dim 1024,256,128 \
        --run_eval_only \
        --noise_scale_factor $noise \
        --resume_checkpoint ./models/metro_release/metro_3dpw_state_dict.bin \
        --output_dir METRO-L-H64_3dpw_eval_noise_$noise/
done