python -m torch.distributed.launch --nproc_per_node=8 \
       metro/tools/run_metro_bodymesh.py \
       --train_yaml Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml \
       --val_yaml 3dpw/test_has_gender.yaml \
       --arch hrnet-w64 \
       --num_workers 4 \
       --per_gpu_train_batch_size 20 \
       --per_gpu_eval_batch_size 20 \
       --num_hidden_layers 4 \
       --num_attention_heads 4 \
       --lr 1e-4 \
       --num_train_epochs 200 \
       --input_feat_dim 2051,512,128 \
       --hidden_feat_dim 1024,256,128 