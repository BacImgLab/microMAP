OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 \
../u2net/train_mix96_ddp.py \
--batch_size=2 \
--num_workers=4 \
--model_name=u2net_full \
--in_compress="clip" \
--data_loader='v5' \
--bg_weight=0.05 \
--epochs=1000 \
--tr_bac_num=9 \
--output_dir="/CX/neuro_tracking/ruifanghai/test_train/"   \
--mix_num=11  \
--dataset='20251002_an' \
--gt_name='gt'  \
--im_name='im_crop'  \
--schedule='cnst'  \
--n_classes=97 \
--data_dir='/CX/neuro_tracking/ruifanghai/'

