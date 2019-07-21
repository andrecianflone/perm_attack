
python main.py \
    --hidden_init \
    --deterministic \
    --LAMBDA=0.05 \
    --carlini_loss \
    --batch_size=32 \
    --prepared_data='dataloader/128_prepared_data.pickle' \
    --nearest_neigh_all \
    --diff_nn \
    --save_model \
    --adv_model_path='saved_models/dag_diff.pt' \
    --temp_decay_schedule=25 \
    --epochs=15 \
    --seed=1 \
    --namestr="Perm attack"
