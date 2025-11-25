python generate_fake_dataset.py \
    --num_rows 500 \
    --domains game,video,music,theme,read \
    --domain_ratio game:0.05,video:0.7,music:0.1,theme:0.1,read:0.05 \
    --domain_field_config "$(cat configs/domain_field_config.json)" \
    --sequence_length_config "$(cat configs/sequence_length_config.json)" \
    --scenario_config "$(cat configs/scenario_config.json)" \
    --fixed_user_item_k 3 \
    --ratio_noise 0.05 \
    --value_range 0,1 \
    --save_path dataset/my_fake_dataset.csv

