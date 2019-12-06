
gpu_id='1'
stop_epoch=500
patience=50
# dataset='cross_char'
# vaegan_exp='omn_noLatin_1114_0956'
# vaegan_step=200000
dataset=$1
vaegan_exp=$2
vaegan_step=$3
# vaegan_exp='omn_noLatin_bn_1119_2057'
# vaegan_step=3000
zvar_lambda=$4
echo $zvar_lambda

for fake_prob in $(seq 0.1 0.1 0.9) # 0.1, 0.2, 0.3, 0.4, 0.5
do
    python ./train.py --vaegan_exp $vaegan_exp --vaegan_step $vaegan_step --zvar_lambda $zvar_lambda --fake_prob $fake_prob --vaegan_is_train --dataset $dataset --model Conv4 --method protonet --stop_epoch $stop_epoch --patience $patience --gpu_id $gpu_id
    python ./save_features.py --vaegan_exp $vaegan_exp --vaegan_step $vaegan_step --zvar_lambda $zvar_lambda --fake_prob $fake_prob --vaegan_is_train --dataset $dataset --model Conv4 --method protonet --gpu_id $gpu_id
    python ./test.py --vaegan_exp $vaegan_exp --vaegan_step $vaegan_step --zvar_lambda $zvar_lambda --fake_prob $fake_prob --vaegan_is_train --dataset $dataset --model Conv4 --method protonet --gpu_id $gpu_id
done
