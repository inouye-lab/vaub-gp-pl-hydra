#python src/train.py trainer.devices=1 trainer.max_epochs=100 logger.wandb.project=vaub-gp-working logger.wandb.name=debug

# baseline
#python src/train.py trainer.devices=1 trainer.max_epochs=500 logger.wandb.project=vaub-gp-working logger.wandb.name=test1
#python src/train.py trainer.devices=1 trainer.max_epochs=500 logger.wandb.project=vaub-gp-working logger.wandb.name=test2

## loop 10
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.loops=10 logger.wandb.project=vaub-gp-working logger.wandb.name=loop_10_test1
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.loops=10 logger.wandb.project=vaub-gp-working logger.wandb.name=loop_10_test2
#
## small gp
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.03 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_3e-2
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.003 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_3e-3
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_0
#
## small gp loop 10
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.03 model.loops=10 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_3e-2_loop_10
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.003 model.loops=10 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_3e-3_loop_10

# binary gp
#python src/train.py trainer.devices=1 trainer.max_epochs=200 model.gp_lambda=0.3 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_3e-1_binary
#python src/train.py trainer.devices=1 trainer.max_epochs=200 model.gp_lambda=0.03 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_3e-2_binary
#python src/train.py trainer.devices=1 trainer.max_epochs=200 model.gp_lambda=0.003 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_3e-3_binary

#python src/train.py trainer.devices=1 trainer.max_epochs=200 model.gp_lambda=0.01 model.loops=10 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_1e-2_binary_loop_10

#python src/train.py trainer.devices=1 trainer.max_epochs=200 model.gp_lambda=0.005 model.loops=5 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_5e-3_binary_loop_5

#python src/train.py trainer.devices=1 trainer.max_epochs=2000 model.gp_lambda=0.005 model.loops=10 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_5e-3_binary_loop_10_long
#python src/train.py trainer.devices=1 trainer.max_epochs=2000 model.gp_lambda=0.01 model.loops=10 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_1e-2_binary_loop_10_long

#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.005 model.loops=10 logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=gp_5e-3_binary_loop_10
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.05 model.loops=10 logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=gp_5e-2_binary_loop_10
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.5 model.loops=10 logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=gp_5e-1_binary_loop_10


# test MAE
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.005 model.loops=10 model.gp.diff="MAE" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-3_binary_loop_10
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.05 model.loops=10 model.gp.diff="MAE" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-2_binary_loop_10
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.5 model.loops=10 model.gp.diff="MAE" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-1_binary_loop_10


# max norm
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.005 model.loops=10 model.gp.mode="max" model.gp.diff="MAE" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-3_max_loop_10
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.05 model.loops=10 model.gp.mode="max" model.gp.diff="MAE" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-2_max_loop_10
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.5 model.loops=10 model.gp.mode="max" model.gp.diff="MAE" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-1_max_loop_10


# L2 in x space
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.005 model.loops=10 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-3_max_L2_loop_10
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.05 model.loops=10 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-2_max_L2_loop_10
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.5 model.loops=10 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-1_max_L2_loop_10


# debug
#python src/train.py trainer.devices=1 trainer.max_epochs=2 model.gp_lambda=0.005 model.loops=10 logger.wandb.project=vaub-gp-working logger.wandb.name=debug
#python src/train.py trainer.devices=1 trainer.max_epochs=2 model.gp_lambda=0.005 model.loops=10 model.gp.diff="MAE" logger.wandb.project=vaub-gp-working logger.wandb.name=debug
#python src/train.py trainer.devices=1 trainer.max_epochs=2 model.gp_lambda=0.005 model.loops=10 model.gp.mode="max" model.gp.diff="MAE" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=debug
#python src/train.py trainer.devices=1 trainer.max_epochs=3 model.warm_score_epochs=1 model.gp_lambda=0.005 model.loops=10 model.gp.dist_x_mode="L2" model.gp.mode="max" model.gp.diff="MAE" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=debug

#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.warm_score_epochs=1 model.gp_lambda=0 model.loops=5 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_0_max_L2_loop_5_wo_warm
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.05 model.loops=10 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-2_max_L2_loop_10
#python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.5 model.loops=10 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-1_max_L2_loop_10

#python src/train.py trainer.devices=1 trainer.max_epochs=1000 model.warm_score_epochs=1 model.gp_lambda=5e-3 model.loops=5 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-3_max_L2_loop_5_wo_warm
#python src/train.py trainer.devices=1 trainer.max_epochs=1000 model.warm_score_epochs=1 model.gp_lambda=5e-2 model.loops=5 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-2_max_L2_loop_5_wo_warm
#python src/train.py trainer.devices=1 trainer.max_epochs=1000 model.warm_score_epochs=1 model.gp_lambda=5e-1 model.loops=5 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-1_max_L2_loop_5_wo_warm

python src/train.py trainer.devices=1 trainer.max_epochs=1000 model.warm_score_epochs=1 model.gp_lambda=5e-1 model.loops=5 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-1_max_L2_loop_5_wo_warm_init
python src/train.py trainer.devices=1 trainer.max_epochs=1000 model.warm_score_epochs=50 model.gp_lambda=5e-1 model.loops=5 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-1_max_L2_loop_5_w_warm_init
python src/train.py trainer.devices=1 trainer.max_epochs=1000 model.warm_score_epochs=50 model.gp_lambda=5e-1 model.loops=5 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-1_max_L2_loop_5_w_warm_init_test2

python src/train.py trainer.devices=1 trainer.max_epochs=400 model.init_scale=0.2 model.warm_score_epochs=0 model.gp_lambda=5e-1 model.loops=5 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-1_max_L2_loop_5_wo_warm_init_scale_2e-1
python src/train.py trainer.devices=1 trainer.max_epochs=400 model.init_scale=0.2 model.warm_score_epochs=50 model.gp_lambda=5e-1 model.loops=5 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-1_max_L2_loop_5_w_warm_init_scale_2e-1

#python src/train.py trainer.devices=1 trainer.max_epochs=1000 model.warm_score_epochs=50 model.gp_lambda=1e-0 model.loops=5 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_1e-1_max_L2_loop_5_w_warm_init
#python src/train.py trainer.devices=1 trainer.max_epochs=1000 model.init_scale=0.2 model.warm_score_epochs=50 model.gp_lambda=5e-1 model.loops=5 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-1_max_L2_loop_5_w_warm_init_scale_2e-1

#CUDA_VISIBLE_DEVICES=1 python src/train.py trainer.devices=1 trainer.max_epochs=20000 model.warm_score_epochs=1 model.gp_lambda=5e-1 model.loops=5 model.gp.mode="max" model.gp.diff="MAE" model.gp.dist_x_mode="L2" logger.wandb.project=vaub-gp-working-fixed logger.wandb.name=MAE_gp_5e-1_max_L2_loop_5_wo_warm_init_long
