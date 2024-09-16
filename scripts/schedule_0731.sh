#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
#
#python src/train.py trainer.max_epochs=1000 model.latent_dim=2 model.gp_lambda=1e-3 logger.wandb.name=gp_lambda_1e-3
#python src/train.py trainer.max_epochs=1000 model.latent_dim=2 model.beta=1e-3 logger.wandb.name=beta_1e-3
#
#python src/train.py trainer.max_epochs=1000 model.latent_dim=8 model.gp_lambda=1e-3 logger.wandb.name=lat_8_gp_lambda_1e-3
#python src/train.py trainer.max_epochs=1000 model.latent_dim=8 model.beta=1e-3 logger.wandb.name=lat_8_beta_1e-3

#python src/train.py trainer.max_epochs=1000 model.latent_dim=2 model.gp_lambda=1e-3 logger.wandb.name=gp_lambda_1e-3_clip_new_plus
#python src/train.py trainer.max_epochs=1000 model.latent_dim=2 model.gp_lambda=2e-3 logger.wandb.name=gp_lambda_2e-3_clip_new_plus
#python src/train.py trainer.max_epochs=1000 model.latent_dim=2 model.gp_lambda=5e-3 logger.wandb.name=gp_lambda_5e-3_clip_new_plus

#python src/train.py trainer.max_epochs=1000 model.latent_dim=64 model.beta=2e-3 logger.wandb.name=lat_64_beta_2e-3_clip_new_plus
#python src/train.py trainer.max_epochs=1000 model.latent_dim=64 model.beta=1e-3 logger.wandb.name=lat_64_beta_1e-3_clip_new_plus
#python src/train.py trainer.max_epochs=1000 model.latent_dim=64 model.beta=5e-4 logger.wandb.name=lat_64_beta_5e-4_clip_new_plus

#python src/train.py trainer.max_epochs=1000 model.latent_dim=64 model.gp_lambda=1e-3 logger.wandb.name=lat_64_gp_lambda_1e-3_clip_new_plus
#python src/train.py trainer.max_epochs=1000 model.latent_dim=64 model.gp_lambda=2e-3 logger.wandb.name=lat_64_gp_lambda_2e-3_clip_new_plus
#python src/train.py trainer.max_epochs=1000 model.latent_dim=64 model.gp_lambda=5e-3 logger.wandb.name=lat_64_gp_lambda_5e-3_clip_new_plus

## compare different gp_lambda
#python src/train.py trainer.max_epochs=300 logger.wandb.project=vaub-gp-gp_compare_1 logger.wandb.name=lat_2_gp_lambda_0 model.latent_dim=2 model.cls_lambda=0 model.gp_lambda=0
#python src/train.py trainer.max_epochs=300 logger.wandb.project=vaub-gp-gp_compare_1 logger.wandb.name=lat_2_gp_lambda_1e-3 model.latent_dim=2 model.cls_lambda=0 model.gp_lambda=1e-3
#python src/train.py trainer.max_epochs=300 logger.wandb.project=vaub-gp-gp_compare_1 logger.wandb.name=lat_2_gp_lambda_2e-3 model.latent_dim=2 model.cls_lambda=0 model.gp_lambda=2e-3
#python src/train.py trainer.max_epochs=300 logger.wandb.project=vaub-gp-gp_compare_1 logger.wandb.name=lat_2_gp_lambda_5e-3 model.latent_dim=2 model.cls_lambda=0 model.gp_lambda=5e-3
#
## compare different gp_lambda
#python src/train.py trainer.max_epochs=300 logger.wandb.project=vaub-gp-gp_compare_1 logger.wandb.name=lat_8_gp_lambda_0 model.latent_dim=8 model.cls_lambda=0 model.gp_lambda=0
#python src/train.py trainer.max_epochs=300 logger.wandb.project=vaub-gp-gp_compare_1 logger.wandb.name=lat_8_gp_lambda_1e-3 model.latent_dim=8 model.cls_lambda=0 model.gp_lambda=1e-3
#python src/train.py trainer.max_epochs=300 logger.wandb.project=vaub-gp-gp_compare_1 logger.wandb.name=lat_8_gp_lambda_2e-3 model.latent_dim=8 model.cls_lambda=0 model.gp_lambda=2e-3
#python src/train.py trainer.max_epochs=300 logger.wandb.project=vaub-gp-gp_compare_1 logger.wandb.name=lat_8_gp_lambda_5e-3 model.latent_dim=8 model.cls_lambda=0 model.gp_lambda=5e-3

#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline logger.wandb.name=gp_lambda_0 model.cls_lambda=0 model.gp_lambda=0 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline logger.wandb.name=cls_lambda_1e-3 model.cls_lambda=1e-3 model.gp_lambda=0 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline logger.wandb.name=gp_lambda_1e-3 model.cls_lambda=0 model.gp_lambda=1e-3 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline logger.wandb.name=cls_lambda_1e-3_gp_lambda_1e-3 model.cls_lambda=1e-3 model.gp_lambda=1e-3 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline logger.wandb.name=gp_lambda_1e-4 model.cls_lambda=0 model.gp_lambda=1e-4 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline logger.wandb.name=gp_lambda_1e-5 model.cls_lambda=0 model.gp_lambda=1e-5 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline logger.wandb.name=beta_1e-3 model.beta=1e-3 model.cls_lambda=0 model.gp_lambda=0 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline logger.wandb.name=beta_1e-3 model.beta=5e-4 model.cls_lambda=0 model.gp_lambda=0 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae

#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline-pnp logger.wandb.name=cls_lambda_1e-3_gp_lambda_1e-3 model.cls_lambda=1e-3 model.gp_lambda=1e-3 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline-pnp logger.wandb.name=gp_lambda_1e-4 model.cls_lambda=0 model.gp_lambda=1e-4 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline-pnp logger.wandb.name=gp_lambda_1e-5 model.cls_lambda=0 model.gp_lambda=1e-5 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline-pnp logger.wandb.name=beta_1e-3 model.beta=1e-3 model.cls_lambda=0 model.gp_lambda=0 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline-pnp logger.wandb.name=beta_1e-3 model.beta=5e-4 model.cls_lambda=0 model.gp_lambda=0 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline-pnp logger.wandb.name=beta_5e-3 model.beta=5e-4 model.cls_lambda=0 model.gp_lambda=0 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
## with cls_lambda
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline-pnp logger.wandb.name=beta_5e-3_cls_lambda_1e-3 model.cls_lambda=1e-3 model.gp_lambda=0 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae
#
## with gp_lambda
#python src/train.py trainer.max_epochs=1000 logger.wandb.project=vaub-gp-baseline-pnp logger.wandb.name=beta_5e-3_gp_lambda_1e-3 model.cls_lambda=0 model.gp_lambda=1e-3 data.postfix=ViT-L-14_commonpool_xl_s13b_b90k_28 model=vaub_gp_baseline_vae

# compare batch_norm help or not
#python src/train.py trainer.max_epochs=300 logger.wandb.project=vaub-gp-batch_norm logger.wandb.name=lat_64_wo_batch_norm model.is_batchnorm=false model.latent_dim=64
#
#python src/train.py trainer.max_epochs=300 logger.wandb.project=vaub-gp-batch_norm logger.wandb.name=lat_64_w_batch_norm model.is_batchnorm=true model.latent_dim=64

#python src/train.py trainer.max_epochs=300 logger.wandb.project=vaub-gp-batch_norm logger.wandb.name=lat_64_w_batch_norm_warm model.is_batchnorm=true model.latent_dim=64 model.warm_epochs=50

#python src/train.py trainer.max_epochs=100 logger.wandb.project=vaub-gp-batch_norm logger.wandb.name=lat_64_w_batch_norm_warm_5e-2_better_plus model.is_batchnorm=true model.latent_dim=64 model.warm_epochs=50 model.beta=5e-2
#
#python src/train.py trainer.max_epochs=100 logger.wandb.project=vaub-gp-batch_norm logger.wandb.name=lat_64_w_batch_norm_warm_5e-1_better_plus model.is_batchnorm=true model.latent_dim=64 model.warm_epochs=50 model.beta=5e-1
#
#python src/train.py trainer.max_epochs=100 logger.wandb.project=vaub-gp-batch_norm logger.wandb.name=lat_64_w_batch_norm_warm_5e-0_better_plus model.is_batchnorm=true model.latent_dim=64 model.warm_epochs=50 model.beta=5e-0

#python src/train.py trainer.max_epochs=5000 logger.wandb.project=vaub-gp-batch_norm logger.wandb.name=lat_64_w_batch_norm_warm_5e-0_better_plus_long model.is_batchnorm=true model.latent_dim=64 model.warm_epochs=50 model.beta=5e-0 model.gp_lambda=5e-2 trainer.devices=1

#python src/train.py trainer.max_epochs=5000 logger.wandb.project=vaub-gp-batch_norm logger.wandb.name=lat_64_w_batch_norm_5e-0_baseline_better_plus_long model.is_batchnorm=true model.latent_dim=64 model.warm_epochs=0 model.beta=5e-0 model.gp_lambda=5e-2 trainer.devices=1

#python src/train.py trainer.max_epochs=2000 logger.wandb.project=vaub-gp-worked logger.wandb.name=gp_lambda_0 model.is_batchnorm=true model.latent_dim=64 model.warm_epochs=0 model.beta=5e-0 model.gp_lambda=0 trainer.devices=1

#python src/train.py trainer.max_epochs=2000 logger.wandb.project=vaub-gp-worked logger.wandb.name=gp_lambda_5e-3 model.is_batchnorm=true model.latent_dim=64 model.warm_epochs=0 model.beta=5e-0 model.gp_lambda=5e-3 trainer.devices=1

#python src/train.py trainer.max_epochs=2000 logger.wandb.project=vaub-gp-worked logger.wandb.name=gp_lambda_0_w_warm model.is_batchnorm=true model.latent_dim=64 model.warm_epochs=50 model.beta=5e-0 model.gp_lambda=0 trainer.devices=1

#python src/train.py trainer.max_epochs=2000 logger.wandb.project=vaub-gp-worked logger.wandb.name=gp_lambda_0_w_warm_beta_5e-1 model.is_batchnorm=true model.latent_dim=64 model.warm_epochs=0 model.beta=5e-1 model.gp_lambda=0 trainer.devices=1

#python src/train.py trainer.max_epochs=2000 logger.wandb.project=vaub-gp-worked logger.wandb.name=gp_lambda_0_w_warm_beta_5e-2 model.is_batchnorm=true model.latent_dim=64 model.warm_epochs=0 model.beta=5e-2 model.gp_lambda=0 trainer.devices=1

python src/train.py trainer.max_epochs=2000 logger.wandb.project=vaub-gp-worked logger.wandb.name=gp_lambda_0_w_warm_beta_5e-2_larger_init model.is_batchnorm=true model.latent_dim=64 model.warm_epochs=0 model.beta=5e-2 model.gp_lambda=0 trainer.devices=1
