#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
#
#python src/train.py trainer.max_epochs=1000 model.latent_dim=2 model.gp_lambda=1e-3 logger.wandb.name=gp_lambda_1e-3
#python src/train.py trainer.max_epochs=1000 model.latent_dim=2 model.beta=1e-3 logger.wandb.name=beta_1e-3
#
#python src/train.py trainer.max_epochs=1000 model.latent_dim=8 model.gp_lambda=1e-3 logger.wandb.name=lat_8_gp_lambda_1e-3
#python src/train.py trainer.max_epochs=1000 model.latent_dim=8 model.beta=1e-3 logger.wandb.name=lat_8_beta_1e-3

python src/train.py trainer.max_epochs=1000 model.latent_dim=2 model.gp_lambda=1e-3 logger.wandb.name=gp_lambda_1e-3_clip_new
python src/train.py trainer.max_epochs=1000 model.latent_dim=2 model.gp_lambda=2e-3 logger.wandb.name=gp_lambda_2e-3_clip_new
python src/train.py trainer.max_epochs=1000 model.latent_dim=2 model.gp_lambda=5e-3 logger.wandb.name=gp_lambda_5e-3_clip_new


