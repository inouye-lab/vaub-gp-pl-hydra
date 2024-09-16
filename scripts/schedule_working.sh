#python src/train.py trainer.devices=1 trainer.max_epochs=100 logger.wandb.project=vaub-gp-working logger.wandb.name=debug

# baseline
#python src/train.py trainer.devices=1 trainer.max_epochs=500 logger.wandb.project=vaub-gp-working logger.wandb.name=test1
#python src/train.py trainer.devices=1 trainer.max_epochs=500 logger.wandb.project=vaub-gp-working logger.wandb.name=test2

# loop 10
python src/train.py trainer.devices=1 trainer.max_epochs=500 model.loops=10 logger.wandb.project=vaub-gp-working logger.wandb.name=loop_10_test1
python src/train.py trainer.devices=1 trainer.max_epochs=500 model.loops=10 logger.wandb.project=vaub-gp-working logger.wandb.name=loop_10_test2

# small gp
python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.03 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_3e-2
python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0.003 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_3e-3
python src/train.py trainer.devices=1 trainer.max_epochs=500 model.gp_lambda=0 logger.wandb.project=vaub-gp-working logger.wandb.name=gp_0
