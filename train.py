from configs import cfg

from core.utils.log_util import Logger
from core.data import create_dataloader
from core.nets import create_network
from core.train import create_trainer, create_optimizer


def main():
    log = Logger()
    log.print_config()

    model = create_network()
    optimizer = create_optimizer(model)
    trainer = create_trainer(model, optimizer)

    train_loader = create_dataloader('train')

    # estimate start epoch
    epoch = trainer.iter // len(train_loader) + 1
    while True:
        if trainer.iter > cfg.train.maxiter:
            break
        
        trainer.train(epoch=epoch,
                      train_dataloader=train_loader)
        epoch += 1

    trainer.finalize()

if __name__ == '__main__':
    main()
