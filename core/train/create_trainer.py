import imp

from configs import cfg

def _query_trainer():
    module = cfg.trainer_module
    trainer_path = module.replace(".", "/") + ".py"
    trainer = imp.load_source(module, trainer_path).Trainer
    return trainer


def create_trainer(network, optimizer):
    Trainer = _query_trainer()
    return Trainer(network, optimizer)
