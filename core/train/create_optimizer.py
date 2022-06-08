import imp

from configs import cfg

def create_optimizer(network):
    module = cfg.optimizer_module
    optimizer_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, optimizer_path).get_optimizer(network)
