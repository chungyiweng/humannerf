import sys
import os

from termcolor import colored
from configs import cfg


class Logger(object):
    r"""Duplicates all stdout to a file."""
    def __init__(self):
        path = os.path.join(cfg.logdir, 'logs.txt')

        log_dir = cfg.logdir
        if not cfg.resume and os.path.exists(log_dir):
            user_input = input(f"log dir \"{log_dir}\" exists. \nRemove? (y/n):")
            if user_input == 'y':
                print(colored('remove contents of directory %s' % log_dir, 'red'))
                os.system('rm -r %s/*' % log_dir)
            else:
                print(colored('exit from the training.', 'red'))
                exit(0)

        if not os.path.exists(log_dir):
            os.makedirs(cfg.logdir)

        self.log = open(path, "a") if os.path.exists(path) else open(path, "w")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        self.stdout.write(message)
        self.stdout.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

    def print_config(self):
        print("\n\n######################### CONFIG #########################\n")
        print(cfg)
        print("\n##########################################################\n\n")
