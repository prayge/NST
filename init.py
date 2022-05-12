import argparse
import os 

class Config():

    def __init__(self):
        self.initialized = False

    def initialize(self,parser):
        parser.add_argument('--contentimage', type=str, default='png/content.jpg')
        parser.add_argument('--styleimage', type=str, default='png/style.jpg')
        parser.add_argument('--max_size', type=int, default=750)
        parser.add_argument('--steps', type=int, default=2000)
        parser.add_argument('--log_step', type=int, default=2)
        parser.add_argument('--sample_step', type=int, default=1000)
        parser.add_argument('--style_weight', type=float, default=100)
        parser.add_argument('--lr', type=float, default=0.003)
        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        cfg = parser.parse_args()
        # set gpu ids
        return cfg