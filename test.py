#from src.utils.builder import build_trainer, build_config
from src.utils.utils import *
from src.utils.builder import *
from src.utils.aws import *
from arguments import args
from time import time
# from src.testers.mil_tester import MILTester
# from src.testers.mil_tester_internal import MILTesterInternal

def main():
    setup_imports()
    parser= args.get_parser()
    opts = parser.parse_args()

    config = build_config(opts)

    # start testing
    trainer = build_trainer(config)
    trainer.test()
    

if __name__=='__main__':
    main()