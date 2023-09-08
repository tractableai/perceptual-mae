from src.utils.builder import setup_imports, build_config, build_trainer
from arguments import args


def main():
    setup_imports()
    
    parser = args.get_parser()
    opts = parser.parse_args()

    config = build_config(opts)

    # start training
    trainer = build_trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
