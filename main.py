import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Computer Vision final project experiment runner.")
    parser.add_argument("--mode", required=True, choices=["setup", "validate", "E1", "E2", "E3", "E4", "demo"])
    parser.add_argument("--image", default="eval_09_h", help="Evaluation image ID for demo mode.")
    args = parser.parse_args()

    if args.mode == "setup":
        from tools.prepare_dataset import setup
        setup()
        return

    if args.mode == "validate":
        from tools.validate_dataset import validate
        validate()
        return

    if args.mode == "E1":
        from experiments.E1_challenge import run_e1
        run_e1()
        return

    if args.mode == "E2":
        from experiments.E2_parameter_search import run_e2
        run_e2()
        return

    if args.mode == "E3":
        from experiments.E3_ablation import run_e3
        run_e3()
        return

    if args.mode == "E4":
        from experiments.E4_failure_point import run_e4
        run_e4()
        return

    if args.mode == "demo":
        from tools.demo import run_demo
        run_demo(args.image)
        return

if __name__ == "__main__":
    main()