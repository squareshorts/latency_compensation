import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run latency-compensation confirmation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if not args.config.is_file():
        raise SystemExit(f"Missing config: {args.config}")
    if args.smoke_test:
        print(f"Smoke test passed for {args.config}")
        return
    raise SystemExit("AV2 confirmation is not runnable until the fixed official cohort is present; see results/av2_confirmation/data_presence_audit.md")


if __name__ == "__main__":
    main()
