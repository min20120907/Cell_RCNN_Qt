import argparse
import json
import os
import sys
import traceback


def _safe_cast(value, cast_func, default_value):
    try:
        return cast_func(value)
    except (TypeError, ValueError):
        return default_value


def _load_profile(profile_path):
    if not profile_path or not os.path.isfile(profile_path):
        return {}

    try:
        with open(profile_path, "r", encoding="utf-8") as profile_file:
            data = json.load(profile_file)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"Warning: failed to load profile '{profile_path}': {exc}", file=sys.stderr)
        return {}


def _build_parser(argv):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--profile", default="profile.json")
    pre_args, _ = pre_parser.parse_known_args(argv)

    profile = _load_profile(pre_args.profile)
    defaults = {
        "profile": pre_args.profile,
        "dataset_path": profile.get("dataset_path", ""),
        "work_dir": profile.get("WORK_DIR", ""),
        "weight_path": profile.get("weight_path", ""),
        "epochs": _safe_cast(profile.get("epoches"), int, 100),
        "confidence": _safe_cast(profile.get("confidence"), float, 0.9),
        "steps": _safe_cast(profile.get("steps"), int, 1000),
        "train_mode": str(profile.get("train_mode", "train")),
    }

    parser = argparse.ArgumentParser(
        description="Headless trainer for Cell RCNN (no Qt GUI required)."
    )
    parser.add_argument(
        "--profile",
        default=defaults["profile"],
        help="Optional profile json path (default: profile.json).",
    )
    parser.add_argument(
        "--dataset-path",
        default=defaults["dataset_path"],
        help="Dataset directory (same as GUI dataset_path).",
    )
    parser.add_argument(
        "--work-dir",
        default=defaults["work_dir"],
        help="Working directory for logs/checkpoints (same as GUI WORK_DIR).",
    )
    parser.add_argument(
        "--weight-path",
        default=defaults["weight_path"],
        help="Initial weights path. Optional if your training code does not use it.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults["epochs"],
        help="Training epochs (default from profile or 100).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=defaults["confidence"],
        help="Detection confidence threshold (default from profile or 0.9).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=defaults["steps"],
        help="Steps per epoch (default from profile or 1000).",
    )
    parser.add_argument(
        "--train-mode",
        default=defaults["train_mode"],
        help="Train mode string passed to trainingThread (default: train).",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=1,
        help="Value passed to trainingThread(test=...).",
    )

    return parser


def _print_metric(payload):
    if isinstance(payload, dict):
        keys = ["loss", "rpn_bbox_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        pieces = []
        for key in keys:
            if key in payload:
                pieces.append(f"{key}={payload[key]:.6f}")
        if pieces:
            print("[loss] " + " | ".join(pieces), flush=True)
    elif payload is not None:
        print(f"[loss] {payload}", flush=True)


def _validate_args(parser, args):
    if not args.dataset_path:
        parser.error("Missing --dataset-path (or set dataset_path in profile.json).")
    if not os.path.isdir(args.dataset_path):
        parser.error(f"Dataset path not found: {args.dataset_path}")

    if not args.work_dir:
        parser.error("Missing --work-dir (or set WORK_DIR in profile.json).")

    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.steps <= 0:
        parser.error("--steps must be > 0")


def run_headless_training(args):
    from trainingThread import trainingThread

    os.makedirs(args.work_dir, exist_ok=True)

    worker = trainingThread(
        test=args.test,
        steps=args.steps,
        train_mode=args.train_mode,
        dataset_path=args.dataset_path,
        confidence=args.confidence,
        epoches=args.epochs,
        WORK_DIR=args.work_dir,
        weight_path=args.weight_path,
    )

    worker.update_training_status.connect(lambda msg: print(f"[status] {msg}", flush=True))
    worker.update_status_bar.connect(lambda msg: print(f"[batch] {msg}", flush=True))
    worker.update_plot_data.connect(_print_metric)
    worker.update_gallery_signal.connect(
        lambda image_path, index: print(f"[viz] sample={index} file={image_path}", flush=True)
    )

    print("Starting headless training with parameters:", flush=True)
    print(f"  dataset_path: {args.dataset_path}", flush=True)
    print(f"  work_dir: {args.work_dir}", flush=True)
    print(f"  weight_path: {args.weight_path or '(empty)'}", flush=True)
    print(f"  epochs: {args.epochs}", flush=True)
    print(f"  steps: {args.steps}", flush=True)
    print(f"  confidence: {args.confidence}", flush=True)
    print(f"  train_mode: {args.train_mode}", flush=True)
    print("", flush=True)

    worker.run()


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = _build_parser(argv)
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    try:
        run_headless_training(args)
        print("Training finished.", flush=True)
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr, flush=True)
        return 130
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
