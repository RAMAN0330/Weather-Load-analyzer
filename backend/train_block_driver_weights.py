import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from .short_term_pipeline import compute_block_driver_weights_elastic_net
except Exception:
    from short_term_pipeline import compute_block_driver_weights_elastic_net


def _resolve_input_path(path_arg: Optional[str]) -> Path:
    if path_arg:
        p = Path(path_arg).expanduser().resolve()
        if p.exists():
            return p
    root_default = Path("final_data.csv").resolve()
    if root_default.exists():
        return root_default
    local_default = (Path(__file__).resolve().parent.parent / "final_data.csv").resolve()
    if local_default.exists():
        return local_default
    raise FileNotFoundError("Could not find input CSV. Pass --input explicitly.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train block-wise Elastic Net load elasticity weights (1..96 blocks)."
    )
    parser.add_argument("--input", type=str, default=None, help="Input historical CSV path (default: final_data.csv).")
    parser.add_argument("--output", type=str, default="block_driver_weights.csv", help="Output CSV path.")
    parser.add_argument("--target-date", type=str, default=None, help="Train using rows before this date (YYYY-MM-DD).")
    parser.add_argument("--min-samples", type=int, default=14, help="Minimum samples required per block to train.")
    parser.add_argument("--max-iter", type=int, default=10000, help="Max iterations for ElasticNetCV.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for deterministic CV splits.")
    parser.add_argument("--smooth-window", type=int, default=0, help="Optional rolling smoothing window across blocks.")
    parser.add_argument("--temp-squared", action="store_true", help="Add temp^2 feature.")
    parser.add_argument("--temp-humidity-interaction", action="store_true", help="Add temp x humidity feature.")
    parser.add_argument("--season-segment", action="store_true", help="Train on target season history only.")
    parser.add_argument("--round", type=int, default=6, help="Decimal places for output.")
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input)
    df = pd.read_csv(input_path)

    weights = compute_block_driver_weights_elastic_net(
        history_df=df,
        target_date=args.target_date,
        min_samples=int(args.min_samples),
        max_iter=int(args.max_iter),
        random_state=int(args.random_state),
        add_temp_squared=bool(args.temp_squared),
        add_temp_humidity_interaction=bool(args.temp_humidity_interaction),
        season_segment=bool(args.season_segment),
        smooth_window=int(args.smooth_window),
    )

    out = weights.rename(
        columns={
            "temp_weight": "temp",
            "humidity_weight": "humidity",
            "rain_weight": "rain",
            "wind_weight": "wind",
            "daytype_weight": "daytype",
            "holiday_weight": "holiday",
        }
    )[["block", "temp", "humidity", "rain", "wind", "daytype", "holiday"]].copy()

    decimals = max(0, int(args.round))
    for col in ("temp", "humidity", "rain", "wind", "daytype", "holiday"):
        out[col] = out[col].astype(float).round(decimals)

    output_path = Path(args.output).expanduser().resolve()
    os.makedirs(output_path.parent, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(out)}")
    print("Columns: block,temp,humidity,rain,wind,daytype,holiday")


if __name__ == "__main__":
    main()
