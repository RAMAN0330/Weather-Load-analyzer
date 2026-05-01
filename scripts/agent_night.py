"""
agent_night.py — Agent 1: blocks 1-24 (00:00-06:00)
Nocturnal humidity-driven AC load, baseline cooling floor.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.agent_analysis import (
    load_section_df, classify_regimes, get_storm_days,
    run_all_regressions, build_storm_profile, detect_monsoon_onset,
    build_coeff_heatmap, build_regime_calendar, build_storm_chart,
    build_yoy_drift_chart, build_onset_chart, save_section_outputs,
)

PERIOD     = 'night'
BLK_START  = 1
BLK_END    = 24


def main():
    print(f"[{PERIOD}] Loading data blocks {BLK_START}–{BLK_END}...")
    df = load_section_df(BLK_START, BLK_END)
    print(f"[{PERIOD}] {len(df):,} rows loaded")

    storm_days  = get_storm_days(df)
    print(f"[{PERIOD}] Storm days detected: {len(storm_days)}")

    regime_df   = classify_regimes(df)
    coeff_df    = run_all_regressions(df, storm_days)
    storm_profile_df, storm_events_df = build_storm_profile(df, storm_days)
    onset_df    = detect_monsoon_onset(df)

    figures = {
        'coeff_heatmap':   build_coeff_heatmap(coeff_df, PERIOD),
        'regime_calendar': build_regime_calendar(regime_df, PERIOD),
        'storm_profile':   build_storm_chart(storm_profile_df, PERIOD),
        'yoy_drift':       build_yoy_drift_chart(coeff_df, PERIOD),
        'onset':           build_onset_chart(onset_df),
    }

    save_section_outputs(PERIOD, coeff_df, regime_df, storm_profile_df,
                         storm_events_df, onset_df, figures)
    print(f"[{PERIOD}] Done.")


if __name__ == '__main__':
    main()
