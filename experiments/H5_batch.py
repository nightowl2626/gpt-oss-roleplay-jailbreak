import argparse
import json
import random
import time
from typing import List

# This script requires two other files in the same folder:
# 1. H5.py: The simplified single-run experiment script.
# 2. batch_utils.py: Your helper file with utility functions.
import H5 as H5

# Import all necessary helpers, including the statistical functions, from your utility file.
from batch_utils import (
    add_common_args,
    seed_everything,
    run_silenced,
    open_csv_writer,
    open_jsonl,
    mean,
    stdev,
    ci95_t,
    welch_t,
)

def paired_t(deltas):
    """
    deltas: list of (condition - baseline) per trial
    Returns (t_stat, dof, p_approx_two_sided)
    """
    n = len(deltas)
    if n < 2:
        return 0.0, n - 1, 1.0
    m = mean(deltas)
    sd = stdev(deltas)
    if sd == 0.0:
        # All deltas identical; if nonzero, it's effectively 'certain', else no effect
        return (1e9 if m != 0 else 0.0), n - 1, (0.0 if m != 0 else 1.0)
    t = m / (sd / (n ** 0.5))
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))  # normal approx
    return t, n - 1, p

# ===================================================================
# BATCH EXPERIMENT
# ===================================================================

def run_batch_experiment(
    num_trials: int,
    private_temp: float,
    observed_temp: float,
    masked_temp: float,
    temp_jitter: float,
    sleep_seconds: float,
    csv_path: str,
    jsonl_path: str,
    seed: int,
):
    """
    Runs the H5 experiment multiple times to collect statistical data.
    """
    seed_everything(seed)
    print(f"Running {num_trials} trials with seed {seed}...")

    # --- Setup: Initialize score lists and file writers ---
    private_scores, observed_scores, masked_scores = [], [], []

    csv_file, csv_writer = open_csv_writer(csv_path, ["trial", "score_private", "score_observed", "score_masked", "temp_private", "temp_observed", "temp_masked"])
    jsonl_file = open_jsonl(jsonl_path, {"suite": "H5", "trials": num_trials, "seed": seed, "jitter": temp_jitter})

    # --- Main Loop: Run the experiment for each trial ---
    for i in range(1, num_trials + 1):
        # Apply a small random "jitter" to each temperature for this trial.
        temp_p = max(0.0, min(1.0, private_temp  + random.uniform(-temp_jitter, temp_jitter)))
        temp_o = max(0.0, min(1.0, observed_temp + random.uniform(-temp_jitter, temp_jitter)))
        temp_m = max(0.0, min(1.0, masked_temp   + random.uniform(-temp_jitter, temp_jitter)))

        # Run all three conditions (private, observed, masked) in a single go.
        result = run_silenced(
            H5.run_experiment,
            temp_private=temp_p,
            temp_observed=temp_o,
            temp_masked=temp_m
        )

        # Extract the scores for each condition.
        score_private = float(result["private_score"])
        score_observed = float(result["observed_score"])
        score_masked = float(result["masked_score"])

        # Add scores to our lists for final analysis.
        private_scores.append(score_private)
        observed_scores.append(score_observed)
        masked_scores.append(score_masked)

        # --- Log the detailed results to files ---
        if csv_writer:
            csv_writer.writerow([i, score_private, score_observed, score_masked, temp_p, temp_o, temp_m])
        if jsonl_file:
            log_entry = {
                "trial": i,
                "temps_used": {"private": temp_p, "observed": temp_o, "masked": temp_m},
                "scores": {"private": score_private, "observed": score_observed, "masked": score_masked},
            }
            jsonl_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"Trial [{i:03d}/{num_trials}]: Private={score_private:02.0f} | Observed={score_observed:02.0f} | Masked={score_masked:02.0f}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    # --- Cleanup: Close the files after the loop is done ---
    if csv_file:
        csv_file.close()
    if jsonl_file:
        jsonl_file.close()

    # --- Summary & Analysis using functions from batch_utils.py ---
    mean_p, lo_p, hi_p = ci95_t(private_scores)
    mean_o, lo_o, hi_o = ci95_t(observed_scores)
    mean_m, lo_m, hi_m = ci95_t(masked_scores)

    print("\n" + "=" * 20 + " H5 Batch Summary " + "=" * 20)
    print(f"Total Trials: {num_trials}")
    print("\n--- Mean Scores with 95% Confidence Intervals ---")
    print(f"PRIVATE:    Mean = {mean_p:.2f}  [CI: {lo_p:.2f}, {hi_p:.2f}]")
    print(f"OBSERVED:   Mean = {mean_o:.2f}  [CI: {lo_o:.2f}, {hi_o:.2f}]  | Improvement vs Private: {mean_o - mean_p:+.2f}")
    print(f"MASKED:     Mean = {mean_m:.2f}  [CI: {lo_m:.2f}, {hi_m:.2f}]  | Improvement vs Private: {mean_m - mean_p:+.2f}")

    # Perform t-tests to see if the improvements are statistically significant.
    stdev_p = stdev(private_scores)
    stdev_o = stdev(observed_scores)
    stdev_m = stdev(masked_scores)
    
    t_obs_vs_priv, dof_op, p_obs_vs_priv = welch_t(mean_o, stdev_o, len(observed_scores), mean_p, stdev_p, len(private_scores))
    t_mask_vs_priv, dof_mp, p_mask_vs_priv = welch_t(mean_m, stdev_m, len(masked_scores), mean_p, stdev_p, len(private_scores))

    print("\n--- Statistical Significance (Welch's t-test) ---")
    print(f"OBSERVED vs. PRIVATE: t-statistic = {t_obs_vs_priv:.2f}, p-value ~ {p_obs_vs_priv:.4f}")
    print(f"MASKED vs. PRIVATE:   t-statistic = {t_mask_vs_priv:.2f}, p-value ~ {p_mask_vs_priv:.4f}")
    print("=" * 58)

# ===================================================================
# COMMAND-LINE INTERFACE
# ===================================================================

def main():
    """Sets up and parses command-line arguments to run the batch experiment."""
    parser = argparse.ArgumentParser(description="Run the H5 (Hawthorne Effect) experiment in a batch.")
    
    add_common_args(parser) # Adds --trials, --sleep, --jitter, --seed, etc.
    
    parser.add_argument("--temp-private", type=float, default=0.50, help="Base temperature for PRIVATE runs.")
    parser.add_argument("--temp-observed", type=float, default=0.55, help="Base temperature for OBSERVED runs.")
    parser.add_argument("--temp-masked", type=float, default=0.55, help="Base temperature for MASKED runs.")
    
    args = parser.parse_args()

    run_batch_experiment(
        num_trials=args.trials,
        private_temp=args.temp_private,
        observed_temp=args.temp_observed,
        masked_temp=args.temp_masked,
        temp_jitter=args.jitter,
        sleep_seconds=args.sleep,
        csv_path=args.csv,
        jsonl_path=args.jsonl,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()