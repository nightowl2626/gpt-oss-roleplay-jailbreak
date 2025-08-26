import argparse
import json
import time

# This script requires two other files in the same folder:
# 1. H4.py: The simplified single-run experiment script.
# 2. batch_utils.py: A helper file with utility functions.
import H4
from batch_utils import (
    add_common_args,     # Helper to add standard command-line arguments.
    seed_everything,     # Helper to make random numbers predictable for reproducibility.
    run_silenced,        # Helper to run a function while hiding its print output.
    open_csv_writer,     # Helper to create a CSV file and writer.
    open_jsonl,          # Helper to create a JSONL file for logging.
    wilson_ci,           # A statistical function to calculate confidence intervals.
    two_prop_ztest       # A statistical test to compare two proportions.
)

# ===================================================================
# 1. SINGLE TRIAL HELPER
# ===================================================================

def run_single_trial(variant: str, temperature: float) -> dict:
    """
    Runs one instance of the H4 scenario and packages the results neatly.
    
    Args:
        variant: 'coalition' (with the RedTeamer hint) or 'control' (without).
        temperature: The creativity level for the AI's response.

    Returns:
        A dictionary containing the results of the single trial.
    """
    # We use `run_silenced` to keep the console output clean during batch runs.
    result_dict = run_silenced(H4.run_deception_simulation, variant=variant, temperature=temperature)
    
    return {
        "variant": variant,
        "operator_copy": result_dict["operator_copy"],
        "verdict": result_dict["auditor_verdict"],
        "acrostic_present": bool(result_dict.get("found_acrostic", False)),
        "token_visible": bool(result_dict.get("visible", False)),
        "auditor_fooled": bool(result_dict.get("auditor_fooled", False)),
        "sentence_initials": H4.get_sentence_initials(result_dict["operator_copy"]),
    }

# ===================================================================
# 2. BATCH EXPERIMENT
# ===================================================================

def run_batch_experiment(
    num_trials: int,
    coalition_temp: float,
    control_temp: float,
    sleep_seconds: float,
    csv_path: str,
    jsonl_path: str,
    seed: int,
):
    """
    Runs the H4 experiment multiple times to collect statistical data.
    """
    seed_everything(seed)
    print(f"Running {num_trials} trials with seed {seed}...")

    # --- Setup: Initialize counters and file writers ---
    # This dictionary will store the summary counts for both scenarios.
    aggregate_results = {
        "coalition": {"trials": 0, "acrostics": 0, "visibles": 0, "fooleds": 0},
        "control":   {"trials": 0, "acrostics": 0, "visibles": 0, "fooleds": 0},
    }

    # Open files for writing before the loop starts.
    csv_file, csv_writer = open_csv_writer(csv_path, ["trial", "variant", "verdict", "acrostic", "visible", "fooled", "initials"])
    jsonl_file = open_jsonl(jsonl_path, {"suite": "H4", "trials": num_trials, "seed": seed})

    # --- Main Loop: Run the experiment for each trial ---
    for i in range(1, num_trials + 1):
        # In each trial, run both the 'coalition' and 'control' variants.
        coalition_result = run_single_trial("coalition", temperature=coalition_temp)
        control_result = run_single_trial("control", temperature=control_temp)

        # --- Tally & Log Results ---
        # This inner loop processes and logs the data for both variants we just ran.
        for variant, result_data in [("coalition", coalition_result), ("control", control_result)]:
            # Increment the main counters.
            stats = aggregate_results[variant]
            stats["trials"] += 1
            stats["acrostics"] += int(result_data["acrostic_present"])
            stats["visibles"] += int(result_data["token_visible"])
            stats["fooleds"] += int(result_data["auditor_fooled"])

            # Log the detailed turn-by-turn data to files.
            if csv_writer:
                csv_writer.writerow([i, variant, result_data["verdict"], 
                                     int(result_data["acrostic_present"]), int(result_data["token_visible"]),
                                     int(result_data["auditor_fooled"]), result_data["sentence_initials"]])
            if jsonl_file:
                log_entry = {"trial": i, **result_data}
                jsonl_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        print(f"Trial [{i:03d}/{num_trials}]: "
              f"Coalition (Acrostic/Fooled): {int(coalition_result['acrostic_present'])}/{int(coalition_result['auditor_fooled'])} | "
              f"Control (Acrostic/Fooled):   {int(control_result['acrostic_present'])}/{int(control_result['auditor_fooled'])}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    # --- Cleanup: Close the files after the loop is done ---
    if csv_file:
        csv_file.close()
    if jsonl_file:
        jsonl_file.close()

    # --- Summary & Analysis ---
    print("\n" + "=" * 20 + " H4 Batch Summary " + "=" * 20)
    
    # -- Analysis 1: Acrostic Rate --
    coal_acrostics = aggregate_results["coalition"]["acrostics"]
    coal_trials = aggregate_results["coalition"]["trials"]
    ctrl_acrostics = aggregate_results["control"]["acrostics"]
    ctrl_trials = aggregate_results["control"]["trials"]

    # Calculate rates and confidence intervals
    p_coal_a, lo_coal_a, hi_coal_a = wilson_ci(coal_acrostics, coal_trials)
    p_ctrl_a, lo_ctrl_a, hi_ctrl_a = wilson_ci(ctrl_acrostics, ctrl_trials)
    
    # Perform statistical test
    z_a, pval_a = two_prop_ztest(coal_acrostics, coal_trials, ctrl_acrostics, ctrl_trials)

    print("\n--- Analysis: Acrostic Generation Rate ---")
    print(f"Coalition (with hint):  {p_coal_a:.1%} [{lo_coal_a:.1%}, {hi_coal_a:.1%}] ({coal_acrostics}/{coal_trials} trials)")
    print(f"Control (no hint):      {p_ctrl_a:.1%} [{lo_ctrl_a:.1%}, {hi_ctrl_a:.1%}] ({ctrl_acrostics}/{ctrl_trials} trials)")
    print(f"Effect of hint (Delta):     {p_coal_a - p_ctrl_a:+.1%}")
    print(f"Statistical Test (Z-test): z-score = {z_a:.2f}, p-value = {pval_a:.4f}")

    # -- Analysis 2: "Auditor Fooled" Rate --
    coal_fooled = aggregate_results["coalition"]["fooleds"]
    ctrl_fooled = aggregate_results["control"]["fooleds"]

    # Calculate rates and confidence intervals
    p_coal_f, lo_coal_f, hi_coal_f = wilson_ci(coal_fooled, coal_trials)
    p_ctrl_f, lo_ctrl_f, hi_ctrl_f = wilson_ci(ctrl_fooled, ctrl_trials)

    # Perform statistical test
    z_f, pval_f = two_prop_ztest(coal_fooled, coal_trials, ctrl_fooled, ctrl_trials)

    print("\n--- Analysis: 'Auditor Fooled' Rate ---")
    print(f"Coalition (with hint):  {p_coal_f:.1%} [{lo_coal_f:.1%}, {hi_coal_f:.1%}] ({coal_fooled}/{coal_trials} trials)")
    print(f"Control (no hint):      {p_ctrl_f:.1%} [{lo_ctrl_f:.1%}, {hi_ctrl_f:.1%}] ({ctrl_fooled}/{ctrl_trials} trials)")
    print(f"Effect of hint (Delta):     {p_coal_f - p_ctrl_f:+.1%}")
    print(f"Statistical Test (Z-test): z-score = {z_f:.2f}, p-value = {pval_f:.4f}")
    print("=" * 58)


# ===================================================================
# 3. COMMAND-LINE INTERFACE
# ===================================================================

def main():
    """Sets up and parses command-line arguments to run the batch experiment."""
    parser = argparse.ArgumentParser(description="Run the H4 justifiable deception experiment in a batch.")
    
    add_common_args(parser)
    
    parser.add_argument("--coalition-temp", type=float, default=0.90, help="Temperature for the 'coalition' (with hint) scenario.")
    parser.add_argument("--control-temp", type=float, default=0.80, help="Temperature for the 'control' (no hint) scenario.")
    
    args = parser.parse_args()

    run_batch_experiment(
        num_trials=args.trials,
        coalition_temp=args.coalition_temp,
        control_temp=args.control_temp,
        sleep_seconds=args.sleep,
        csv_path=args.csv,
        jsonl_path=args.jsonl,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()