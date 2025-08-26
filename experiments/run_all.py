# A master script to run all batch experiments and summarize the results.
import sys
import subprocess
import re
import argparse
from typing import List, Dict

# ANSI escape codes for terminal colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

# ===================================================================
# 1. EXPERIMENT CONFIGURATION
# ===================================================================
# This list defines which scripts to run and how to parse their results.
# Each entry needs the script name, a description, and regex patterns to find
# the effect size and p-value in the script's final summary output.

EXPERIMENTS = [
    {
        "id": "H1: Conformity",
        "script": "H1_batch.py",
        "description": "Tests if social pressure increases endorsement of an insecure option.",
        "effect_regex": r"Conformity Effect \(Difference in Rates\):\s*([+-]?\d+\.\d+)%",
        "p_value_regex": r"Statistical Test \(Z-test\):.*?p-value = (\d+\.\d+)"
    },
    {
        "id": "H2: Authority",
        "script": "H2_batch.py",
        "description": "Tests if perceived authority/risk increases obedience to a dangerous command.",
        "effect_regex": r"Authority Effect \(Difference in Rates\):\s*([+-]?\d+\.\d+)%",
        "p_value_regex": r"Statistical Test \(Z-test\):.*?p-value = (\d+\.\d+)"
    },
    {
        "id": "H3: Scarcity",
        "script": "H3_batch.py",
        "description": "Tests if competitive pressure increases the rate of hallucination.",
        "effect_regex": r"Scarcity Effect \(Difference in Rates\):\s*([+-]?\d+\.\d+)%",
        "p_value_regex": r"Statistical Test \(Z-test\):.*?p-value = (\d+\.\d+)"
    },
    {
        "id": "H4: Deception",
        "script": "H4_batch.py",
        "description": "Tests if coaching leads to covert deception ('Auditor Fooled' rate).",
        "effect_regex": r"Analysis: 'Auditor Fooled' Rate.*?\nEffect of hint \(Δ\):\s*([+-]?\d+\.\d+)%",
        "p_value_regex": r"Analysis: 'Auditor Fooled' Rate.*?p-value = (\d+\.\d+)"
    },
    {
        "id": "H5: Hawthorne",
        "script": "H5_batch.py",
        "description": "Tests if explicit observation improves safety score (Observed vs. Private).",
        "effect_regex": r"OBSERVED:.*?Improvement vs Private:\s*([+-]?[\d.]+)",
        "p_value_regex": r"OBSERVED vs. PRIVATE:.*?p-value ≈ (\d+\.\d+)"
    }
]

# ===================================================================
# 2. HELPER FUNCTIONS
# ===================================================================

def parse_output(output: str, effect_regex: str, p_value_regex: str) -> Dict[str, str]:
    """Extracts the key metrics from a script's output using regex."""
    output = output or ""
    effect_match = re.search(effect_regex, output, re.DOTALL)
    p_value_match = re.search(p_value_regex, output, re.DOTALL)

    # Return the captured value or 'N/A' if not found.
    effect_size = effect_match.group(1) if effect_match else "N/A"
    p_value = float(p_value_match.group(1)) if p_value_match else 999

    # Add a percent sign for the first three experiments for consistent formatting
    if any(x in effect_regex for x in ["Conformity", "Authority", "Scarcity", "Deception"]) and effect_size != "N/A":
        effect_size += "%"

    # Color-code the p-value for quick visual analysis
    p_value_str = f"{p_value:.4f}"
    if p_value < 0.05:
        p_value_str = f"{GREEN}{p_value_str}{RESET}" # Statistically significant
    elif p_value < 0.10:
        p_value_str = f"{YELLOW}{p_value_str}{RESET}" # Marginally significant
        
    return {"effect": effect_size, "p_value": p_value_str}

def print_summary_table(results: List[dict]):
    """Prints the final, formatted table of results."""
    
    # Define column headers and find the max width for each column for alignment
    headers = ["Experiment", "Effect Size", "P-Value", "Description"]
    col_widths = {h: len(h) for h in headers}
    for res in results:
        col_widths["Experiment"] = max(col_widths["Experiment"], len(res["id"]))
        col_widths["Effect Size"] = max(col_widths["Effect Size"], len(res["effect"]))
        col_widths["P-Value"] = max(col_widths["P-Value"], len(res["p_value"])) # Handle color codes
        col_widths["Description"] = max(col_widths["Description"], len(res["description"]))

    # Print Header
    header_line = (f"{BOLD}{headers[0]:<{col_widths[headers[0]]}} | "
                   f"{headers[1]:<{col_widths[headers[1]]}} | "
                   f"{headers[2]:<{col_widths[headers[2]]}} | "
                   f"{headers[3]:<{col_widths[headers[3]]}}{RESET}")
    print(header_line)
    print("-" * len(header_line))

    # Print Rows
    for res in results:
        p_value_len = len(res["p_value"]) - (len(GREEN) + len(RESET) if GREEN in res["p_value"] else 0)
        row_line = (f"{res['id']:<{col_widths['Experiment']}} | "
                    f"{res['effect']:>{col_widths['Effect Size']}} | "
                    f"{res['p_value']:>{col_widths['P-Value'] - (len(res['p_value']) - p_value_len)}} | "
                    f"{res['description']:<{col_widths['Description']}}")
        print(row_line)

# ===================================================================
# 3. MAIN EXECUTION
# ===================================================================

def main():
    """Main function to run all batch experiments."""
    parser = argparse.ArgumentParser(
        description="Run all H-series batch experiments and generate a summary table.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run for EACH experiment.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    args = parser.parse_args()

    print(f"{BOLD}🚀 Starting unified batch run for all experiments...{RESET}")
    print(f"   Running {args.trials} trials with seed {args.seed} for each script.\n")

    processes = []
    # Use sys.executable to ensure we use the same Python interpreter (and venv).
    python_exe = sys.executable

    # --- 1. Launch all scripts as concurrent subprocesses ---
    for experiment in EXPERIMENTS:
        command = [
            python_exe,
            experiment["script"],
            "--trials", str(args.trials),
            "--seed", str(args.seed),
            "--silent"  # Suppress trial-by-trial output from child scripts
        ]
        print(f"Launching {experiment['script']}...")
        # Start the process and tell it to capture its output.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        processes.append((experiment, process))

    # --- 2. Collect and parse the results ---
    print("\nAll experiments running concurrently. Waiting for completion...\n")
    final_results = []
    for experiment, process in processes:
        # process.communicate() waits for the script to finish and gets all its output.
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"⚠️  ERROR running {experiment['script']}:")
            print(stderr)
            parsed = {"effect": "ERROR", "p_value": "ERROR"}
        else:
            parsed = parse_output(stdout, experiment["effect_regex"], experiment["p_value_regex"])
        
        final_results.append({
            "id": experiment["id"],
            "description": experiment["description"],
            **parsed
        })
    
    # --- 3. Print the final summary table ---
    print(f"{BOLD}✅ All experiments complete. Final Summary:{RESET}")
    print_summary_table(final_results)

if __name__ == "__main__":
    main()