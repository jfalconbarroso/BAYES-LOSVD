import re
import sys
import glob
import argparse
import warnings
import pandas as pd
import lib.misc_functions as misc
from collections import defaultdict, Counter
#==============================================================================
def check_results(filelist, RHAT_MAX, ESS_MIN):

    problems = defaultdict(list)
    bin_problem_counts = Counter()

    for file in sorted(filelist):

        bin_id = file.split("bin")[-1].split(".")[0]

        with open(file) as f:
            lines = [re.split(r"\s+", line.strip()) for line in f if line.strip()]

        if not lines:
            continue

        # Detect header line
        if any("r_hat" in col for col in lines[0]):
            header, data = lines[0], lines[1:]
        else:
            header, data = None, lines

        df = pd.DataFrame(data)
        ncols = df.shape[1]

        # Always name first column 'variable'
        colnames = ["variable"] + [f"col{i}" for i in range(1, ncols)]
        if ncols == 8:
            colnames = ["variable", "mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "ess_tail", "r_hat"]
        elif ncols == 9:
            colnames = ["variable", "mean", "sd", "mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat", "extra"]
        elif ncols == 10:
            colnames = ["variable", "mean", "sd", "mcse_mean", "mcse_sd", "hdi_3%", "hdi_97%", "ess_bulk", "ess_tail", "r_hat"]
        df.columns = colnames[:ncols]

        # Convert numeric columns
        for c in df.columns:
            if c != "variable":
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Keep only variable-like names
        df = df[df["variable"].str.match(r"[A-Za-z_]", na=False)]

        # Find problematic rows
        if all(col in df.columns for col in ["r_hat", "ess_bulk", "ess_tail"]):
            bad = df[(df["r_hat"] > RHAT_MAX) | (df["ess_bulk"] < ESS_MIN) | (df["ess_tail"] < ESS_MIN)]
            if not bad.empty:
                bin_problem_counts[bin_id] = len(bad)
                for var in bad["variable"]:
                    problems[var].append(bin_id)

    # ---- REPORT ----
    print("# Concise Fit Report -------")
    if not problems:
        print("✅ All fits look good in all bins!")
    else:
        for var, bins in sorted(problems.items()):
            print(f"{var:20s} -> bad in bins: {', '.join(map(str, bins))}")

    # ---- SUMMARY ----
    print("\n# Summary by Number of Problematic Variables per Bin -------")
    if not bin_problem_counts:
        print("✅ No bins have problematic variables.")
    else:
        group_1_5 = sum(1 for v in bin_problem_counts.values() if 1 <= v <= 5)
        group_6_10 = sum(1 for v in bin_problem_counts.values() if 6 <= v <= 10)
        group_10p = sum(1 for v in bin_problem_counts.values() if v > 10)
        total_bins = len(bin_problem_counts)

        print(f"Total bins with problems: {total_bins}")
        print(f"  • 1–5 bad variables   : {group_1_5}")
        print(f"  • 6–10 bad variables  : {group_6_10}")
        print(f"  • >10 bad variables   : {group_10p}")
        print("")

    return

#==============================================================================
if (__name__ == '__main__'):

    warnings.filterwarnings("ignore")

    print("===========================================")
    print("               BAYES-LOSVD                 ")
    print("             (check_results)               ")
    print("===========================================")
    print("")

    parser = argparse.ArgumentParser(
        prog="myscript.py",
        usage="%(prog)s -f file [options]",
        description="Process spectra with configurable parameters."
    )

    parser.add_argument("-d", "--dirname",   type=str,   default=None, help="Directory with the results")
    parser.add_argument("-r", "--rhat_max",  type=float, default=1.1,  help="Maximum Rhat for flagging")
    parser.add_argument("-e", "--ess_min",   type=float, default=100,  help="Minimum ESS for flagging")

    args = parser.parse_args()

    filelist = glob.glob(args.dirname+"/*_summary_bin*.txt")
    if filelist == []:
       misc.printFAILED("ERROR: cannot find results file in "+args.dirname)
       sys.exit()

    check_results(filelist, args.rhat_max, args.ess_min)

    # --- END -------------------------------------------------------------
    misc.printDONE("FINISHED!")
    sys.exit()
