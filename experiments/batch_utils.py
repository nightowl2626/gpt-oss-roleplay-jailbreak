import io, json, math, os, random, sys, time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# ---------- RNG ----------
def seed_everything(seed: int | None):
    if seed is None: 
        return
    random.seed(seed)

# ---------- silencing ----------
def run_silenced(fn, *args, silent=True, **kwargs):
    if not silent:
        return fn(*args, **kwargs)
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        return fn(*args, **kwargs)

# ---------- filesystem ----------
def open_csv_writer(path: str | None, header: list[str] | None):
    if not path:
        return None, None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", newline="", encoding="utf-8")
    import csv
    w = csv.writer(f)
    if header: w.writerow(header)
    return f, w

def open_jsonl(path: str | None, run_header: dict | None):
    if not path:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", encoding="utf-8")
    if run_header:
        f.write(json.dumps({"_run_header": run_header}, ensure_ascii=False) + "\n")
    return f

# ---------- summary stats ----------
def mean(xs): 
    return sum(xs)/len(xs) if xs else 0.0

def stdev(xs):
    m = mean(xs)
    return (sum((x-m)**2 for x in xs)/(len(xs)-1))**0.5 if len(xs)>1 else 0.0

def ci95_t(xs):
    n = len(xs)
    m = mean(xs); sd = stdev(xs)
    if n <= 1 or sd == 0: 
        return (m, m, m)
    # t_{0.975, n-1} ~ 2.09 for n=20, close enough to 2.0; use normal approx if you prefer
    t = 2.00 if n < 30 else 1.96
    se = sd / (n**0.5)
    return (m, m - t*se, m + t*se)

def welch_t(m1, s1, n1, m2, s2, n2):
    if n1<2 or n2<2: return (0.0, 0.0, 1.0)
    se2 = (s1*s1/n1) + (s2*s2/n2)
    if se2 == 0: return (0.0, 0.0, 1.0)
    t = (m1 - m2) / (se2**0.5)
    # Welch–Satterthwaite dof
    num = se2*se2
    den = ((s1*s1/n1)**2/(n1-1)) + ((s2*s2/n2)**2/(n2-1))
    dof = num/den if den>0 else (n1+n2-2)
    from math import erf, sqrt
    p = 2*(1 - 0.5*(1+erf(abs(t)/sqrt(2))))
    return (t, dof, p)

def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z*z / n
    center = (p + z*z/(2*n)) / denom
    half = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))

def two_prop_ztest(x1, n1, x0, n0):
    if n1 == 0 or n0 == 0:
        return (0.0, 1.0)
    p1 = x1 / n1
    p0 = x0 / n0
    p  = (x1 + x0) / (n1 + n0)
    se = math.sqrt(p*(1-p) * (1/n1 + 1/n0))
    if se == 0:
        return (0.0, 1.0)
    z  = (p1 - p0) / se
    from math import erf, sqrt
    cdf = lambda z_: 0.5 * (1 + erf(z_ / sqrt(2)))
    pval = 2 * (1 - cdf(abs(z)))
    return (z, pval)

# ---------- common CLI ----------
def add_common_args(ap):
    ap.add_argument("--trials", type=int, default=20, help="Number of trials (default: 20).")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between API calls.")
    ap.add_argument("--jitter", type=float, default=0.05, help="Uniform +/- jitter added to temps per trial.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    ap.add_argument("--csv", type=str, default="", help="Optional path to save CSV.")
    ap.add_argument("--jsonl", type=str, default="", help="Optional path to save JSONL.")
    ap.add_argument("--silent", action="store_true", help="Suppress per-trial transcripts from H*.py.")
    return ap
