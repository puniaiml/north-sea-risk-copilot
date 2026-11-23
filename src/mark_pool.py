"""Mark the retrieval pool in the terminal and merge the marks of several raters.

Each rater works on a private copy, data/benchmark/pool_<name>.csv, shown in random order with
the retrieval method hidden so that nobody is nudged towards one method. Progress is saved after
every mark and the session can be stopped and resumed.

Relevance rule (agree it in the group first): mark 1 when the document contains information that a
supervisor or technician planning offshore work could use to answer the query, such as guidance on the
same hazard or operation, or an incident whose lessons apply to it. Mark 0 when it only shares words.

Usage:
    python src/mark_pool.py mark --rater puni
    python src/mark_pool.py status --rater puni
    python src/mark_pool.py merge puni raterB raterC
    python src/mark_pool.py merge puni            (one rater: copies the marks into pool.csv)
Merge writes the agreed marks into pool.csv, leaves disagreements blank, lists them in
data/benchmark/disagreements.csv and prints agreement and Cohen's kappa for each pair of raters.
Resolve the blanks together, then run: python src/benchmark_retrieval.py evaluate --core-only
"""

import argparse
import sys
import webbrowser
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

SHOW_COLUMNS = ["qid", "query", "doc_id", "title", "url", "snippet"]


def rater_path(name):
    """Private marking file of one rater."""
    return cfg.BENCH_DIR / f"pool_{name}.csv"


def start(name, pool_path=None):
    """Load the rater's file, or create it from pool.csv with shuffled documents and no method column."""
    path = rater_path(name)
    if path.exists():
        return pd.read_csv(path)
    pool = pd.read_csv(pool_path or cfg.POOL_PATH)
    order = {q: i for i, q in enumerate(pool["qid"].unique())}
    pool = pool.sample(frac=1, random_state=cfg.SEED)
    pool = pool.assign(_q=pool["qid"].map(order)).sort_values("_q", kind="stable").drop(columns="_q")
    out = pool[SHOW_COLUMNS].assign(relevant=np.nan).reset_index(drop=True)
    out.to_csv(path, index=False)
    return out


def progress(df):
    """Number of marked and total rows."""
    return int(df["relevant"].notna().sum()), len(df)


def mark(name, pool_path=None, ask=input, show=print, opener=webbrowser.open):
    """Ask for a 1 or 0 for every unmarked row. Returns True when the whole file is marked."""
    df = start(name, pool_path)
    path = rater_path(name)
    for i in df.index[df["relevant"].isna()]:
        row = df.loc[i]
        done, total = progress(df)
        show(f"\n[{done}/{total}] {row['qid']}: {row['query']}")
        show(f"Title: {row['title']}\nURL:   {row['url']}\n{row['snippet']}")
        while True:
            a = ask("1 relevant, 0 not relevant, o open page, s skip, q save and quit > ").strip().lower()
            if a in ("1", "0"):
                df.loc[i, "relevant"] = int(a)
                df.to_csv(path, index=False)
                break
            if a == "o":
                opener(row["url"])
            elif a == "s":
                break
            elif a == "q":
                df.to_csv(path, index=False)
                return False
    return not df["relevant"].isna().any()


def kappa(a, b):
    """Cohen's kappa for two 0/1 arrays."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    po = float((a == b).mean())
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def merge(names, pool_path=None, out_path=None, disagreements_path=None):
    """Write majority marks into pool.csv. Ties and disagreements stay blank. Returns a summary dict."""
    pool_path = Path(pool_path or cfg.POOL_PATH)
    base = pd.read_csv(pool_path)
    key = pd.MultiIndex.from_frame(base[["qid", "doc_id"]])
    marks = {}
    for n in names:
        m = pd.read_csv(rater_path(n)).set_index(["qid", "doc_id"])["relevant"]
        if m.isna().any():
            raise SystemExit(f"{n} has {int(m.isna().sum())} unmarked rows.")
        marks[n] = m.reindex(key).to_numpy()
    stack = pd.DataFrame(marks)
    share = stack.mean(axis=1)
    base["relevant"] = np.where(share > 0.5, 1.0, np.where(share < 0.5, 0.0, np.nan))
    base.to_csv(out_path or pool_path, index=False)
    pairs = {f"{a} vs {b}": dict(agreement=float((stack[a] == stack[b]).mean()), kappa=float(kappa(stack[a], stack[b])))
             for a, b in combinations(names, 2)}
    open_rows = base[base["relevant"].isna()].copy()
    for n in names:
        open_rows[n] = stack.loc[open_rows.index, n].to_numpy()
    open_rows[["qid", "query", "doc_id", "title", "url", *names]].to_csv(
        disagreements_path or cfg.BENCH_DIR / "disagreements.csv", index=False)
    return dict(rows=len(base), relevant=int((base["relevant"] == 1).sum()), unresolved=len(open_rows), pairs=pairs)


def main():
    """Command line entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=["mark", "status", "merge"])
    p.add_argument("names", nargs="*", help="rater names for merge")
    p.add_argument("--rater", help="your name, for mark and status")
    a = p.parse_args()
    if a.step == "merge":
        if not a.names:
            raise SystemExit("Give at least one rater name to merge.")
        res = merge(a.names)
        print(f"{res['rows']} rows, {res['relevant']} marked relevant, {res['unresolved']} unresolved")
        for k, v in res["pairs"].items():
            print(f"{k}: agreement {v['agreement']:.1%}, kappa {v['kappa']:.2f}")
        if res["unresolved"]:
            print(f"Resolve the rows in {cfg.BENCH_DIR / 'disagreements.csv'} and type the agreed 1 or 0 into pool.csv.")
        return
    if not a.rater:
        raise SystemExit("Add --rater yourname.")
    if not cfg.POOL_PATH.exists() and not rater_path(a.rater).exists():
        raise SystemExit(f"{cfg.POOL_PATH} not found. Run: python src/benchmark_retrieval.py pool --core-only")
    if a.step == "status":
        done, total = progress(start(a.rater))
        print(f"{a.rater}: {done} of {total} marked")
        return
    finished = mark(a.rater)
    done, total = progress(pd.read_csv(rater_path(a.rater)))
    print(f"\nSaved {done} of {total}." + (" All marked." if finished else " Run the same command to continue."))


if __name__ == "__main__":
    main()
