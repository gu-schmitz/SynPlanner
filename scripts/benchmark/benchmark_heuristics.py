#!/usr/bin/env python3
"""
Benchmark a variety of heuristics across all SAScore bins using the ranking policy.

For each heuristic x bin combination, runs MCTS search in parallel and saves
solved/unsolved SMILES to: results/{score_function}/{bin_type}/
"""

from pathlib import Path
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent

SCORE_FUNCTIONS = [
    "sascore",
    "scscore", 
    "syba",
    "heavyAtomCount",
    "weight",
    "weightXsascore",
    "WxWxSAS",
    "heavyatomsXsascore",
    "heavyatomsXscscore",
    "sascoreXscscore",
]

BIN_TYPES = [
    "1.5_2.5",
    "2.5_3.5",
    "3.5_4.5",
    "4.5_5.5",
    "5.5_6.5",
    "6.5_7.5",
    "7.5_8.5",
]

NUM_CPU = 8 
# syba loads a 26M-entry fragment dict (~5 GB per worker process).
# Cap its worker count so the total stays within available RAM.
# Adjust if your machine has more or less than 128 GB.
HEURISTIC_MAX_WORKERS = {
    "syba": 15,  # 15 × 5 GB ≈ 75 GB
}
RESULTS_ROOT = HERE / "heuristics_benchmark"


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def run_one(smi, tree_config, reaction_rules, building_blocks,
            policy_network, evaluation_function):
    try:
        import os
        from synplan.chem.utils import mol_from_smiles
        from synplan.mcts.tree import Tree

        mol = mol_from_smiles(smi)
        tree = Tree(
            target=mol,
            config=tree_config,
            reaction_rules=reaction_rules,
            building_blocks=building_blocks,
            expansion_function=policy_network,
            evaluation_function=evaluation_function,
        )

        n_iters = 0
        for solved, node_id in tree:
            n_iters += 1
            if solved:
                tree._tqdm = None
                return smi, 1

        if n_iters == 0:
            print(f"[PID {os.getpid()}] 0-ITER EXIT: {smi[:50]}", flush=True)

        tree._tqdm = None
        return smi, 0

    except Exception as e:
        import traceback
        print(f"[PID {os.getpid()}] EXCEPTION for {smi[:30]}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return smi, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from synplan.utils.loading import (
        download_preset,
        load_building_blocks,
        load_reaction_rules,
        load_policy_function,
        load_evaluation_function,
    )
    from synplan.utils.config import TreeConfig, RDKitEvaluationConfig

    # Load shared resources once
    print("Loading resources...")
    paths = download_preset("synplanner-article", save_to=HERE / "synplan_data")
    reaction_rules = load_reaction_rules(paths["reaction_rules"])
    building_blocks = load_building_blocks(paths["building_blocks"], standardize=False)
    policy_network = load_policy_function(weights_path=paths["ranking_policy"])
    policy_network.config.top_rules = 50
    print(f"  {len(reaction_rules)} rules, {len(building_blocks)} building blocks\n")

    tree_config = TreeConfig(
        search_strategy="expansion_first",
        max_iterations=100000, # unreachable within 600 seconds 
        max_time=600,
        #max_iterations=1000, # compare heuristic computing time
        #max_time=6000,
        max_depth=15,
        min_mol_size=6,
        init_node_value=0.5,
        ucb_type="uct",
        c_ucb=0.1,
        evaluation_agg="max",
    )

    RESULTS_ROOT.mkdir(exist_ok=True)

    # Summary table accumulated across all runs
    summary = {}  # (score_function, bin_type) -> (solved, total)

    for score_function in SCORE_FUNCTIONS:
        print(f"{'='*60}")
        print(f"Heuristic: {score_function}")
        print(f"{'='*60}")

        eval_config = RDKitEvaluationConfig(score_function=score_function)
        evaluation_function = load_evaluation_function(eval_config)

        for bin_type in BIN_TYPES:
            target_path = HERE / "sascore" / f"targets_with_sascore_{bin_type}.smi"
            if not target_path.exists():
                print(f"  [{bin_type}] target file not found, skipping")
                continue

            smiles = [
                line.strip().split()[0]
                for line in open(target_path, encoding="utf-8")
                if line.strip() and not line.startswith("#")
            ]

            n_jobs = HEURISTIC_MAX_WORKERS.get(score_function, NUM_CPU)
            print(f"  [{bin_type}] {len(smiles)} molecules, {n_jobs} workers...")

            results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=50)(
                delayed(run_one)(
                    smi, tree_config, reaction_rules, building_blocks,
                    policy_network, evaluation_function,
                )
                for smi in smiles
            )

            solved_smiles = [smi for smi, ok in results if ok]
            unsolved_smiles = [smi for smi, ok in results if not ok]
            n_solved = len(solved_smiles)
            n_total = len(smiles)

            print(f"  [{bin_type}] {n_solved} / {n_total} solved "
                  f"({100 * n_solved / n_total:.1f}%)")

            summary[(score_function, bin_type)] = (n_solved, n_total)

            # Save results
            out_dir = RESULTS_ROOT / score_function / bin_type
            out_dir.mkdir(parents=True, exist_ok=True)

            (out_dir / "solved.smi").write_text(
                "\n".join(solved_smiles) + ("\n" if solved_smiles else ""),
                encoding="utf-8",
            )
            (out_dir / "unsolved.smi").write_text(
                "\n".join(unsolved_smiles) + ("\n" if unsolved_smiles else ""),
                encoding="utf-8",
            )

        print()

    # Print final summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Heuristic':<25}" + "".join(f"{b:>10}" for b in BIN_TYPES)
    print(header)
    print("-" * len(header))
    for score_function in SCORE_FUNCTIONS:
        row = f"{score_function:<25}"
        for bin_type in BIN_TYPES:
            if (score_function, bin_type) in summary:
                n, total = summary[(score_function, bin_type)]
                row += f"{n:>5}/{total:<4}"
            else:
                row += f"{'N/A':>10}"
        print(row)


if __name__ == "__main__":
    main()
