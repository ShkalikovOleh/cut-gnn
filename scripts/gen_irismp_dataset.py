import argparse
import os

import cvxpy

from cut_gnn.datasets import generate_iris_dataset, load_iris_df


def main(args: argparse.Namespace) -> None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    iris_df = load_iris_df(os.path.join(parent_dir, "data", "iris.csv"))
    generate_iris_dataset(
        n_graphs=args.n_graphs,
        out_path=args.out_dir,
        iris_df=iris_df,
        n_nodes_min=args.n_nodes_min,
        n_nodes_max=args.n_nodes_max,
        sigma=args.sigma,
        n_sample_feat=args.n_sample_feat,
        include_cycles=args.include_cycles,
        max_workers=args.max_workers,
        solver=args.solver,
        start_idx=args.start_idx,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate IrisMP dataset")
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--n_graphs", default=22000, type=int)
    parser.add_argument("--n_nodes_min", default=16, type=int)
    parser.add_argument("--n_nodes_max", default=24, type=int)
    parser.add_argument("--sigma", default=0.6, type=float)
    parser.add_argument("--n_sample_feat", default=2, type=int)
    parser.add_argument(
        "--no-cycles",
        action="store_false",
        dest="include_cycles",
        default=True,
    )
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument(
        "--solver",
        type=str,
        default=cvxpy.GLPK_MI,
        choices=[cvxpy.GLPK_MI, cvxpy.GUROBI],
    )
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--chunksize", default=16, type=int)

    args = parser.parse_args()

    main(args)
