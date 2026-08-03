"""Standalone worker script for running cell-eval in an isolated subprocess.

This avoids multiprocessing pool deadlocks that occur when pdex's fork-based
mp.Pool runs inside a PyTorch Lightning process with CUDA contexts and threads.
"""
import argparse
import logging
import sys
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run cell-eval metrics in an isolated process")
    parser.add_argument("--pred", required=True, help="Path to predicted anndata h5ad")
    parser.add_argument("--real", required=True, help="Path to real anndata h5ad")
    parser.add_argument("--control-pert", required=True, help="Control perturbation name")
    parser.add_argument("--pert-col", required=True, help="Perturbation column name")
    parser.add_argument("--num-threads", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--outdir", required=True, help="Output directory for results")
    parser.add_argument("--profile", default="vcc", help="cell-eval profile")
    args = parser.parse_args()

    from cell_eval import MetricsEvaluator

    logger.info(f"Running cell-eval: pred={args.pred}, real={args.real}, outdir={args.outdir}")
    evaluator = MetricsEvaluator(
        adata_pred=args.pred,
        adata_real=args.real,
        control_pert=args.control_pert,
        pert_col=args.pert_col,
        num_threads=args.num_threads,
        batch_size=args.batch_size,
        outdir=args.outdir,
    )
    evaluator.compute(profile=args.profile)
    logger.info("cell-eval completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
