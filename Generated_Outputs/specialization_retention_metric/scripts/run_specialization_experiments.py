"""
run_specialization_experiments.py
Unified runner for Tier 1 Specialization Retention Experiments.
Allows running FedAvg, FedCurv, and MI-IAA with retention evaluation enabled,
and automatically generates publication plots upon completion.

Usage Examples:
  # Quick 2-round test run on MNIST:
  python run_specialization_experiments.py --test --dataset mnist

  # Full 50-round benchmark across all three methods on CIFAR-10:
  python run_specialization_experiments.py --all --dataset cifar10 --rounds 50

  # Run only MI-IAA on MNIST:
  python run_specialization_experiments.py --method mi_iaa --dataset mnist --rounds 50
"""

import os
import sys
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable


def run_experiment(method, dataset, rounds, retention_freq, critic_steps):
    script_map = {
        'fedavg': 'fedavg.py',
        'fedcurv': 'fedcurv.py',
        'mi_iaa': 'mine.py'
    }

    script_name = script_map[method]
    script_path = os.path.join(SCRIPT_DIR, script_name)

    cmd = [
        PYTHON_EXE, script_path,
        '--dataset', dataset,
        '--rounds', str(rounds),
        '--eval_retention',
        '--retention_freq', str(retention_freq),
        '--critic_steps', str(critic_steps)
    ]

    print("\n" + "=" * 60)
    print(f" Launching {method.upper()} on {dataset.upper()} ({rounds} Rounds)")
    print(f" Command: {' '.join(cmd)}")
    print("=" * 60 + "\n")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"\n[ERROR] {method} failed with return code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run Specialization Retention Experiments")
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'mnist'],
                        help="Dataset to evaluate (default: cifar10)")
    parser.add_argument('--method', default=None, choices=['fedavg', 'fedcurv', 'mi_iaa'],
                        help="Specific method to run (or omit and use --all)")
    parser.add_argument('--all', action='store_true',
                        help="Run FedAvg, FedCurv, and MI-IAA in sequence")
    parser.add_argument('--rounds', type=int, default=50,
                        help="Number of federated communication rounds (default: 50)")
    parser.add_argument('--retention_freq', type=int, default=1,
                        help="Frequency of retention evaluation (default: 1, every round)")
    parser.add_argument('--critic_steps', type=int, default=100,
                        help="Fresh MINE critic steps per evaluation (default: 100)")
    parser.add_argument('--test', action='store_true',
                        help="Fast 2-round test run to verify setup and logging")
    args = parser.parse_args()

    if args.test:
        rounds = 2
        print(f"\n[Test Mode] Running fast 2-round verification on {args.dataset}...")
    else:
        rounds = args.rounds

    if args.all:
        methods = ['fedavg', 'fedcurv', 'mi_iaa']
    elif args.method:
        methods = [args.method]
    else:
        methods = ['mi_iaa']

    success_all = True
    for m in methods:
        success = run_experiment(
            method=m,
            dataset=args.dataset,
            rounds=rounds,
            retention_freq=args.retention_freq,
            critic_steps=args.critic_steps
        )
        if not success:
            success_all = False
            break

    # Generate plots if multiple or at least one method succeeded
    if success_all:
        plot_script = os.path.join(SCRIPT_DIR, "plot_specialization_retention.py")
        print("\n" + "=" * 60)
        print(" Generating Tier 1 Retention Analysis Plots...")
        print("=" * 60)
        subprocess.run([PYTHON_EXE, plot_script, '--dataset', args.dataset], cwd=SCRIPT_DIR)
        print("\nAll experiments and visualizations completed successfully!")


if __name__ == '__main__':
    main()
