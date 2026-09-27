"""Run the calculations used in the paper."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def run(script, *args):
    print(f'\nRunning {script}', flush=True)
    subprocess.run([sys.executable, script, *args], cwd=HERE, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', nargs='?', default='all',
                        choices=('all', 'empirical', 'model', 'rankings', 'power'))
    args = parser.parse_args()
    cache = HERE / 'results/.matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    if args.part in ('all', 'empirical'):
        run('empirical.py')
    if args.part in ('all', 'model', 'power'):
        run('experiment_predictions.py')
        run('exact_arithmetic.py')
        run('verify_exact.py')
        subprocess.run([sys.executable, '-m', 'unittest', 'discover'], cwd=HERE, check=True)
    if args.part in ('all', 'rankings'):
        run('ranking_simulation.py')
    if args.part in ('all', 'power'):
        run('experiment_power.py')
        run('power_tables.py')
        run('null_checks.py')
        run('allocation_comparison.py')
        run('experiment_budget.py')
    if args.part == 'all':
        run('check_results.py')


if __name__ == '__main__':
    main()
