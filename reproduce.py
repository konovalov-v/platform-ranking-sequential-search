"""Run the calculations used in the paper."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def run(script, *args):
    print(f'\nRunning {script}', flush=True)
    subprocess.run([sys.executable, '-m', script, *args], cwd=HERE, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', nargs='?', default='all',
                        choices=('all', 'empirical', 'model', 'rankings', 'power'))
    args = parser.parse_args()
    cache = HERE / 'results/.matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    if args.part in ('all', 'empirical'):
        run('empirical.empirical')
    if args.part in ('all', 'model', 'power'):
        run('experiment.experiment_predictions')
        run('experiment.exact_arithmetic')
        run('experiment.verify_exact')
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], cwd=HERE, check=True)
    if args.part in ('all', 'rankings'):
        run('theory.ranking_simulation')
    if args.part in ('all', 'power'):
        run('experiment.experiment_power')
        run('experiment.power_tables')
        run('experiment.null_checks')
        run('experiment.allocation_comparison')
        run('experiment.experiment_budget')
    if args.part == 'all':
        run('tests.check_results')


if __name__ == '__main__':
    main()
