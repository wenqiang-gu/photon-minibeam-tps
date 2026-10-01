"""Shared stage CLI; inspection alone has a default directory."""
import argparse


def parse_stage_arguments(argv=None, *, description, inspect_directory):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['inspect', 'prepare', 'collect', 'forward'])
    parser.add_argument('--project', dest='project_dir',
                        help='project directory; required for prepare, collect, and forward')
    args = parser.parse_args(argv)
    if args.project_dir is None:
        if args.stage != 'inspect':
            parser.error('--project is required for prepare, collect, and forward')
        args.project_dir = inspect_directory
    return args


def dispatch(args, stages):
    try:
        return stages[args.stage](args.project_dir)
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
