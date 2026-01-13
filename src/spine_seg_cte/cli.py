"""Command-line interface for spine segmentation."""

import sys
from spine_seg_cte.pipeline import main


def cli():
    """Entry point for command-line interface."""
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
