"""Command-line interface for spine segmentation."""

import sys
import runpy


def main():
    """Entry point for command-line interface."""
    try:
        runpy.run_module('spine_seg_cte.pipeline', run_name='__main__')
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()