# Spine Segmentation CTE

A comprehensive pipeline for vertebral CT segmentation and analysis, featuring cortical, trabecular, and endplate segmentation with MR-assisted anatomical labeling.

## Features

- **Automated Vertebral Segmentation**: Uses TotalSegmentator for initial vertebral body detection
- **Multi-Region Analysis**: Segments and analyzes:
  - Cortical bone (with thickness measurements)
  - Trabecular bone
  - Superior and inferior endplates
- **Anatomical Labeling**: Automatic identification of vertebral levels (C1-C7, T1-T12, L1-L5, Sacrum)
- **Robust Processing**: Handles various CT scan orientations and qualities
- **Parallel Processing**: Efficient multi-core processing for batch analysis
- **Comprehensive Metrics**: Computes HU values, volumes, and morphometric measurements

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Install from PyPI

```bash
pip install spine-seg-cte
```

### Install from Source

```bash
git clone https://github.com/yourusername/spine-seg-cte.git
cd spine-seg-cte
pip install -e .
```

## Quick Start

### Command Line Interface

```bash
# Run interactive mode
spine-seg-cte

# Or specify directories directly
spine-seg-cte --input /path/to/ct/scans --output /path/to/output
```

### Python API

```python
from spine_seg_cte import CONFIG
from spine_seg_cte.pipeline import (
    discover_scans,
    setup_output_structure,
    process_all_patients_parallel
)

# Configure pipeline
CONFIG.update({
    'use_gpu': True,
    'analysis_workers': 16
})

# Setup
input_dir = "/path/to/ct/scans"
output_dir = "/path/to/output"
output_dirs = setup_output_structure(output_dir)

# Discover and process scans
scans = discover_scans(input_dir)
results = process_all_patients_parallel(scans, output_dirs, num_workers=16)

print(f"Processed {len([r for r in results if r['success']])} vertebrae")
```

## Configuration

The pipeline can be configured via command-line arguments or configuration file:

```python
CONFIG = {
    'skip_totalsegmentator': False,  # Skip if segmentations exist
    'use_gpu': True,                  # Use GPU acceleration
    'totalseg_workers': 4,            # Parallel TotalSegmentator jobs
    'analysis_workers': 16,           # Parallel analysis workers
    'num_layers': 100,                # Layers for thickness computation
    'min_vertebra_volume': 1000,      # Minimum voxels for valid vertebra
    'save_segmentations': True,       # Save segmentation masks
    'use_plane_pruning': True,        # Use plane-based pruning
}
```

## Input Requirements

- **File Format**: NIfTI (`.nii` or `.nii.gz`)
- **Naming Convention**: Files should be named `input.*.nii.gz` (e.g., `input.patient001.nii.gz`)
- **CT Scans**: Bone window or standard CT reconstructions
- **Directory Structure**: Place all CT scans in a single input directory

## Output

The pipeline generates:

### CSV Results (`all_results_v10.csv`)
Contains per-vertebra metrics including:
- Anatomical labels (C1-C7, T1-T12, L1-L5, Sacrum)
- Cortical thickness and HU values
- Trabecular HU values
- Endplate thickness and HU values
- Volumes and voxel counts

### Segmentation Masks
- Individual vertebra segmentations with labeled regions:
  - Label 0: Background
  - Label 1: Cortical bone
  - Label 2: Trabecular bone
  - Label 3: Superior endplate
  - Label 4: Inferior endplate

### Summary Statistics (`summary_v10.json`)
- Overall statistics across all processed vertebrae
- Quality metrics and processing information

## Algorithm Overview

1. **Vertebral Detection**: Uses TotalSegmentator to identify vertebral bodies
2. **Canonical Orientation**: Reorients images to RAS+ space for consistent processing
3. **Cortical Segmentation**: 
   - Computes signed distance field normals
   - Performs constrained region growing
   - Estimates cortical thickness
4. **Trabecular Segmentation**: Identifies internal trabecular structure
5. **Endplate Segmentation**: Detects superior and inferior endplates using PCA-based alignment
6. **Anatomical Labeling**: Cross-references with TotalSegmentator labels for anatomical identification

## Requirements

- numpy >= 1.20.0
- pandas >= 1.3.0
- nibabel >= 3.2.0
- scipy >= 1.7.0
- scikit-learn >= 0.24.0
- tqdm >= 4.60.0
- TotalSegmentator >= 2.0.0

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{spine_seg_cte_2024,
  title={Spine Segmentation CTE: Comprehensive Vertebral CT Segmentation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/spine-seg-cte}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or contributions, please open an issue on GitHub or contact the maintainers.

## Acknowledgments

- TotalSegmentator for vertebral body segmentation
- The medical imaging community for open-source tools
