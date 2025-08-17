#!/usr/bin/env python3
"""
Simple example of using the synthetic dataset generator.

This script demonstrates how to use the astroflow.dataset.simulator module
programmatically with custom configurations.
"""

import os
from astroflow.dataset.simulator import SimulationConfig, generate_synthetic_dataset


def main():
    """Simple example of dataset generation."""
    
    # Configure the simulation
    config = SimulationConfig(
        input_dir="/home/lingh/work/astroflow/lingh/FAST_FRB_DATA",
        output_dir="/home/lingh/work/astroflow/simulated_candidates_test",
        num_candidates=10,  # Generate only 10 samples for testing
        bg_samples_per_candidate=2,
        
        # Signal parameters
        dm_range=(320, 570),
        toa_range=(1, 4),
        width_range_ms=(2, 6),
        amp_ratio_range=(0.007, 0.02),
        
        # Frequency parameters
        freq_min_range=(1000, 1300),
        freq_max_range=(1400, 1499),
        
        # Dedispersion parameters
        dm_low=300,
        dm_high=600,
        dm_step=0.5,
        f_start=1000.0,
        f_end=1499.0,
        
        # Image parameters
        image_size=(512, 512),
    )
    
    # Check if input directory exists
    if not os.path.exists(config.input_dir):
        print(f"Error: Input directory {config.input_dir} not found")
        print("Please update the input_dir in this script to point to your data")
        return
    
    print("Starting synthetic dataset generation...")
    print(f"Input directory: {config.input_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Number of candidates: {config.num_candidates}")
    print()
    
    try:
        # Generate the dataset
        stats = generate_synthetic_dataset(config)
        
        # Print results
        print("\nGeneration completed successfully!")
        print("=" * 50)
        print(f"Total generated: {stats['total_generated']}")
        print(f"FRB samples: {stats['frb_samples']}")
        print(f"Weak FRB samples: {stats['weak_frb_samples']}")
        print(f"Background samples: {stats['background_samples']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Output saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
