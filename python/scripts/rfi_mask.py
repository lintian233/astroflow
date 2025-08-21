#!/usr/bin/env python3
"""
Batch processing script for RFI masking of FITS/FIL files
Processes all FITS and FIL files in a specified directory using your_rfimask.py
Supports concurrent processing for improved performance
"""

import os
import sys
import glob
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing

from tqdm import tqdm


def find_files(data_dir, extensions=None):
    """Find all files with specified extensions in the data directory"""
    if extensions is None:
        extensions = ['*.fits', '*.fil']
    
    files = []
    for ext in extensions:
        pattern = os.path.join(data_dir, ext)
        files.extend(glob.glob(pattern))
    
    return sorted(files)


def run_rfi_mask_single(input_file, output_dir, sg_sigma=1, sg_frequency=5, sk_sigma=0, nspectra=8192, verbose=False):
    """Run your_rfimask.py on a single file"""
    
    # Construct the command
    cmd = [
        'your_rfimask.py',
        '-f', input_file,
        '-sg_sigma', str(sg_sigma),
        '-sg_frequency', str(sg_frequency),
        '-o', output_dir
    ]
    
    # Add optional parameters
    if sk_sigma > 0:
        cmd.extend(['-sk_sigma', str(sk_sigma)])
    
    if nspectra != 8192:
        cmd.extend(['-n', str(nspectra)])
    
    if verbose:
        cmd.append('-v')
    
    filename = os.path.basename(input_file)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {
            'file': filename,
            'success': True,
            'message': f"✓ Successfully processed {filename}",
            'stdout': result.stdout if verbose else None
        }
    except subprocess.CalledProcessError as e:
        return {
            'file': filename,
            'success': False,
            'message': f"✗ Error processing {filename}: {e}",
            'stderr': e.stderr
        }
    except FileNotFoundError:
        return {
            'file': filename,
            'success': False,
            'message': f"✗ Error: your_rfimask.py not found in PATH for {filename}",
            'stderr': "Please ensure your_rfimask.py is installed and accessible"
        }


def run_rfi_mask(input_file, output_dir, sg_sigma=1, sg_frequency=5, sk_sigma=0, nspectra=8192, verbose=False):
    """Legacy function for single-threaded processing"""
    result = run_rfi_mask_single(input_file, output_dir, sg_sigma, sg_frequency, sk_sigma, nspectra, verbose)
    
    print(f"Processing: {result['file']}")
    print(result['message'])
    if result['success'] and verbose and result['stdout']:
        print(f"Output: {result['stdout']}")
    elif not result['success'] and 'stderr' in result:
        print(f"Error details: {result['stderr']}")
    
    return result['success']


def process_files_concurrent(files, output_dir, sg_sigma, sg_frequency, sk_sigma, nspectra, verbose, max_workers):
    """Process files concurrently using multiprocessing"""
    
    # Create a partial function with fixed parameters
    process_func = partial(
        run_rfi_mask_single,
        output_dir=output_dir,
        sg_sigma=sg_sigma,
        sg_frequency=sg_frequency,
        sk_sigma=sk_sigma,
        nspectra=nspectra,
        verbose=verbose
    )
    
    successful = 0
    failed = 0
    results = []
    
    print(f"Processing {len(files)} files with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_func, file_path): file_path for file_path in files}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)
                
                if result['success']:
                    successful += 1
                else:
                    failed += 1
                
                pbar.set_postfix({
                    'success': successful,
                    'failed': failed,
                    'current': result['file'][:20] + '...' if len(result['file']) > 20 else result['file']
                })
                pbar.update(1)
    
    # Print detailed results
    print("\n" + "="*80)
    print("PROCESSING RESULTS:")
    print("="*80)
    
    for result in sorted(results, key=lambda x: x['file']):
        print(result['message'])
        if not result['success'] and 'stderr' in result and result['stderr']:
            print(f"  Error details: {result['stderr']}")
        elif result['success'] and verbose and result['stdout']:
            print(f"  Output: {result['stdout']}")
    
    return successful, failed


def astrorfimask():
    parser = argparse.ArgumentParser(
        description="Batch process FITS/FIL files with RFI masking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-d', '--data_dir',
        default='/data/QL/lingh/FAST_FRB_DATA/',
        help='Directory containing FITS/FIL files to process'
    )
    
    parser.add_argument(
        '-o', '--output_dir',
        default='/home/lingh/work/astroflow/ql/FAST_PREFIX_RFI_MASK',
        help='Output directory for RFI masks'
    )
    
    parser.add_argument(
        '-sg_sigma', '--savgol_sigma',
        type=float,
        default=2,
        help='Sigma for Savgol filter RFI mitigation'
    )
    
    parser.add_argument(
        '-sg_frequency', '--savgol_frequency',
        type=float,
        default=5,
        help='Filter window for savgol filter (in MHz)'
    )
    
    parser.add_argument(
        '-sk_sigma', '--spectral_kurtosis_sigma',
        type=float,
        default=2,
        help='Sigma for spectral kurtosis based RFI mitigation'
    )
    
    parser.add_argument(
        '-n', '--nspectra',
        type=int,
        default=8192,
        help='Number of spectra to read and apply filters to'
    )
    
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['*.fits', '*.fil'],
        help='File extensions to process'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be processed without actually running'
    )
    
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=8,
        help='Number of concurrent jobs (default: 8)'
    )
    
    parser.add_argument(
        '--no_concurrent',
        action='store_true',
        help='Disable concurrent processing (use single-threaded mode)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all files to process
    files = find_files(args.data_dir, args.extensions)
    
    if not files:
        print(f"No files found in {args.data_dir} with extensions {args.extensions}")
        sys.exit(1)
    
    # Validate number of jobs
    max_cpu_count = multiprocessing.cpu_count()
    if args.jobs > max_cpu_count:
        print(f"Warning: Requested {args.jobs} jobs, but only {max_cpu_count} CPUs available")
        print(f"Using {max_cpu_count} jobs instead")
        args.jobs = max_cpu_count
    
    print(f"Found {len(files)} files to process:")
    for f in files[:10]:  # Show first 10 files
        print(f"  - {os.path.basename(f)}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
    
    if args.dry_run:
        print("\nDry run mode - no files will be processed")
        print(f"Would use parameters:")
        print(f"  sg_sigma: {args.savgol_sigma}")
        print(f"  sg_frequency: {args.savgol_frequency}")
        print(f"  sk_sigma: {args.spectral_kurtosis_sigma}")
        print(f"  nspectra: {args.nspectra}")
        print(f"  output_dir: {args.output_dir}")
        print(f"  concurrent jobs: {'disabled' if args.no_concurrent else args.jobs}")
        return
    
    print(f"\nStarting batch processing...")
    print(f"Output directory: {args.output_dir}")
    print(f"Concurrent processing: {'disabled' if args.no_concurrent else f'enabled ({args.jobs} workers)'}")
    
    # Process files
    if args.no_concurrent or len(files) == 1:
        # Single-threaded processing
        successful = 0
        failed = 0
        
        for file_path in tqdm(files, desc="Processing files"):
            success = run_rfi_mask(
                input_file=file_path,
                output_dir=args.output_dir,
                sg_sigma=args.savgol_sigma,
                sg_frequency=args.savgol_frequency,
                sk_sigma=args.spectral_kurtosis_sigma,
                nspectra=args.nspectra,
                verbose=args.verbose
            )
            
            if success:
                successful += 1
            else:
                failed += 1
            
            print()  # Add blank line between files
    else:
        # Concurrent processing
        successful, failed = process_files_concurrent(
            files=files,
            output_dir=args.output_dir,
            sg_sigma=args.savgol_sigma,
            sg_frequency=args.savgol_frequency,
            sk_sigma=args.spectral_kurtosis_sigma,
            nspectra=args.nspectra,
            verbose=args.verbose,
            max_workers=args.jobs
        )
    
    # Summary
    print("\n" + "="*80)
    print(f"BATCH PROCESSING COMPLETE!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Total files: {len(files)}")
    print(f"Success rate: {(successful/len(files)*100):.1f}%")
    print("="*80)
    

    print(f"BATCH PROCESSING COMPLETE!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Total files: {len(files)}")
    print(f"Success rate: {(successful/len(files)*100):.1f}%")
    
    if failed > 0:
        sys.exit(1)