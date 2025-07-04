import os

def reverse_mask_channels(input_file='mask.txt', output_file='mask.txt', total_channels=2047):
    """
    Reads space-separated channel numbers from an input file,
    subtracts each from a total number of channels, and writes
    the result to an output file.
    """
    try:
        with open(input_file, 'r') as f:
            original_channels_str = f.read()
        
        original_channels = [int(ch) for ch in original_channels_str.split()]
        
        reversed_channels = [total_channels - ch for ch in original_channels]

        output_content = ' '.join(map(str, reversed_channels))
        
        with open(output_file, 'w') as f:
            f.write(output_content)
            
        print(f"成功！已将 {input_file} 中 {len(original_channels)} 个通道号反转，并存入 {output_file}")
        print(f"文件保存在: {os.path.abspath(output_file)}")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file}' 未找到。")
    except ValueError:
        print(f"错误: '{input_file}' 中包含无效的非整数值。")
    except Exception as e:
        print(f"发生未知错误: {e}")

def argparse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Reverse Mask Channels")
    parser.add_argument('--input', type=str, default='mask.txt', help='Input file with channel numbers')
    parser.add_argument('--output', type=str, default='mask.txt', help='Output file for reversed channel numbers')
    parser.add_argument('-c', '--total_channels', type=int, default=2047, help='Total number of channels (default: 2047)')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = argparse_args()
    reverse_mask_channels(input_file=args.input, output_file=args.output, total_channels=args.total_channels)
