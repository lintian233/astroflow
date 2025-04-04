import your
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Plot a spectrum from a filterbank file.")
    parser.add_argument("filename", type=str, help="Path to the filterbank file.")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--end", type=float, default=1.0, help="End time in seconds.")
    parser.add_argument("--freq_start", type=float, default=0.0, help="Start frequency in MHz.")
    parser.add_argument("--freq_end", type=float, default=1.0, help="End frequency in MHz.")
    return parser.parse_args()


def plot_spectrum(filename, start_time, end_time, freq_start, freq_end,dpi=100):

    your_reader = your.Your(filename)
    header = your_reader.your_header
    print("header", header)
    tstart = int(start_time / header.tsamp)
    tend = int(end_time / header.tsamp)
    ndata = tend - tstart
    print("tstart", tstart)
    print("tend", tend)
    data = your_reader.get_data(tstart, ndata)
    foff = header.foff
    fch1 = header.fch1
    if foff < 0:
        fch1 = header.fch1 + (header.nchans - 1) * foff
        foff = -foff
        data = data[:,::-1]

    chan_start = int((freq_start - fch1) / foff)
    chan_end = int((freq_end - fch1) / foff)
    data = data[:, chan_start:chan_end]
    ndata, nchans = data.shape
    print("ndata", ndata)
    print("nchans", nchans)
    plt.figure(figsize=(10, 6), dpi=dpi)
    freqs = np.arange(chan_start, chan_end) * foff + fch1
    times = np.arange(ndata) * header.tsamp + start_time
    plt.imshow(data.T, 
               aspect='auto', 
               origin='lower', 
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               cmap='viridis')
    

    
    savename = filename.split("/")[-1].split(".")[0]

    plt.savefig(f"{savename}_{start_time}_{end_time}_{freq_start}_{freq_end}.png", dpi=dpi)
    plt.close()
    print(f"Saved plot to {savename}_{start_time}_{end_time}_{freq_start}_{freq_end}.png")


def main():
    args = parse_args()
    plot_spectrum(args.filename, args.start, args.end, args.freq_start, args.freq_end)

if __name__ == "__main__":
    main()
