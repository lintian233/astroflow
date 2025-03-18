#type: ignore
import _astroflow_core as af
import numpy as np
import matplotlib.pyplot as plt

fil = af.Filterbank("/home/lingh/work/astroflow/tests/FRB180417.fil")
print(fil.data[:,0,:].shape)
raw_data = fil.data

vmin, vmax = np.percentile(raw_data, [1, 99])

x_shape = raw_data.shape[0]
y_shape = raw_data.shape[2]
plt.figure(figsize=(20, 8), dpi=300)
plt.rcParams['image.origin'] = 'lower'  

time_axis = np.arange(fil.data.shape[0]) * fil.tsamp
freq_axis = fil.fch1 + np.arange(fil.data.shape[2]) * fil.foff

plt.pcolormesh(time_axis, freq_axis, raw_data[:,0,:].T,
               shading='nearest',  
               cmap='viridis',
               vmin=vmin, vmax=vmax,
               rasterized=True)  

plt.xlabel(f"Time (s)\nTSAMP={fil.tsamp:.6e}s")
plt.ylabel(f"Frequency (MHz)\nFCH1={fil.fch1:.3f} MHz, FOFF={fil.foff:.3f} MHz")
plt.xlim(time_axis[1600], time_axis[2300])
cbar = plt.colorbar()
cbar.set_label('Flux Density (arb. units)')

# 保存为16-bit PNG保持动态范围
plt.savefig("test_fil.png", 
           dpi=300, 
           bbox_inches='tight',
           facecolor='white',
           format='png',
           pil_kwargs={'compress_level': 0})  # 禁用压缩
