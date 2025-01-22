import segyio
import numpy as np
import matplotlib.pyplot as plt
filename='/home/data/TQH_WAIXIE_DATA_WJG_PSDM_DEPTH_GAIN.segy'
# filename='/home/data/TQH_WAIXIE_DATA_WJG_PSDM_MODEL.SEGY'
with segyio.open(filename) as segyfile:
    # for trace in segyfile.trace:
    #     print(trace.shape)
 
    # Memory map file for faster reading (especially if file is big...)
    segyfile.mmap()
 
    # Print binary header info
    print(segyfile.bin)
    print(segyfile.bin[segyio.BinField.Traces])
 
    # Read headerword inline for trace 10
    print(segyfile.header[10][segyio.TraceField.INLINE_3D])
    # 
    # Print inline and crossline axis
    print(segyfile.xlines)
    print(segyfile.ilines)
    print(segyfile.depth_slice)
    # Read data along first xline
    data = segyfile.xline[segyfile.xlines[50]]
    print(segyfile.depth_slice.shape)
    print(data.shape)

    plt.figure()
    plt.imshow(data.T)
    plt.colorbar()
    plt.savefig("/home/pengyaoguang/program_learn/2D/1.png")
    plt.close()
    # Read data along last iline
    data = segyfile.iline[segyfile.ilines[-1]]
 
    # Read data along 100th time slice
    data = segyfile.depth_slice[100]
 
    # Read data cube
    data = segyio.tools.cube(filename)
    
    # Print offsets
    print(segyfile.offset)
 
    # Read data along first iline and offset 100:  data [nxl x nt]
    data = segyfile.iline[0, 100]

 
    # Read data along first iline and all offsets gath:  data [noff x nxl x nt]
    data = np.asarray([np.copy(x) for x in segyfile.iline[0:1, :]])
 
    # Read data along first 5 ilines and all offsets gath:  data [noff nil x nxl x nt]
    data = np.asarray([np.copy(x) for x in segyfile.iline[0:5, :]])
 
    # Read data along first xline and all offsets gath:  data [noff x nil x nt]
    data = np.asarray([np.copy(x) for x in segyfile.xline[0:1, :]])
