import h5py
import torch
# import pdb; pdb.set_trace()
with h5py.File("/leonardo_scratch/fast/IscrB_FM-EEG24/ychen004/gen4/train/moorea_2019-02-19_001_td_2684500000_2744500000/event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5") as h5f:
    ev_repr = h5f["data"]
