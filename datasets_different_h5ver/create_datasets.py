import h5py
import numpy


test_data = numpy.ndarray(shape=(3,10,10), dtype="<u4")
for frame in range(3):
    test_data[frame,:,:] = frame

for v in ["earliest", "v108", "v110", "v112"]:
    f = h5py.File(f"dataset_{v}.h5", 'w', libver=v)
    grp = f.create_group("data")
    dataset = grp.create_dataset("chunked",(3,10,10), maxshape=(None, 10, 10), chunks=(1,10,10), dtype="<u4", compression=None, data=test_data)
    grp["chunked_soft"] = h5py.SoftLink('/data/chunked')
    grp["chunked_hard"] = dataset
    dataset_nochunks = grp.create_dataset("not_chunked",(3,10,10), chunks=None, dtype="<u4", compression=None, data=test_data)
    grp["not_chunked_soft"] = h5py.SoftLink('/data/not_chunked')
    grp["not_chunked_hard"] = dataset_nochunks
    f.close()

    f = h5py.File(f"external_data_{v}.h5", 'w', libver=v)
    grp = f.create_group("data")
    grp['chunked'] = h5py.ExternalLink(f"dataset_{v}.h5", "/data/chunked")
    grp['chunked_soft'] = h5py.ExternalLink(f"dataset_{v}.h5", "/data/chunked_soft")
    grp['chunked_hard'] = h5py.ExternalLink(f"dataset_{v}.h5", "/data/chunked_hard")
    grp['not_chunked'] = h5py.ExternalLink(f"dataset_{v}.h5", "/data/not_chunked")
    grp['not_chunked_soft'] = h5py.ExternalLink(f"dataset_{v}.h5", "/data/not_chunked_soft")
    grp['not_chunked_hard'] = h5py.ExternalLink(f"dataset_{v}.h5", "/data/not_chunked_hard")
    f.close()
