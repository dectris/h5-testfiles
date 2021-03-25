import hdf5plugin
import h5py
import numpy

def test_data(shape):
    output = numpy.ndarray(shape=shape, dtype="<u4")
    for frame in range(shape[0]):
        output[frame,:,:] = frame
    return output

for v in ["earliest", "v108", "v110", "v112"]:
    filename = f"dataset_{v}.h5"
    external_filename = f"dataset_{v}_external.h5"

    f = h5py.File(filename, 'w', libver=v)
    grp = f.create_group("data")

    # we need to create both uncompressed and compressed datasets as the internal chunk messages vary
    # will create data layout: fixed array (for superblock 3) [completely fixed maxshape]
    grp.create_dataset("chunked_fixed_array",(20,2,2), maxshape=(20, 2, 2), chunks=(1,2,2), dtype="<u4", compression=None, data=test_data((20, 2, 2)))
    grp.create_dataset("chunked_fixed_array_bslz4",(20,2,2), **hdf5plugin.Bitshuffle(), maxshape=(20, 2, 2), chunks=(1,2,2), dtype="<u4", data=test_data((20, 2, 2)))

    # will create data layout: fixed array (paged) (for superblock 3) [completely fixed maxshape, high number of chunks]
    grp.create_dataset("chunked_fixed_array_paged",(5000,2,2), maxshape=(5000, 2, 2), chunks=(1,2,2), dtype="<u4", compression=None, data=test_data((5000, 2, 2)))
    grp.create_dataset("chunked_fixed_array_paged_bslz4",(5000,2,2), **hdf5plugin.Bitshuffle(), maxshape=(5000, 2, 2), chunks=(1,2,2), dtype="<u4", data=test_data((5000, 2, 2)))

    # will create data layout: b-tree v2 (for superblock 3) [unlimited maxshape]
    grp.create_dataset("chunked_unlimited_maxshape",(3,10,10), maxshape=(None, None, None), chunks=(1,10,10), dtype="<u4", compression=None, data=test_data((3,10,10)))
    grp.create_dataset("chunked_unlimited_maxshape_bslz4",(3,10,10), **hdf5plugin.Bitshuffle(), maxshape=(None, None, None), chunks=(1,10,10), dtype="<u4", data=test_data((3,10,10)))

    # will create data layout: extensible array (for superblock 3) [single dimension unlimited maxshape]
    dataset = grp.create_dataset("chunked",(3,10,10), maxshape=(None, 10, 10), chunks=(1,10,10), dtype="<u4", compression=None, data=test_data((3,10,10)))
    grp.create_dataset("chunked_bslz4",(3,10,10), **hdf5plugin.Bitshuffle(), maxshape=(None, 10, 10), chunks=(1,10,10), dtype="<u4", data=test_data((3,10,10)))

    # create all different kinds of links
    grp["chunked_external"] = h5py.ExternalLink(external_filename, "/data/chunked")
    grp["chunked_soft"] = h5py.SoftLink('/data/chunked')
    grp["chunked_hard"] = dataset
    sub = f.create_group("sub")
    sub["chunked_external"] = h5py.ExternalLink(external_filename, "/data/chunked")
    sub["chunked_soft"] = h5py.SoftLink('/data/chunked')
    sub["chunked_hard"] = dataset
    grp["sub_hard"] = sub
    grp["sub_soft"] = h5py.SoftLink('/sub')
    dataset_nochunks = grp.create_dataset("not_chunked",(3,10,10), chunks=None, dtype="<u4", compression=None, data=test_data((3,10,10)))
    grp["not_chunked_external"] = h5py.ExternalLink(external_filename, "/data/not_chunked")
    grp["not_chunked_soft"] = h5py.SoftLink('/data/not_chunked')
    grp["not_chunked_hard"] = dataset_nochunks
    f.close()

    # create file with the external data
    f = h5py.File(external_filename, 'w', libver=v)
    grp = f.create_group("data")
    grp.create_dataset("chunked",(3,10,10), maxshape=(None, 10, 10), chunks=(1,10,10), dtype="<u4", compression=None, data=test_data((3,10,10)))
    grp.create_dataset("not_chunked",(3,10,10), chunks=None, dtype="<u4", compression=None, data=test_data((3,10,10)))
    f.close()
