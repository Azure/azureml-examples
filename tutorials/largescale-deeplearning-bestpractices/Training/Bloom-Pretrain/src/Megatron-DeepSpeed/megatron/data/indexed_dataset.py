# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# copied from fairseq/fairseq/data/indexed_dataset.py
# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

from functools import lru_cache
import os
import stat
import shutil
import struct
from itertools import accumulate

import numpy as np
import torch

from megatron import print_rank_0


def best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def get_available_dataset_impl():
    return ['lazy', 'cached', 'mmap']


def infer_dataset_impl(path):
    if IndexedDataset.exists(path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return 'cached'
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return 'mmap'
            else:
                return None
    else:
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None


def make_builder(out_file, impl, dtype=None):
    if impl == 'mmap':
        assert dtype is not None
        return MMapIndexedDatasetBuilder(out_file, dtype=dtype)
    else:
        assert dtype is None
        return IndexedDatasetBuilder(out_file)


def make_dataset(path, impl, skip_warmup=False):
    if not IndexedDataset.exists(path):
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None
    if impl == 'infer':
        impl = infer_dataset_impl(path)
    if impl == 'lazy' and IndexedDataset.exists(path):
        return IndexedDataset(path)
    elif impl == 'cached' and IndexedDataset.exists(path):
        return IndexedCachedDataset(path)
    elif impl == 'mmap' and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup)
    print(f"Unknown dataset implementation: {impl}")
    return None


def dataset_exists(path, impl):
    if impl == 'mmap':
        return MMapIndexedDataset.exists(path)
    else:
        return IndexedDataset.exists(path)


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
    8: np.uint16
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.bin'


def create_doc_idx(sizes):
    doc_idx = [0]
    for i, s in enumerate(sizes):
        if s == 0:
            doc_idx.append(i + 1)
    return doc_idx


class IndexedDataset(torch.utils.data.Dataset):
    """Loader for IndexedDataset"""
    _HDR_MAGIC = b'TNTIDX\x00\x00'

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                'Index file doesn\'t match expected format. '
                'Make sure that --dataset-impl is configured properly.'
            )
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1,)
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code]
            self._len, self.s = struct.unpack('<QQ', f.read(16))
            self.doc_count = struct.unpack('<Q', f.read(8))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)
            self.doc_idx = read_longs(f, self.doc_count)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if not self.data_file:
            self.read_data(self.path)
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            return a
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sizes = self.sizes[self.dim_offsets[start]:self.dim_offsets[stop]]
            size = sum(sizes)
            a = np.empty(size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[start] * self.element_size)
            self.data_file.readinto(a)
            offsets = list(accumulate(sizes))
            sents = np.split(a, offsets[:-1])
            return sents

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory


class IndexedCachedDataset(IndexedDataset):

    def __init__(self, path):
        super().__init__(path)
        self.cache = None
        self.cache_index = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        if all(i in self.cache_index for i in indices):
            return
        if not self.data_file:
            self.read_data(self.path)
        indices = sorted(set(indices))
        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]
        self.cache = np.empty(total_size, dtype=self.dtype)
        ptx = 0
        self.cache_index.clear()
        for i in indices:
            self.cache_index[i] = ptx
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            a = self.cache[ptx: ptx + size]
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            ptx += size
        if self.data_file:
            # close and delete data file after prefetch so we can pickle
            self.data_file.close()
            self.data_file = None

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            ptx = self.cache_index[i]
            np.copyto(a, self.cache[ptx: ptx + a.size])
            return a
        elif isinstance(idx, slice):
            # Hack just to make this work, can optimizer later if necessary
            sents = []
            for i in range(*idx.indices(len(self))):
                sents.append(self[i])
            return sents


class IndexedDatasetBuilder(object):
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.uint16: 2,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float: 4,
        np.double: 8
    }

    @staticmethod
    def write_header(fout, dtype, numdata, numsize, numdoc):
        """Writes header for cached indexed dataset to given file handle, return number of bytes written."""
        startpos = fout.tell()

        fout.write(IndexedDataset._HDR_MAGIC)
        fout.write(struct.pack('<Q', 1))
        fout.write(struct.pack('<Q', code(dtype)))
        fout.write(struct.pack('<Q', IndexedDatasetBuilder.element_sizes[dtype]))
        fout.write(struct.pack('<Q', numdata - 1))
        fout.write(struct.pack('<Q', numsize))
        fout.write(struct.pack('<Q', numdoc))

        endpos = fout.tell()
        return endpos - startpos

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, 'wb')
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]
        self.doc_idx = [0]

    def add_item(self, tensor):
        bytes = self.out_file.write(np.array(tensor.numpy(), dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def end_document(self):
        self.doc_idx.append(len(self.sizes))

    def merge_file_(self, another_file):
        index = IndexedDataset(another_file)
        assert index.dtype == self.dtype

        doc_offset = len(self.sizes)

        begin = self.data_offsets[-1]
        for data_offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + data_offset)
        self.sizes.extend(index.sizes)
        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)
        self.doc_idx.extend( (doc_offset + index.doc_idx)[1:] )

        with open(data_file_path(another_file), 'rb') as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, 'wb')
        IndexedDatasetBuilder.write_header(index, self.dtype, len(self.data_offsets), len(self.sizes), len(self.doc_idx))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        write_longs(index, self.doc_idx)
        index.close()


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


def exscan_from_cumsum_(arr):
    # given an array holding the result of an inclusive scan (cumsum),
    # convert to an exclusive scan (shift to the right)
    # [10, 30, 35, 50] --> [0, 10, 30, 35]
    if arr.size > 1:
        arr[1:] = arr[:-1]
    if arr.size > 0:
        arr[0] = 0


def get_pointers_with_total(sizes, elemsize, dtype):
    """Return a numpy array of type np.dtype giving the byte offsets.

    Multiplies values in the sizes array by elemsize (bytes),
    and then computes an exclusive scan to get byte offsets.
    Returns the total number of bytes as second item in a tuple.
    """

    # scale values in sizes array by elemsize to get sizes in bytes
    pointers = np.array(sizes, dtype=dtype)
    pointers *= elemsize
    np.cumsum(pointers, axis=0, out=pointers)

    # get total number of bytes from all sizes (last element)
    bytes_last = pointers[-1] if len(sizes) > 0 else 0

    # convert to byte offsets
    exscan_from_cumsum_(pointers)

    return pointers, bytes_last


class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        @staticmethod
        def write_header(fout, dtype, numsizes, numdocs):
            """Writes header for mmap indexed dataset to given file handle, return number of bytes written."""
            startpos = fout.tell()

            fout.write(MMapIndexedDataset.Index._HDR_MAGIC)
            fout.write(struct.pack('<Q', 1))
            fout.write(struct.pack('<B', code(dtype)))
            fout.write(struct.pack('<Q', numsizes))
            fout.write(struct.pack('<Q', numdocs))

            endpos = fout.tell()
            return endpos - startpos

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, 'wb')
                    return self

                @staticmethod
                def _get_pointers(sizes, npdtype):
                    """Return a numpy array of byte offsets given a list of sizes.

                    Multiplies values in the sizes array by dtype size (bytes),
                    and then computes an exclusive scan to get byte offsets.
                    """

                    # compute element sizes in bytes
                    pointers, _ = get_pointers_with_total(sizes, dtype().itemsize, npdtype)
                    return pointers

                def write(self, sizes, doc_idx):
                    MMapIndexedDataset.Index.write_header(self._file, dtype, len(sizes), len(doc_idx))

                    sizes32 = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes32.tobytes(order='C'))
                    del sizes32

                    pointers = self._get_pointers(sizes, np.int64)
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order='C'))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(
                self._bin_buffer,
                dtype=np.int32,
                count=self._len,
                offset=offset)
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
                                          offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                 count=length, offset=ptr)
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    def size(self, index):
        return self._index.sizes[index]

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )

    @property
    def dtype(self):
        return self._index.dtype


class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, 'wb')
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        total_len = len(index.sizes)+len(self._sizes)
        print(f"    concat {another_file} size={len(index.sizes)} for a total size of {total_len}")

        offset = len(self._sizes)
        self._sizes.extend(index.sizes)
        self._doc_idx.extend( (offset + index.doc_idx)[1:] )

        # Concatenate data
        with open(data_file_path(another_file), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()

        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


# To merge a set of binary files, one can simply concatenate them in order.
# We stat each binary file to determine its size, execute a scan to compute
# the byte offset where the calling rank should write its data, seek to proper
# spot, and copy each file.
def gather_files_dist_bin(outfile, filelist, distctx):
    """Concatenate binary files in filelist into a new file given by outfile"""
    # lookup size of each of our binary files
    filesizes = [os.stat(data_file_path(f))[stat.ST_SIZE] for f in filelist]

    # compute total bytes of the merged file and the offset
    # at which this rank will write data from its files
    numbytes = sum(filesizes)
    count = distctx.sum(numbytes)
    offset = distctx.exscan(numbytes)

    # We first write to a temporary file name.  We rename to the final name
    # if successful or delete the temporary file if not.
    # This way if the final name appears, the user knows it's a valid file.
    finalname = data_file_path(outfile)
    finalnametmp = finalname + ".tmp"

    # First delete the final file if it already exists
    distctx.remove(finalname)

    # Catch I/O errors from any process
    err = None
    try:
        # Create shared output file and pre-truncate to its final size.
        with distctx.open(finalnametmp, truncate=count) as fout:
            # Seek to appropriate starting offset in the merged file.
            fout.seek(offset)

            # Copy in contents of each of our files.
            for f in filelist:
                with open(data_file_path(f), "rb") as fsrc:
                    shutil.copyfileobj(fsrc, fout)

    except Exception as e:
        err = e

    # Check that all ranks wrote successfully.
    # This will raise an exception all on ranks if we detect
    # an exception on any rank.
    distctx.allraise_if(err)

    # Everyone wrote their part successfully.
    # Rename the temporary file to the final file.
    distctx.rename(finalnametmp, finalname)


def write_list_at_offset(fout, file_offset, vals, shift, elem_offset, dtype):
    """Write list of vals to fout starting at an offset given by file_offset, elem_offset, and dtype.

    Copies list of values in vals to a numpy array of type dtype.
    Adds a constant shift value to all elements.
    Writes the numpy array to the file handle at given offset and scaled by size of the datatype.
        offset = file_offset + elem_offset * dtype().itemsize

    Parameters
    ----------
    fout : file handle
        Open file handle to which to write list of vals
    file_offset : int
        Byte offset within the file where the global list starts
    vals : list[int]
        List of values to be written
    shift : int
        Value to add to each element in vals before writing (use 0 for no change)
    elem_offset : int
        Zero-based element index where vals starts within the global list.
        This value is scaled by dtype().itemsize to convert to a corresponding byte offset.
    dtype : np.dtype
        numpy datatype to be used when writing the list to the file
    """

    # Make a copy of the vals list using the requested datatype.
    npvals = np.array(vals, dtype=dtype)

    # Shift values in the list by a constant value.
    npvals += shift

    # Seek to proper offset for this rank and write
    # values into file, stored as given datatype.
    fout.seek(file_offset + elem_offset * dtype().itemsize)
    fout.write(npvals.tobytes(order='C'))


def gather_files_dist_check_dtype(filelist, dtype_rank_consistent, dtype_code, distctx):
    # Verify that no rank has found an inconsistent value in its own set of files.
    # This includes an allreduce to verify that dtype_rank_consistent is True everywhere.
    distctx.allassert(dtype_rank_consistent, "Some rank found inconsistent dtype values")

    # Verify that at least one rank found a dtype value.
    # Because of the bcast, the the value of first_dtype_code is the same on all ranks.
    first_dtype_code = distctx.bcast_first(dtype_code)
    assert first_dtype_code is not None, "Failed to find a dtype value in any index file"

    # Verify that the dtype is consistent on all ranks, if a rank has a dtype value.
    distctx.allassert(dtype_code == first_dtype_code or dtype_code is None, "Different dtype values detected in index files")

    # return the dtype
    return dtypes[first_dtype_code]


def gather_files_dist_idx_cached(outfile, filelist, distctx):
    # Read each index file and append items to our lists
    sizes = []
    data_offsets = [0]
    dim_offsets = [0]
    doc_idx = [0]
    dtype_rank_consistent = True # whether this rank identifies inconsistent dtype values in its files
    dtype_value = None # the current dtype code to compare against, if any
    for f in filelist:
        # read index file for this file
        index = IndexedDataset(f)

        # append its size, data, dim, and doc entries to our lists
        doc_offset = len(sizes)
        sizes.extend(index.sizes)
        data_offsets.extend(index.data_offsets[1:] + data_offsets[-1])
        dim_offsets.extend(index.dim_offsets[1:] + dim_offsets[-1])
        doc_idx.extend(index.doc_idx[1:] + doc_offset)

        # check that the dtype in this index matches the dtype in our other files
        dtype_code = code(index.dtype)
        if dtype_value is None:
            dtype_value = dtype_code
        if dtype_value != dtype_code:
            dtype_rank_consistent = False

    # Check that we have consistent dtypes in all files from all ranks,
    # and return the dtype being used.
    dtype = gather_files_dist_check_dtype(filelist, dtype_rank_consistent, dtype_value, distctx)

    # Capture the last value in the data array before we delete any items.
    # Note this may be zero on any rank that has no items,
    # but zero is the correct value in that case.
    # We use this last value to compute a shift value that
    # is later be added to each element in our data list.
    data_shift = distctx.exscan(data_offsets[-1])

    # Drop the zero entry from the lists that start with
    # a "0" value unless we're rank 0.
    if distctx.rank != 0:
        del data_offsets[0]
        del dim_offsets[0]
        del doc_idx[0]

    # Compute total number of entires in data, size, dim,
    # and doc_idx lists across all ranks.  Also compute the offset
    # of the calling rank for each list considering the number
    # of entries for all ranks before the calling rank.
    numdata = len(data_offsets)
    numsize = len(sizes)
    numdim = len(dim_offsets)
    numdoc = len(doc_idx)

    global_data_count = distctx.sum(numdata)
    global_size_count = distctx.sum(numsize)
    global_dim_count = distctx.sum(numdim)
    global_doc_count = distctx.sum(numdoc)

    global_data_offset = distctx.exscan(numdata)
    global_size_offset = distctx.exscan(numsize)
    global_dim_offset = distctx.exscan(numdim)
    global_doc_offset = distctx.exscan(numdoc)

    # We first write to a temporary file name.  We rename to the final name
    # if successful or delete the temporary file if not.
    # This way if the final name appears, the user knows it's a valid file.
    finalname = index_file_path(outfile)
    finalnametmp = finalname + ".tmp"

    # First delete the final file if it already exists
    distctx.remove(finalname)

    # Catch and I/O errors to later determine whether all ranks wrote successfully.
    err = None
    try:
        # Create shared output file
        with distctx.open(finalnametmp) as fout:
            # Have rank 0 write the file header
            file_offset = 0
            if distctx.rank == 0:
                try:
                    file_offset = fout.tell()
                    file_offset += IndexedDatasetBuilder.write_header(fout, dtype, global_data_count, global_size_count, global_doc_count)
                except Exception as e:
                    err = e
            distctx.allraise_if(err)

            # Broadcast current file position from rank 0.
            file_offset = distctx.bcast(file_offset, root=0)

            # The dimension list records the offset within
            # the sizes list for each sentence.
            # We shift our dimension index values to account for the number of size values
            # that come before the calling rank which is in global_size_offset.
            write_list_at_offset(fout, file_offset, dim_offsets, global_size_offset, global_dim_offset, np.int64)
            file_offset += global_dim_count * np.int64().itemsize

            # The data index records the element offset to the start of each
            # sentence within the binary data file.  Note that this is an
            # element offset, not a byte offset.  Each element is pyhsically stored
            # in the data file as dtype().itemsize bytes.
            # We shift the data index values according to the number of elements that
            # come before the calling rank, which is stored in data_shift.
            write_list_at_offset(fout, file_offset, data_offsets, data_shift, global_data_offset, np.int64)
            file_offset += global_data_count * np.int64().itemsize

            # Each sentence is stored as a tensor.
            # The tensor for each sentence can be multidimensional.
            # The number of tensor dimensions per sentence is variable,
            # and the size of each dimension of a sentence is arbitrary.
            # The size list records a flattened list of the sizes
            # for each dimension of a sentence.
            # No shift value is needed.
            write_list_at_offset(fout, file_offset, sizes, 0, global_size_offset, np.int64)
            file_offset += global_size_count * np.int64().itemsize

            # The document index records the offset within the sizes
            # array for the first sentence of each document.
            # We shift the document index values for number of size values that
            # come before the calling rank which is in global_size_offset.
            write_list_at_offset(fout, file_offset, doc_idx, global_size_offset, global_doc_offset, np.int64)
            file_offset += global_doc_count * np.int64().itemsize

    except Exception as e:
        # if we encounter any exception while writing, store it for later
        err = e

    # Check that all ranks wrote successfully
    distctx.allraise_if(err)

    # Everyone wrote their part successfully.
    # Rename the temporary file to the final file.
    distctx.rename(finalnametmp, finalname)


def gather_files_dist_idx_mmap(outfile, filelist, distctx):
    # Read each index file and append items to the size and doc_idx lists
    sizes = []
    doc_idx = [0]
    dtype_rank_consistent = True # whether rank identifies inconsistent dtype values in its files
    dtype_value = None # the current dtype code to compare against, if any
    for f in filelist:
        # read index file for this file
        index = MMapIndexedDataset.Index(index_file_path(f))

        # append its size and doc entries to our lists
        docs_offset = len(sizes)
        sizes.extend(index.sizes)
        doc_idx.extend(index.doc_idx[1:] + docs_offset)

        # check that the dtype in this index matches the dtype in our other files
        dtype_code = code(index.dtype)
        if dtype_value is None:
            dtype_value = dtype_code
        if dtype_value != dtype_code:
            dtype_rank_consistent = False

    # Check that we have consistent dtypes in all files from all ranks,
    # and return the dtype being used.
    dtype = gather_files_dist_check_dtype(filelist, dtype_rank_consistent, dtype_value, distctx)

    # Drop the zero entry from the lists that start with
    # a "0" value unless we're rank 0
    if distctx.rank != 0:
        del doc_idx[0]

    # Compute total number of size and document index
    # values across all ranks.  Also compute the offset
    # of the calling rank for each value considering
    # the values of sizes/docs for all ranks before the
    # calling rank.
    numsizes = len(sizes)
    numdocs = len(doc_idx)

    global_size_count = distctx.sum(numsizes)
    global_docs_count = distctx.sum(numdocs)

    global_size_offset = distctx.exscan(numsizes)
    global_docs_offset = distctx.exscan(numdocs)

    # Compute local byte offsets for each of our sentences given
    # the token count and byte size of the vocab dtype.
    pointers, pointers_bytes = get_pointers_with_total(sizes, dtype().itemsize, np.int64)

    # Determine total number of bytes for all sentences on ranks
    # before the calling rank.
    pointer_offset = distctx.exscan(pointers_bytes)

    # We first write to a temporary file name.  We rename to the final name
    # if successful or delete the temporary file if not.
    # This way if the final name appears, the user knows it's a valid file.
    finalname = index_file_path(outfile)
    finalnametmp = finalname + ".tmp"

    # First delete the final file if it already exists
    distctx.remove(finalname)

    # Catch and I/O errors to later determine whether all ranks wrote successfully.
    err = None
    try:
        # Create shared output file
        with distctx.open(finalnametmp) as fout:
            # Have rank 0 write the file header
            file_offset = 0
            if distctx.rank == 0:
                try:
                    file_offset = fout.tell()
                    file_offset += MMapIndexedDataset.Index.write_header(fout, dtype, global_size_count, global_docs_count)
                except Exception as e:
                    err = e
            distctx.allraise_if(err)

            # Broadcast current file position from rank 0.
            file_offset = distctx.bcast(file_offset, root=0)

            # The list of size values from each rank are
            # concatenated and stored as int32.
            write_list_at_offset(fout, file_offset, sizes, 0, global_size_offset, np.int32)
            file_offset += global_size_count * np.int32().itemsize

            # The pointer values store the byte offset to each sentence when in memory.
            # A sentence has a variable number of tokens, given by
            # its corresponding entry in the size array.  Each token
            # of a sentence is stored in units of type dtype, which consumes
            # dtype().itemsize bytes (often a standard type that is just
            # large enough to represent all elements of the vocabulary).
            # Since the pointers array is the same length as the sizes array,
            # we use global_size_offset and global_size_count to position
            # within the file for writing the pointer values.
            write_list_at_offset(fout, file_offset, pointers, pointer_offset, global_size_offset, np.int64)
            file_offset += global_size_count * np.int64().itemsize

            # The document index points to the position in the sizes
            # array for the starting sentence of each document.
            # A variable number of sentences can be in each document.
            # We shift the document index for number of sentences that
            # come before the calling rank which is in global_size_offset.
            write_list_at_offset(fout, file_offset, doc_idx, global_size_offset, global_docs_offset, np.int64)
            file_offset += global_docs_count * np.int64().itemsize

    except Exception as e:
        # if we encounter any exception while writing, store it for later
        err = e

    # Check that all ranks wrote successfully
    distctx.allraise_if(err)

    # Everyone wrote their part successfully.
    # Rename the temporary file to the final file.
    distctx.rename(finalnametmp, finalname)


# Verify that all files in filelist are of the same index type.
# Returns the identified type {cached, mmap} as a string.
def gather_files_dist_check_impltype(filelist, distctx):
    # Sanity check for typos in file names.
    # Check that a data file exists for each of our files.
    all_files_exist = all([os.path.exists(data_file_path(f)) for f in filelist])

    # Check that all ranks have all of their files.
    distctx.allassert(all_files_exist, "Some rank is missing its input file")

    # map type string to an integer for easier bcast, use 0 for unknown
    implmap = {"cached": 1, "mmap": 2}

    # check that all files in filelist are of the same type
    sametype = True
    ourtype = None
    for f in filelist:
        # read header of index file to determine its type
        impl = infer_dataset_impl(f)
        implval = implmap[impl] if impl in implmap else 0

        # check that the type matches our other files
        if ourtype is None:
            ourtype = implval
        if ourtype != implval:
            sametype = False

    # Check that all ranks have the same type,
    # and that there is no unknown type.
    # This checks that:
    #   - all of our own files (if any) are of the same type AND
    #   - either we have no files or the type of our files match the broadcast type AND
    #   - the broadcast type is of a known type: {cached, mmap}
    bcasttype = distctx.bcast_first(ourtype)
    matchtype = sametype and (ourtype is None or ourtype == bcasttype) and bcasttype != 0
    distctx.allassert(matchtype, "Cannot merge dataset files of different types")

    # map back to return index string name
    for key in implmap.keys():
        if implmap[key] == bcasttype:
            return key

    # raise exception if key for bcasttype was not found
    raise UnreachableCode


def gather_files_dist(filemain, filelist, distctx):
    """Collectively merge files into a new output file specified in filemain.

    Each rank contributes a distinct list of zero or more files in filelist,
    and each rank directly merges its set of files into filemain.
    It is allowed for the input files in filelist to only be readable from the calling process.
    In particular, the input files specified by the calling process may be in storage
    that only the calling process can access, like /dev/shm or a node-local SSD.
    The output file in filemain should be in a location that is writable by all processes.
    
    NOTE: This uses parallel writes to a shared file to achieve high write bandwidth.
    To do so, this implementation seeks beyond the end of the file to write at different
    offsets from different processes via the seek() method on a python file handle.
    The behavior of seek() is not well documented, but it seems to map to fseek()/lseek(),
    and it works as desired on POSIX-compliant file systems like Lustre and GPFS."""

    # Check that at least one input file is listed
    filecount = distctx.sum(len(filelist))
    assert filecount > 0, "All ranks have no input files to merge"

    # Check that files are all of the same index type
    indexstr = gather_files_dist_check_impltype(filelist, distctx)

    # Concatenate the data files
    gather_files_dist_bin(filemain, filelist, distctx)

    # Combine index files into a single index file
    if indexstr == "cached":
        gather_files_dist_idx_cached(filemain, filelist, distctx)
    elif indexstr == "mmap":
        gather_files_dist_idx_mmap(filemain, filelist, distctx)


def get_start_end(count, rank, numranks):
    """Return (start, end) index values for calling rank to evenly divide count items among numranks.

    Example usage:
        start, end = get_start_end(len(itemlist), distctx.rank, distctx.numranks)
        sublist = itemlist[start:end]

    Parameters
    ----------
    count : int
        Total number of items to be divided
    rank : int
        Rank of the calling process, within range of [0, numranks)
    numranks : int
        Number of ranks by which to divide count items

    Returns
    ----------
    (start, end) : tuple(int)
        Start and end index values that define the [start, end) range for rank
    """
    num, remainder = divmod(count, numranks)
    if rank < remainder:
        start = (num + 1) * rank
        end = start + num + 1
    else:
        start = (num + 1) * remainder + num * (rank - remainder)
        end = start + num
    return start, end


def merge_files_dist(filemain, filelist, distctx):
    """Merge list of indexed datasets into a single indexed dataset named in filemain.

    Given a list of indexed datasets in filelist, and the set of processes defined
    by the distributed environment in distctx, collectively merge files into
    a new, single output indexed dataset named in filemain.  This overwrites filemain
    if it already exists.  It does not delete the input datasets in filelist.  The input
    parameters filemain and filelist must be identical on all calling processes,
    and all processes in distctx must call this method collectively.
    It requires that all ranks be able to read any file in filelist, and all
    ranks must be able to write to the single output file named in filemain."""

    # TODO: if file sizes vary significantly, it might be better to consider
    # file size when splitting the list to different ranks.

    # evenly divide list of files among ranks
    start, end = get_start_end(len(filelist), distctx.rank, distctx.numranks)
    sublist = filelist[start:end]

    # delegate merge to gather implementation
    return gather_files_dist(filemain, sublist, distctx)
