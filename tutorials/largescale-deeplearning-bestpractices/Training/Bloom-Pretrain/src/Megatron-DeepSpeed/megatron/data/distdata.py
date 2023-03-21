import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist

class DistDataError(Exception):
    """Defines an empty exception to throw when some other rank hit a real exception."""
    pass

class DistData(object):
    def __init__(self, backend='gloo'):
        assert backend in ['gloo', 'mpi'], f"torch.distributed backend '{backend}' is not supported, valid options are 'gloo' or 'mpi'"

        dist.init_process_group(backend, init_method="env://")

        # lookup our process rank and the group size
        self.rank = dist.get_rank()
        self.numranks = dist.get_world_size()

    def allassert(self, cond, msg):
        """Check that cond is True on all ranks, assert with msg everywhere if not.

        To prevent deadlocks in cases where an assertion might only fail on one rank,
        this executes an allreduce to ensure that if any rank finds that an assertion
        has been violated, all ranks fail an assertion check.
        The condition must be true on all ranks for this not to assert.
        """
        alltrue = self.alltrue(cond)
        assert alltrue, msg

    def allraise_if(self, err):
        """Raise exception if err is not None on any rank.

        Similarly to allassert, this raises an exception on all ranks if err
        is set to an exception on any rank.  Rank(s) where err is not None
        re-raise err as exception, and ranks where err is None raise DistDataError.
        Thus all ranks raise an exception if any rank has an active exception,
        which helps avoid deadlocks in cases where an exception may be raised
        on a subset of ranks.
        """
        alltrue = self.alltrue(err is None)
        if not alltrue:
            # At least one rank raised an exception.
            # Re-raise the actual exception if this rank threw one.
            if err is not None:
                raise err

            # TODO: is there a better exception to use here?
            # On other ranks, raise an "empty" exception to indicate
            # that we're only failing because someone else did.
            raise DistDataError

    def barrier(self):
        """Globally synchronize all processes"""
        dist.barrier()

    def bcast(self, val, root):
        """Broadcast a scalar value from root to all ranks"""
        vals = [val]
        dist.broadcast_object_list(vals, src=root)
        return vals[0]

    def scatterv_(self, invals: np.array, counts: list, root:int=0):
        """Scatter int64 values from invals according to counts array, return received portion in a new tensor"""

        self.allassert(len(counts) == self.numranks,
            f"Length of counts list {len(counts)} does not match number of ranks {self.numranks}")

        # Define list of tensors to scatter on the root.
        # torch.distributed.scatter requires each tensor to be the same shape,
        # so find the max size across all count values and pad.
        max_size = max(counts)
        scatterlist = None
        if self.rank == root:
            slices = list(torch.split(torch.from_numpy(invals), counts))
            scatterlist = [F.pad(s, (0, max_size - len(s))) for s in slices]

        # Receive a tensor of the max count size from the root,
        # then copy values into output numpy array, which may be smaller.
        recvtensor = torch.zeros(max_size, dtype=torch.int64)
        dist.scatter(recvtensor, scatterlist, src=root)
        return recvtensor[:counts[self.rank]]

    def alltrue(self, val):
        """Returns True if all procs input True, False otherwise"""
        # torch.dist does not support reductions with bool types
        # so we cast to int and cast the result back to bool
        tensor = torch.tensor([int(val)], dtype=torch.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.BAND)
        return bool(tensor[0])

    def sum(self, val):
        """Compute sum of a scalar val, and return total on all ranks."""
        tensor = torch.tensor([val])
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor[0]

    def exscan(self, val: int):
        """Compute prefix sum (exclusive scan) of int64 val, and return offset of each rank."""
        # torch.distributed doesn't have a scan, so fallback to allreduce
        tensor = torch.zeros(self.numranks, dtype=torch.int64)
        tensor[self.rank:] = val
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return int(tensor[self.rank]) - val

    def min(self, val):
        """Return minimum of scalar val to all ranks."""
        tensor = torch.tensor([val])
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return tensor[0]

    def minrank(self, cond):
        """Find first rank whose condition is True, return that rank if any, None otherwise."""
        minrank = self.numranks
        if cond:
            minrank = self.rank
        minrank = self.min(minrank)

        if minrank < self.numranks:
            return minrank
        return None

    def bcast_first(self, val):
        """Broadcast val from first rank where it is not None, return val if any, None otherwise"""
        # Find the first rank with a valid value.
        minrank = self.minrank(val is not None)

        # If there is no rank with a valid value, return None
        if minrank is None:
            return None

        # Otherwise broadcast the value from the first valid rank.
        val = self.bcast(val, root=minrank)
        return val

    def all_sum_(self, vals: np.array):
        """Sums values in numpy array vals element-wise and update vals in place with final result on all ranks"""
        # Builds torch.tensor with from_numpy to use same underlying memory as numpy array.
        tensor = torch.from_numpy(vals)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def open(self, filename, truncate=None):
        """Create, truncate, and open a file shared by all ranks."""

        # Don't truncate existing file until all ranks reach this point
        self.barrier()

        # We'll capture any exception in this variable
        err = None

        # Rank 0 creates and truncates file.
        if self.rank == 0:
            try:
                f = open(filename, 'wb')

                # Some file systems like GPFS deliver faster write speed
                # if the file size is known before data is written to the file.
                if truncate is not None:
                    f.truncate(truncate)

            except Exception as e:
                err = e

        # Verify that rank 0 created the file
        self.allraise_if(err)

        # Wait for rank 0 to open (and truncate) file,
        # then have all ranks open file for writing.
        if self.rank != 0:
            try:
                f = open(filename, 'r+b')
            except Exception as e:
                err = e

        # Verify that all ranks successfully opened the file
        self.allraise_if(err)

        return f

    def remove(self, filename):
        """Remove a shared file."""

        # Don't remove the file until all are ready
        self.barrier()

        # We'll capture any exception in this variable
        err = None

        # Rank 0 removes the file if it exists.
        if self.rank == 0:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                err = e

        # Verify that rank 0 successfully removed the file.
        self.allraise_if(err)

    def rename(self, srcfile, destfile):
        """Rename a shared file."""

        # Don't rename until all are ready
        self.barrier()

        # We'll capture any exception in this variable
        err = None

        # Rank 0 renames the file.
        if self.rank == 0:
            try:
                if os.path.exists(srcfile):
                    os.rename(srcfile, destfile)
            except Exception as e:
                err = e

        # Verify that the rename succeeded
        self.allraise_if(err)
