# common/mappedbus/bus.pyx
import numpy as np
cimport numpy as np

from libc.stdint cimport uint8_t, uint64_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cython cimport boundscheck, wraparound

cdef int CACHE_LINE_SIZE = 64

cdef class MappedBus:
    cdef uint8_t* buffer
    cdef readonly int slot_size, num_slots, aligned_size
    cdef uint64_t write_idx
    cdef uint64_t[:] read_indices
    cdef readonly int max_consumers
    cdef readonly str name

    def __cinit__(self, str name, int slot_size, int num_slots, int max_consumers):
        self.name = name
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.max_consumers = max_consumers
        self.aligned_size = ((1 + slot_size + CACHE_LINE_SIZE - 1) // CACHE_LINE_SIZE) * CACHE_LINE_SIZE
        self.buffer = <uint8_t*> malloc(self.aligned_size * num_slots)
        self.write_idx = 0
        self.read_indices = np.zeros(max_consumers, dtype=np.uint64)
        np.import_array()

    def __dealloc__(self):
        if self.buffer != NULL:
            free(self.buffer)

    @boundscheck(False)
    @wraparound(False)
    cpdef int send(self, np.ndarray[np.float32_t, ndim=1] arr):
        cdef int idx = self.write_idx % self.num_slots
        cdef int base = idx * self.aligned_size
        cdef uint64_t min_read_idx = self.write_idx
        for i in range(self.max_consumers):
            if self.read_indices[i] < min_read_idx:
                min_read_idx = self.read_indices[i]
        if self.write_idx - min_read_idx >= self.num_slots:
            return 0  # full (wrap protection)
        if self.buffer[base] == 1:
            return 0  # slot not consumed yet
        memcpy(self.buffer + base + 1, <void*> arr.data, self.slot_size)
        self.buffer[base] = 1
        self.write_idx += 1
        return 1

    @boundscheck(False)
    @wraparound(False)
    cpdef int recv(self, np.ndarray[np.float32_t, ndim=1] arr_out, int consumer_id):
        cdef int idx = self.read_indices[consumer_id] % self.num_slots
        cdef int base = idx * self.aligned_size
        if self.buffer[base] == 0:
            return 0  # empty
        memcpy(<void*> arr_out.data, self.buffer + base + 1, self.slot_size)
        self.read_indices[consumer_id] += 1

        # Check if all consumers have read
        cdef bint all_read = True
        cdef uint64_t current_read = self.read_indices[consumer_id] - 1
        for i in range(self.max_consumers):
            if self.read_indices[i] <= current_read:
                all_read = False
                break
        if all_read:
            self.buffer[base] = 0
        return 1
