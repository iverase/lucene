package org.apache.lucene.util;

import org.apache.lucene.search.DocIdSet;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.packed.PackedInts;

import java.util.ArrayList;
import java.util.List;

/**
 * Sparse structure to collect docIds. The structure will grow until the provided threshold.
 */
class Buffers {

    private static class Buffer {
        int[] array;
        int length;

        Buffer(int length) {
            this.array = new int[length];
            this.length = 0;
        }

        Buffer(int[] array, int length) {
            this.array = array;
            this.length = length;
        }
    }
    
    private final List<Buffer> buffers;
    private final int threshold;
    private int totalAllocated; // accumulated size of the allocated buffers
    private Buffer current;

    Buffers(int threshold) {
        this.buffers = new ArrayList<>();
        this.threshold = threshold;
    }

    public void addDoc(int doc) {
        current.array[current.length++] = doc;
    }

    /**
     * return true if the buffer can allocate numDocs, otherwise false
     */
    public boolean ensureBufferCapacity(int numDocs) {
        if ((long) totalAllocated + numDocs > threshold) {
            return false;
        }
        if (buffers.isEmpty()) {
            addBuffer(additionalCapacity(numDocs));
            return true;
        }
        if (current.array.length - current.length >= numDocs) {
            // current buffer is large enough
            return true;
        }
        if (current.length < current.array.length - (current.array.length >>> 3)) {
            // current buffer is less than 7/8 full, resize rather than waste space
            growBuffer(current, additionalCapacity(numDocs));
        } else {
            addBuffer(additionalCapacity(numDocs));
        }
        return true;
    }

    public long toBitSet(BitSet bitSet) {
        long counter = 0;
        for (Buffer buffer : buffers) {
            int[] array = buffer.array;
            int length = buffer.length;
            counter += length;
            for (int i = 0; i < length; ++i) {
                bitSet.set(array[i]);
            }
        }
        return counter;
    }

    public DocIdSet toDocIdSet(int maxDoc, boolean multivalued) {
        Buffer concatenated = concat(buffers);
        LSBRadixSorter sorter = new LSBRadixSorter();
        sorter.sort(PackedInts.bitsRequired(maxDoc - 1), concatenated.array, concatenated.length);
        final int l;
        if (multivalued) {
            l = dedup(concatenated.array, concatenated.length);
        } else {
            assert noDups(concatenated.array, concatenated.length);
            l = concatenated.length;
        }
        assert l <= concatenated.length;
        concatenated.array[l] = DocIdSetIterator.NO_MORE_DOCS;
        return new IntArrayDocIdSet(concatenated.array, l);
    }

    /**
     * Concatenate the buffers in any order, leaving at least one empty slot in the end NOTE: this
     * method might reuse one of the arrays
     */
    private static Buffer concat(List<Buffer> buffers) {
        int totalLength = 0;
        Buffer largestBuffer = null;
        for (Buffer buffer : buffers) {
            totalLength += buffer.length;
            if (largestBuffer == null || buffer.array.length > largestBuffer.array.length) {
                largestBuffer = buffer;
            }
        }
        if (largestBuffer == null) {
            return new Buffer(1);
        }
        int[] docs = largestBuffer.array;
        if (docs.length < totalLength + 1) {
            docs = ArrayUtil.growExact(docs, totalLength + 1);
        }
        totalLength = largestBuffer.length;
        for (Buffer buffer : buffers) {
            if (buffer != largestBuffer) {
                System.arraycopy(buffer.array, 0, docs, totalLength, buffer.length);
                totalLength += buffer.length;
            }
        }
        return new Buffer(docs, totalLength);
    }

    private static boolean noDups(int[] a, int len) {
        for (int i = 1; i < len; ++i) {
            assert a[i - 1] < a[i];
        }
        return true;
    }

    private Buffer addBuffer(int len) {
        Buffer buffer = new Buffer(len);
        buffers.add(buffer);
        this.current = buffer;
        totalAllocated += buffer.length;
        return buffer;
    }

    private void growBuffer(Buffer buffer, int additionalCapacity) {
        buffer.array = ArrayUtil.growExact(buffer.array, buffer.length + additionalCapacity);
        totalAllocated += additionalCapacity;
    }

    private int additionalCapacity(int numDocs) {
        // exponential growth: the new array has a size equal to the sum of what
        // has been allocated so far
        int c = totalAllocated;
        // but is also >= numDocs + 1 so that we can store the next batch of docs
        // (plus an empty slot so that we are more likely to reuse the array in build())
        c = Math.max(numDocs + 1, c);
        // avoid cold starts
        c = Math.max(32, c);
        // do not go beyond the threshold
        c = Math.min(threshold - totalAllocated, c);
        return c;
    }

    private static int dedup(int[] arr, int length) {
        if (length == 0) {
            return 0;
        }
        int l = 1;
        int previous = arr[0];
        for (int i = 1; i < length; ++i) {
            final int value = arr[i];
            assert value >= previous;
            if (value != previous) {
                arr[l++] = value;
                previous = value;
            }
        }
        return l;
    }
}
