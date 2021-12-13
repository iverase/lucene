package org.apache.lucene.util;

import org.apache.lucene.search.DocIdSetIterator;

import java.io.IOException;

public class TestBuffers extends LuceneTestCase {
    
    public void testSingleValue() throws IOException {
        int threshold = random().nextInt(50) + 50;
        int maxDoc = 0;
        Buffers buffers = new Buffers(threshold);
        while (true) {
            int batchsize = random().nextInt(25);
            if (buffers.ensureBufferCapacity(batchsize) == false) {
                break;
            }
            for (int i = 0; i < batchsize; i++) {
                buffers.addDoc(maxDoc++);
            }
        }
        FixedBitSet fixedBitSet = new FixedBitSet(maxDoc);
        long counter = buffers.toBitSet(fixedBitSet);
        DocIdSetIterator docIdSetIterator = buffers.toDocIdSet(maxDoc, random().nextBoolean()).iterator();
        assertEquals(counter, fixedBitSet.cardinality());
        assertEquals(maxDoc, counter);
        for (int i = 0; i < maxDoc; i++) {
            int doc = docIdSetIterator.nextDoc();
            assertEquals(i, doc);
            assertTrue(fixedBitSet.get(doc));
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, docIdSetIterator.nextDoc());
    }


    public void testMultiValue() throws IOException {
        int threshold = random().nextInt(50) + 50;
        int maxDoc = 0;
        int totalDocs = 0;
        Buffers buffers = new Buffers(threshold);
        while (true) {
            int batchsize = 1 + random().nextInt(25);
            if (buffers.ensureBufferCapacity(batchsize) == false) {
                break;
            }
            for (int i = 0; i < batchsize; i++) {
                totalDocs++;
                buffers.addDoc(maxDoc);
            }
            maxDoc++;
        }
        FixedBitSet fixedBitSet = new FixedBitSet(maxDoc);
        long counter = buffers.toBitSet(fixedBitSet);
        DocIdSetIterator docIdSetIterator = buffers.toDocIdSet(maxDoc, true).iterator();
        assertEquals(totalDocs, counter);
        for (int i = 0; i < maxDoc; i++) {
            int doc = docIdSetIterator.nextDoc();
            assertEquals(i, doc);
            assertTrue(fixedBitSet.get(doc));
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, docIdSetIterator.nextDoc());
    }

    public void testRandomValues() throws IOException {
        int threshold = random().nextInt(50) + 50;
        int maxDoc = 0;
        int totalDocs = 0;
        Buffers buffers = new Buffers(threshold);
        while (true) {
            int batchsize = 1 + random().nextInt(25);
            if (buffers.ensureBufferCapacity(batchsize) == false) {
                break;
            }
            for (int i = 0; i < batchsize; i++) {
                int doc = random().nextInt(1000000);
                buffers.addDoc(doc);
                maxDoc = Math.max(maxDoc, doc);
                totalDocs++;
            }
        }
        FixedBitSet fixedBitSet = new FixedBitSet(maxDoc + 1);
        long counter = buffers.toBitSet(fixedBitSet);
        DocIdSetIterator docIdSetIterator = buffers.toDocIdSet(maxDoc, true).iterator();
        assertEquals(totalDocs, counter);
        int iteratedDocs = 0;
        int doc;
        while((doc = docIdSetIterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            assertTrue(fixedBitSet.get(doc));
            iteratedDocs++;
        
        }
        assertEquals(iteratedDocs, fixedBitSet.cardinality());
    }
    
}
