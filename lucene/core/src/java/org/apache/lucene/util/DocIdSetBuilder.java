/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.util;

import java.io.IOException;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.Terms;
import org.apache.lucene.search.DocIdSet;
import org.apache.lucene.search.DocIdSetIterator;

/**
 * A builder of {@link DocIdSet}s. At first it uses a sparse structure to gather documents, and then
 * upgrades to a non-sparse bit set once enough hits match.
 *
 * <p>To add documents, you first need to call {@link #grow} in order to reserve space, and then
 * call {@link BulkAdder#add(int)} on the returned {@link BulkAdder}.
 *
 * @lucene.internal
 */
public final class DocIdSetBuilder {

  /**
   * Utility class to efficiently add many docs in one go.
   *
   * @see DocIdSetBuilder#grow
   */
  public abstract static class BulkAdder {
    public abstract void add(int doc);

    public void add(DocIdSetIterator iterator) throws IOException {
      int docID;
      while ((docID = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
        add(docID);
      }
    }

    protected abstract boolean ensureCapacity(int numDocs);

    protected abstract DocIdSet toDocIdSet();

    protected abstract int toBitSet(BitSet bitSet);
  }

  private static class FixedBitSetAdder extends BulkAdder {
    final FixedBitSet bitSet;
    final double numValuesPerDoc;
    long counter;

    FixedBitSetAdder(FixedBitSet bitSet, double numValuesPerDoc) {
      this.bitSet = bitSet;
      this.numValuesPerDoc = numValuesPerDoc;
    }

    @Override
    public void add(int doc) {
      bitSet.set(doc);
    }

    @Override
    public void add(DocIdSetIterator iterator) throws IOException {
      bitSet.or(iterator);
    }
    
    protected boolean ensureCapacity(int numDocs) {
      counter += numDocs;
      return true;
    }

    @Override
    protected DocIdSet toDocIdSet() {
      assert counter >= 0;
      final long cost = Math.round(counter / numValuesPerDoc);
      return new BitDocIdSet(bitSet, cost);
    }

    @Override
    protected int toBitSet(BitSet bitSet) {
      throw new UnsupportedOperationException();
    }
  }

  private static class BufferAdder extends BulkAdder {
    final Buffers buffers;
    final int maxDoc;
    final boolean multivalued;
   

    BufferAdder(Buffers buffers, int maxDoc, boolean multivalued) {
      this.buffers = buffers;
      this.maxDoc = maxDoc;
      this.multivalued = multivalued;
    }

    @Override
    public void add(int doc) {
      buffers.addDoc(doc);
    }

    protected boolean ensureCapacity(int numDocs) {
      return buffers.ensureBufferCapacity(numDocs);
    }

    @Override
    protected DocIdSet toDocIdSet() {
      return buffers.toDocIdSet(maxDoc, multivalued);
    }

    @Override
    protected int toBitSet(BitSet bitSet) {
      return buffers.toBitSet(bitSet);
    }
  }

  private final int maxDoc;
  // pkg-private for testing
  final boolean multivalued;
  final double numValuesPerDoc;
  
  private BulkAdder adder;

  /** Create a builder that can contain doc IDs between {@code 0} and {@code maxDoc}. */
  public DocIdSetBuilder(int maxDoc) {
    this(maxDoc, -1, -1);
  }

  /**
   * Create a {@link DocIdSetBuilder} instance that is optimized for accumulating docs that match
   * the given {@link Terms}.
   */
  public DocIdSetBuilder(int maxDoc, Terms terms) throws IOException {
    this(maxDoc, terms.getDocCount(), terms.getSumDocFreq());
  }

  /**
   * Create a {@link DocIdSetBuilder} instance that is optimized for accumulating docs that match
   * the given {@link PointValues}.
   */
  public DocIdSetBuilder(int maxDoc, PointValues values, String field) throws IOException {
    this(maxDoc, values.getDocCount(), values.size());
  }

  DocIdSetBuilder(int maxDoc, int docCount, long valueCount) {
    this.maxDoc = maxDoc;
    this.multivalued = docCount < 0 || docCount != valueCount;
    if (docCount <= 0 || valueCount < 0) {
      // assume one value per doc, this means the cost will be overestimated
      // if the docs are actually multi-valued
      this.numValuesPerDoc = 1;
    } else {
      // otherwise compute from index stats
      this.numValuesPerDoc = (double) valueCount / docCount;
    }

    assert numValuesPerDoc >= 1 : "valueCount=" + valueCount + " docCount=" + docCount;

    // For ridiculously small sets, we'll just use a sorted int[]
    // maxDoc >>> 7 is a good value if you want to save memory, lower values
    // such as maxDoc >>> 11 should provide faster building but at the expense
    // of using a full bitset even for quite sparse data
    Buffers buffers = new Buffers(maxDoc >>> 7);
    this.adder = new BufferAdder(buffers, maxDoc, multivalued);
  }

  /**
   * Add the content of the provided {@link DocIdSetIterator} to this builder. NOTE: if you need to
   * build a {@link DocIdSet} out of a single {@link DocIdSetIterator}, you should rather use {@link
   * RoaringDocIdSet.Builder}.
   */
  public void add(DocIdSetIterator iter) throws IOException {
    int cost = (int) Math.min(Integer.MAX_VALUE, iter.cost());
    grow(cost);
    adder.add(iter);
  }

  /**
   * Reserve space and return a {@link BulkAdder} object that can be used to add up to {@code
   * numDocs} documents.
   */
  public BulkAdder grow(int numDocs) {
    if (adder.ensureCapacity(numDocs) == false) {
      FixedBitSet bitSet = new FixedBitSet(maxDoc);
      int counter = adder.toBitSet(bitSet);
      this.adder = new FixedBitSetAdder(bitSet, numValuesPerDoc);
      adder.ensureCapacity(counter + numDocs);
    }
    return adder;
  }
  
  /** Build a {@link DocIdSet} from the accumulated doc IDs. */
  public DocIdSet build() {
    try {
      return adder.toDocIdSet();
    } finally {
      this.adder = null;
    }
  }
}
