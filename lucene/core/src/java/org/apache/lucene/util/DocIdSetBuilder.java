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
  }

  private static class FixedBitSetAdder extends BulkAdder {
    final FixedBitSet bitSet;

    FixedBitSetAdder(FixedBitSet bitSet) {
      this.bitSet = bitSet;
    }

    @Override
    public void add(int doc) {
      bitSet.set(doc);
    }

    @Override
    public void add(DocIdSetIterator iterator) throws IOException {
      bitSet.or(iterator);
    }
  }

  private static class BufferAdder extends BulkAdder {
    final Buffers buffer;

    BufferAdder(Buffers buffer) {
      this.buffer = buffer;
    }

    @Override
    public void add(int doc) {
      buffer.addDoc(doc);
    }
  }

  private final int maxDoc;
  // pkg-private for testing
  final boolean multivalued;
  final double numValuesPerDoc;

  private Buffers buffers;
  private FixedBitSet bitSet;

  private long counter = -1;
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
    this.buffers = new Buffers(maxDoc >>> 7);
    this.adder = new BufferAdder(buffers);
    this.bitSet = null;
  }

  /**
   * Add the content of the provided {@link DocIdSetIterator} to this builder. NOTE: if you need to
   * build a {@link DocIdSet} out of a single {@link DocIdSetIterator}, you should rather use {@link
   * RoaringDocIdSet.Builder}.
   */
  public void add(DocIdSetIterator iter) throws IOException {
    if (bitSet != null) {
      bitSet.or(iter);
      return;
    }
    int cost = (int) Math.min(Integer.MAX_VALUE, iter.cost());
    BulkAdder adder = grow(cost);
    for (int i = 0; i < cost; ++i) {
      int doc = iter.nextDoc();
      if (doc == DocIdSetIterator.NO_MORE_DOCS) {
        return;
      }
      adder.add(doc);
    }
    for (int doc = iter.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iter.nextDoc()) {
      grow(1).add(doc);
    }
  }

  /**
   * Reserve space and return a {@link BulkAdder} object that can be used to add up to {@code
   * numDocs} documents.
   */
  public BulkAdder grow(int numDocs) {
    if (bitSet == null) {
      if (buffers.ensureBufferCapacity(numDocs) == false) {
        upgradeToBitSet();
        counter += numDocs;
      }
    } else {
      counter += numDocs;
    }
    return adder;
  }

  private void upgradeToBitSet() {
    assert bitSet == null;
    FixedBitSet bitSet = new FixedBitSet(maxDoc);
    this.counter = buffers.toBitSet(bitSet);
    this.bitSet = bitSet;
    this.buffers = null;
    this.adder = new FixedBitSetAdder(bitSet);
  }

  /** Build a {@link DocIdSet} from the accumulated doc IDs. */
  public DocIdSet build() {
    try {
      if (bitSet != null) {
        assert counter >= 0;
        final long cost = Math.round(counter / numValuesPerDoc);
        return new BitDocIdSet(bitSet, cost);
      } else {
        return buffers.toDocIdSet(maxDoc, multivalued);
      }
    } finally {
      this.buffers = null;
      this.bitSet = null;
    }
  }
}
