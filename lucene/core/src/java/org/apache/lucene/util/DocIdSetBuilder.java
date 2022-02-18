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
import java.util.function.IntConsumer;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.Terms;
import org.apache.lucene.search.DocIdSet;
import org.apache.lucene.search.DocIdSetIterator;

/**
 * A builder of {@link DocIdSet}s for Terms. At first it uses a sparse structure to gather
 * documents, and then upgrades to a non-sparse bit set once enough hits match.
 *
 * <p>Documents are added via {@link #add(DocIdSetIterator)} as a bulk or via {@link #add(int)} for
 * individual documents.
 *
 * <p>See {@link PointsDocIdSetBuilder} if you are not working with {@link PointValues}
 *
 * @lucene.internal
 */
public final class DocIdSetBuilder {

  private final int maxDoc;
  // pkg-private for testing
  final double numValuesPerDoc;
  Buffers buffers;
  private FixedBitSet bitSet;
  private IntConsumer adder;
  private long counter = -1;

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

  private DocIdSetBuilder(int maxDoc, int docCount, long valueCount) {
    this.maxDoc = maxDoc;
    final boolean multivalued = docCount < 0 || docCount != valueCount;
    if (docCount <= 0 || valueCount < 0) {
      // assume one value per doc, this means the cost will be overestimated
      // if the docs are actually multi-valued
      this.numValuesPerDoc = 1;
    } else {
      // otherwise compute from index stats
      this.numValuesPerDoc = (double) valueCount / docCount;
    }
    assert numValuesPerDoc >= 1 : "valueCount=" + docCount + " docCount=" + valueCount;
    this.buffers = new Buffers(maxDoc, multivalued);
    this.bitSet = null;
    this.adder = doc -> buffers.addDoc(doc);
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
    grow(cost);
    for (int i = 0; i < cost; ++i) {
      int doc = iter.nextDoc();
      if (doc == DocIdSetIterator.NO_MORE_DOCS) {
        return;
      }
      adder.accept(doc);
    }
    for (int doc = iter.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iter.nextDoc()) {
      add(doc);
    }
  }

  /** Add a single document to this builder. */
  public void add(int doc) {
    grow(1);
    adder.accept(doc);
  }

  /** Reserve space and up to {@code numDocs} documents. */
  private void grow(int numDocs) {
    if (bitSet == null) {
      if (buffers.ensureBufferCapacity(numDocs) == false) {
        upgradeToBitSet();
        counter += numDocs;
      }
    } else {
      counter += numDocs;
    }
  }

  private void upgradeToBitSet() {
    assert bitSet == null;
    FixedBitSet bitSet = new FixedBitSet(maxDoc);
    this.counter = buffers.toBitSet(bitSet);
    this.bitSet = bitSet;
    adder = doc -> bitSet.set(doc);
    this.buffers = null;
  }

  /** Build a {@link DocIdSet} from the accumulated doc IDs. */
  public DocIdSet build() {
    try {
      if (bitSet != null) {
        assert counter >= 0;
        final long cost = Math.round(counter / numValuesPerDoc);
        return new BitDocIdSet(bitSet, cost);
      } else {
        return buffers.toDocIdSet();
      }
    } finally {
      this.buffers = null;
      this.bitSet = null;
    }
  }
}
