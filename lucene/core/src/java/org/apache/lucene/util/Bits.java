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

import org.apache.lucene.search.DocIdSetIterator;

/**
 * Interface for Bitset-like structures.
 *
 * @lucene.experimental
 */
public interface Bits {
  /**
   * Returns the value of the bit with the specified <code>index</code>.
   *
   * @param index index, should be non-negative and &lt; {@link #length()}. The result of passing
   *     negative or out of bounds values is undefined by this interface, <b>just don't do it!</b>
   * @return <code>true</code> if the bit is set, <code>false</code> otherwise.
   */
  boolean get(int index);

  /** Returns the number of bits in this set */
  int length();

  /**
   * Apply this {@code Bits} instance to the given {@link FixedBitSet}, which starts at the given
   * {@code offset}.
   *
   * <p>This should behave the same way as the default implementation, which does the following:
   *
   * <pre class="prettyprint">
   * for (int i = bitSet.nextSetBit(0);
   *     i != DocIdSetIterator.NO_MORE_DOCS;
   *     i = i + 1 >= bitSet.length() ? DocIdSetIterator.NO_MORE_DOCS : bitSet.nextSetBit(i + 1)) {
   *   if (get(offset + i) == false) {
   *     bitSet.clear(i);
   *   }
   * }
   * </pre>
   */
  default void applyMask(FixedBitSet bitSet, int offset) {
    for (int i = bitSet.nextSetBit(0);
        i != DocIdSetIterator.NO_MORE_DOCS;
        i = i + 1 >= bitSet.length() ? DocIdSetIterator.NO_MORE_DOCS : bitSet.nextSetBit(i + 1)) {
      if (get(offset + i) == false) {
        bitSet.clear(i);
      }
    }
  }

  Bits[] EMPTY_ARRAY = new Bits[0];

  /** Bits impl of the specified length with all bits set. */
  class MatchAllBits implements Bits {
    final int len;

    public MatchAllBits(int len) {
      this.len = len;
    }

    @Override
    public boolean get(int index) {
      return true;
    }

    @Override
    public int length() {
      return len;
    }
  }

  /** Bits impl of the specified length with no bits set. */
  class MatchNoBits implements Bits {
    final int len;

    public MatchNoBits(int len) {
      this.len = len;
    }

    @Override
    public boolean get(int index) {
      return false;
    }

    @Override
    public int length() {
      return len;
    }
  }
}
