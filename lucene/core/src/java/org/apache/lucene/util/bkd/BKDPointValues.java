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
package org.apache.lucene.util.bkd;

import java.io.IOException;
import java.io.UncheckedIOException;
import org.apache.lucene.index.PointValues;

/**
 * A {@link PointValues} wrapper for {@link BKDReader} to handle intersections.
 *
 * @lucene.experimental
 */
public final class BKDPointValues extends PointValues {

  final BKDReader in;

  /** Sole constructor */
  public BKDPointValues(BKDReader in) throws IOException {
    this.in = in;
  }

  /** Create a new {@link BKDReader.IndexTree} */
  public BKDReader.IndexTree getIndexTree() throws IOException {
    return in.getIndexTree();
  }

  @Override
  public void intersect(IntersectVisitor visitor) throws IOException {
    final BKDReader.IndexTree indexTree = in.getIndexTree();
    intersect(visitor, indexTree);
    assert indexTree.moveToParent() == false;
  }

  @Override
  public long estimatePointCount(IntersectVisitor visitor) {
    try {
      final BKDReader.IndexTree indexTree = in.getIndexTree();
      final long count = estimatePointCount(visitor, indexTree);
      assert indexTree.moveToParent() == false;
      return count;
    } catch (IOException ioe) {
      throw new UncheckedIOException(ioe);
    }
  }

  /** Fast path: this is called when the query box fully encompasses all cells under this node. */
  private void addAll(IntersectVisitor visitor, BKDReader.IndexTree index)
      throws IOException {
    // System.out.println("R: addAll nodeID=" + nodeID);
    long maxPointCount = index.size();
    while (maxPointCount >= Integer.MAX_VALUE) {
      // could be >MAX_VALUE if there are more than 2B points in total
      visitor.grow(Integer.MAX_VALUE);
      maxPointCount -= Integer.MAX_VALUE;
    }
    visitor.grow((int) maxPointCount);
    index.visitDocIDs(visitor);
  }

  private void intersect(IntersectVisitor visitor, BKDReader.IndexTree index) throws IOException {
    Relation r = visitor.compare(index.getMinPackedValue(), index.getMaxPackedValue());
    if (r == Relation.CELL_OUTSIDE_QUERY) {
      // This cell is fully outside of the query shape: stop recursing
    } else if (r == Relation.CELL_INSIDE_QUERY) {
      // This cell is fully inside of the query shape: recursively add all points in this cell
      // without filtering
      addAll(visitor, index);
      // The cell crosses the shape boundary, or the cell fully contains the query, so we fall
      // through and do full filtering:
    } else if (index.moveToChild()) {
      do {
        intersect(visitor, index);
      } while (index.moveToSibling());
      index.moveToParent();
    } else {
      // TODO: we can assert that the first value here in fact matches what the index claimed?
      // Leaf node; scan and filter all points in this block:
      index.visitDocValues(visitor);
    }
  }

  private long estimatePointCount(IntersectVisitor visitor, BKDReader.IndexTree index)
      throws IOException {

    Relation r = visitor.compare(index.getMinPackedValue(), index.getMaxPackedValue());

    if (r == Relation.CELL_OUTSIDE_QUERY) {
      // This cell is fully outside of the query shape: stop recursing
      return 0L;
    } else if (r == Relation.CELL_INSIDE_QUERY) {
      return index.size();
    } else if (index.moveToChild()) {
      long cost = 0;
      do {
        cost += estimatePointCount(visitor, index);
      } while (index.moveToSibling());
      index.moveToParent();
      return cost;
    } else {
      // Assume half the points matched
      return (in.getConfig().maxPointsInLeafNode + 1) / 2;
    }
  }

  @Override
  public byte[] getMinPackedValue() {
    return in.getMinPackedValue();
  }

  @Override
  public byte[] getMaxPackedValue() {
    return in.getMaxPackedValue();
  }

  @Override
  public int getNumDimensions() {
    return in.getConfig().numDims;
  }

  @Override
  public int getNumIndexDimensions() {
    return in.getConfig().numIndexDims;
  }

  @Override
  public int getBytesPerDimension() {
    return in.getConfig().bytesPerDim;
  }

  @Override
  public long size() {
    return in.getPointCount();
  }

  @Override
  public int getDocCount() {
    return in.getDocCount();
  }
}
