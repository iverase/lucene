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
import java.util.Arrays;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.MathUtil;

/**
 * Handles reading a block KD-tree previously written with {@link BKDWriter}.
 *
 * @lucene.experimental
 */
public class BKDDefaultReader implements BKDReader {

  final BKDConfig config;
  final int numLeaves;
  // Packed array of byte[] holding all docs and values:
  final IndexInput in;
  final byte[] minPackedValue;
  final byte[] maxPackedValue;
  final long pointCount;
  final int docCount;
  final int version;
  final long minLeafBlockFP;
  // Packed array of byte[] holding all split values in the full binary tree:
  private final IndexInput packedIndex;

  /**
   * Caller must pre-seek the provided {@link IndexInput} to the index location that {@link
   * BKDWriter#finish} returned. BKD tree is always stored off-heap.
   */
  public BKDDefaultReader(IndexInput metaIn, IndexInput indexIn, IndexInput dataIn)
      throws IOException {
    version =
        CodecUtil.checkHeader(
            metaIn, BKDWriter.CODEC_NAME, BKDWriter.VERSION_START, BKDWriter.VERSION_CURRENT);
    final int numDims = metaIn.readVInt();
    final int numIndexDims;
    if (version >= BKDWriter.VERSION_SELECTIVE_INDEXING) {
      numIndexDims = metaIn.readVInt();
    } else {
      numIndexDims = numDims;
    }
    final int maxPointsInLeafNode = metaIn.readVInt();
    final int bytesPerDim = metaIn.readVInt();
    config = new BKDConfig(numDims, numIndexDims, bytesPerDim, maxPointsInLeafNode);

    // Read index:
    numLeaves = metaIn.readVInt();
    assert numLeaves > 0;

    minPackedValue = new byte[config.packedIndexBytesLength];
    maxPackedValue = new byte[config.packedIndexBytesLength];

    metaIn.readBytes(minPackedValue, 0, config.packedIndexBytesLength);
    metaIn.readBytes(maxPackedValue, 0, config.packedIndexBytesLength);

    for (int dim = 0; dim < config.numIndexDims; dim++) {
      if (Arrays.compareUnsigned(
              minPackedValue,
              dim * config.bytesPerDim,
              dim * config.bytesPerDim + config.bytesPerDim,
              maxPackedValue,
              dim * config.bytesPerDim,
              dim * config.bytesPerDim + config.bytesPerDim)
          > 0) {
        throw new CorruptIndexException(
            "minPackedValue "
                + new BytesRef(minPackedValue)
                + " is > maxPackedValue "
                + new BytesRef(maxPackedValue)
                + " for dim="
                + dim,
            metaIn);
      }
    }

    pointCount = metaIn.readVLong();
    docCount = metaIn.readVInt();

    int numIndexBytes = metaIn.readVInt();
    long indexStartPointer;
    if (version >= BKDWriter.VERSION_META_FILE) {
      minLeafBlockFP = metaIn.readLong();
      indexStartPointer = metaIn.readLong();
    } else {
      indexStartPointer = indexIn.getFilePointer();
      minLeafBlockFP = indexIn.readVLong();
      indexIn.seek(indexStartPointer);
    }
    this.packedIndex = indexIn.slice("packedIndex", indexStartPointer, numIndexBytes);
    this.in = dataIn;
  }

  @Override
  public BKDConfig getConfig() {
    return config;
  }

  @Override
  public byte[] getMinPackedValue() {
    return minPackedValue.clone();
  }

  @Override
  public byte[] getMaxPackedValue() {
    return maxPackedValue.clone();
  }

  @Override
  public long getPointCount() {
    return pointCount;
  }

  @Override
  public int getDocCount() {
    return docCount;
  }

  @Override
  public BKDReader.IndexTree getIndexTree() throws IOException {
    return new IndexTree(
        packedIndex.clone(),
        this.in,
        config,
        numLeaves,
        version,
        minPackedValue,
        maxPackedValue);
  }
  
  private static class ScratchObjects {

    private final byte[] scratchDataPackedValue, scratchMinIndexPackedValue, scratchMaxIndexPackedValue;
    private final int[] commonPrefixLengths;
    private final BKDReaderDocIDSetIterator scratchIterator;
    
    public ScratchObjects(BKDConfig config) {
      this.scratchIterator =  new BKDReaderDocIDSetIterator(config.maxPointsInLeafNode);
      this.commonPrefixLengths = new int[config.numDims];
      this.scratchDataPackedValue = new byte[config.packedBytesLength];
      this.scratchMinIndexPackedValue =  new byte[config.packedIndexBytesLength];
      this.scratchMaxIndexPackedValue = new byte[config.packedIndexBytesLength];
    }
    
  }

  private static class IndexTree implements BKDReader.IndexTree {
    private int nodeID;
    // during clone, the node root can be different to 1
    private final int nodeRoot;
    // level is 1-based so that we can do level-1 w/o checking each time:
    private int level;
    // used to read the packed tree off-heap
    private final IndexInput innerNodes;
    // used to read the packed leaves off-heap
    private final IndexInput leafNodes;
    // holds the minimum (left most) leaf block file pointer for each level we've recursed to:
    private final long[] leafBlockFPStack;
    // holds the address, in the off-heap index, of the right-node of each level:
    private final int[] rightNodePositions;
    // holds the splitDim for each level:
    private final int[] splitDimsPos;
    // true if the per-dim delta we read for the node at this level is a negative offset vs. the
    // last split on this dim; this is a packed
    // 2D array, i.e. to access array[level][dim] you read from negativeDeltas[level*numDims+dim].
    // this will be true if the last time we
    // split on this dimension, we next pushed to the left sub-tree:
    private final boolean[] negativeDeltas;
    // holds the packed per-level split values
    private final byte[][] splitValuesStack;
    // holds the min / max value of the current node.
    private final byte[] minPackedValue, maxPackedValue;
    // holds the previous value of the split dimension
    private final byte[][] splitDimValueStack;
    // tree parameters
    private final BKDConfig config;
    // number of leaves
    private final int leafNodeOffset;
    // version of the index
    private final int version;
    // helper object for reading doc values
    private final ScratchObjects scratcObjects;

    private IndexTree(
        IndexInput innerNodes,
        IndexInput leafNodes,
        BKDConfig config,
        int numLeaves,
        int version,
        byte[] minPackedValue,
        byte[] maxPackedValue)
        throws IOException {
      this(
          innerNodes,
          leafNodes,
          config,
          numLeaves,
          version,
          1,
          1,
          minPackedValue,
          maxPackedValue,
          new ScratchObjects(config)
      );
      // read root node
      readNodeData(false);
    }

    private IndexTree(
        IndexInput innerNodes,
        IndexInput leafNodes,
        BKDConfig config,
        int numLeaves,
        int version,
        int nodeID,
        int level,
        byte[] minPackedValue,
        byte[] maxPackedValue,
        ScratchObjects scratchObjects) {
      this.config = config;
      this.version = version;
      this.nodeID = nodeID;
      this.nodeRoot = nodeID;
      this.level = level;
      leafNodeOffset = numLeaves;
      this.innerNodes = innerNodes;
      this.leafNodes = leafNodes;
      this.minPackedValue = minPackedValue.clone();
      this.maxPackedValue = maxPackedValue.clone();
      // stack arrays that keep information at different levels
      int treeDepth = getTreeDepth(numLeaves);
      splitDimValueStack = new byte[treeDepth + 1][];
      splitValuesStack = new byte[treeDepth + 1][];
      splitValuesStack[0] = new byte[config.packedIndexBytesLength];
      leafBlockFPStack = new long[treeDepth + 1];
      rightNodePositions = new int[treeDepth + 1];
      splitDimsPos = new int[treeDepth + 1];
      negativeDeltas = new boolean[config.numIndexDims * (treeDepth + 1)];
      // scratch objects, reused between clones so NN search are not creating those objects
      // in every clone.
      this.scratcObjects = scratchObjects;
    }

    @Override
    public BKDReader.IndexTree clone() {
      BKDDefaultReader.IndexTree index =
          new BKDDefaultReader.IndexTree(
              innerNodes.clone(),
              leafNodes,
              config,
              leafNodeOffset,
              version,
              nodeID,
              level,
              minPackedValue,
              maxPackedValue,
              scratcObjects);
      index.leafBlockFPStack[index.level] = leafBlockFPStack[level];
      //if (isLeafNode() == false) {
        // copy node data
        index.rightNodePositions[index.level] = rightNodePositions[level];
        index.splitValuesStack[index.level] = splitValuesStack[level].clone();
        System.arraycopy(
            negativeDeltas,
            level * config.numIndexDims,
            index.negativeDeltas,
            level * config.numIndexDims,
            config.numIndexDims);
        index.splitDimsPos[level] = splitDimsPos[level];
      //}
      return index;
    }

    @Override
    public byte[] getMinPackedValue() {
      return minPackedValue;
    }

    @Override
    public byte[] getMaxPackedValue() {
      return maxPackedValue;
    }

    @Override
    public boolean moveToChild() throws IOException {
      if (isLeafNode()) {
        return false;
      }
      final int splitDimPos = splitDimsPos[level];
      pushLeft(splitDimPos);
      // add the split dim value:
      System.arraycopy(
              splitValuesStack[level-1], splitDimPos, maxPackedValue, splitDimPos, config.bytesPerDim);
      return true;
    }

    private void pushLeft(int splitDimPos) throws IOException {
      if (splitDimValueStack[level] == null) {
        splitDimValueStack[level] = new byte[config.bytesPerDim];
      }
      // save the dimension we are going to change
      System.arraycopy(
          maxPackedValue, splitDimPos, splitDimValueStack[level], 0, config.bytesPerDim);
      assert Arrays.compareUnsigned(
                  maxPackedValue,
                  splitDimPos,
                  splitDimPos + config.bytesPerDim,
                  splitValuesStack[level],
                  splitDimPos,
                  splitDimPos + config.bytesPerDim)
              >= 0
          : "config.bytesPerDim="
              + config.bytesPerDim
              + " splitDim="
              + (splitDimsPos[level] / config.bytesPerDim)
              + " config.numIndexDims="
              + config.numIndexDims
              + " config.numDims="
              + config.numDims;
      nodeID *= 2;
      level++;
      readNodeData(true);
    }

    private void pushRight(int splitDimPos) throws IOException {
      // we should have already visit the left node
      assert splitDimValueStack[level] != null;
      // save the dimension we are going to change
      System.arraycopy(
          minPackedValue, splitDimPos, splitDimValueStack[level], 0, config.bytesPerDim);
      assert Arrays.compareUnsigned(
                  minPackedValue,
                  splitDimPos,
                  splitDimPos + config.bytesPerDim,
                  splitValuesStack[level],
                  splitDimPos,
                  splitDimPos + config.bytesPerDim)
              <= 0
          : "config.bytesPerDim="
              + config.bytesPerDim
              + " splitDim="
              + (splitDimsPos[level] / config.bytesPerDim)
              + " config.numIndexDims="
              + config.numIndexDims
              + " config.numDims="
              + config.numDims;
      final int nodePosition = rightNodePositions[level];
      assert nodePosition >= innerNodes.getFilePointer()
          : "nodePosition = " + nodePosition + " < currentPosition=" + innerNodes.getFilePointer();
      innerNodes.seek(nodePosition);
      nodeID = 2 * nodeID + 1;
      level++;
      readNodeData(false);
    }

    @Override
    public boolean moveToSibling() throws IOException {
      if (nodeID != nodeRoot && (nodeID & 1) == 0) {
        moveToParent();
        final int splitDimPos = splitDimsPos[level];
        pushRight(splitDimPos);
        // add the split dim value:
        System.arraycopy(
                splitValuesStack[level-1], splitDimPos, minPackedValue, splitDimPos, config.bytesPerDim);
        assert nodeExists();
        return true;
      }
      return false;
    }

    private void pop() {
      nodeID /= 2;
      level--;
    }

    @Override
    public boolean moveToParent() {
      if (nodeID == nodeRoot) {
        return false;
      }
      // restore the split dimension
      System.arraycopy(
              splitDimValueStack[level - 1],
              0,
              (nodeID & 1) == 0 ? maxPackedValue : minPackedValue,
              splitDimsPos[level - 1],
              config.bytesPerDim);
      pop();
      return true;
    }

    private boolean isLeafNode() {
      return nodeID >= leafNodeOffset;
    }

    private boolean nodeExists() {
      return nodeID - leafNodeOffset < leafNodeOffset;
    }

    /** Only valid after pushLeft or pushRight, not pop! */
    private long getLeafBlockFP() {
      assert isLeafNode() : "nodeID=" + nodeID + " is not a leaf";
      return leafBlockFPStack[level];
    }

    @Override
    public long size() {
      int leftMostLeafNode = nodeID;
      while (leftMostLeafNode < leafNodeOffset) {
        leftMostLeafNode = leftMostLeafNode * 2;
      }
      int rightMostLeafNode = nodeID;
      while (rightMostLeafNode < leafNodeOffset) {
        rightMostLeafNode = rightMostLeafNode * 2 + 1;
      }
      final int numLeaves;
      if (rightMostLeafNode >= leftMostLeafNode) {
        // both are on the same level
        numLeaves = rightMostLeafNode - leftMostLeafNode + 1;
      } else {
        // left is one level deeper than right
        numLeaves = rightMostLeafNode - leftMostLeafNode + 1 + leafNodeOffset;
      }
      assert numLeaves == getNumLeavesSlow(nodeID) : numLeaves + " " + getNumLeavesSlow(nodeID);
      return (long) numLeaves * config.maxPointsInLeafNode;
    }

    @Override
    public void visitDocIDs(PointValues.IntersectVisitor visitor) throws IOException {
      if (isLeafNode()) {
        // TODO: we can assert that the first value here in fact matches what the index claimed?
        // Leaf node
        leafNodes.seek(getLeafBlockFP());
        // How many points are stored in this leaf cell:
        final int count = leafNodes.readVInt();
        DocIdsWriter.readInts(leafNodes, count, visitor);
      } else {
        final int splitDimPos = splitDimsPos[level];
        pushLeft(splitDimPos);
        visitDocIDs(visitor);
        pop();
        pushRight(splitDimPos);
        visitDocIDs(visitor);
        pop();
      }
    }

    @Override
    public void visitDocValues(PointValues.IntersectVisitor visitor) throws IOException {
      visitDocValues(visitor, getLeafBlockFP());
    }

    private void visitDocValues(PointValues.IntersectVisitor visitor, long fp) throws IOException {
      // Leaf node; scan and filter all points in this block:
      int count = readDocIDs(leafNodes, fp, scratcObjects.scratchIterator);
      if (version >= BKDWriter.VERSION_LOW_CARDINALITY_LEAVES) {
        visitDocValuesWithCardinality(
                scratcObjects.commonPrefixLengths,
                scratcObjects.scratchDataPackedValue,
                scratcObjects.scratchMinIndexPackedValue,
                scratcObjects.scratchMaxIndexPackedValue,
            leafNodes,
                scratcObjects.scratchIterator,
            count,
            visitor);
      } else {
        visitDocValuesNoCardinality(
                scratcObjects.commonPrefixLengths,
                scratcObjects.scratchDataPackedValue,
                scratcObjects.scratchMinIndexPackedValue,
                scratcObjects.scratchMaxIndexPackedValue,
            leafNodes,
                scratcObjects.scratchIterator,
            count,
            visitor);
      }
    }

    private int readDocIDs(IndexInput in, long blockFP, BKDReaderDocIDSetIterator iterator)
        throws IOException {
      in.seek(blockFP);
      // How many points are stored in this leaf cell:
      int count = in.readVInt();

      DocIdsWriter.readInts(in, count, iterator.docIDs);

      return count;
    }

    // for assertions
    private int getNumLeavesSlow(int node) {
      if (node >= 2 * leafNodeOffset) {
        return 0;
      } else if (node >= leafNodeOffset) {
        return 1;
      } else {
        final int leftCount = getNumLeavesSlow(node * 2);
        final int rightCount = getNumLeavesSlow(node * 2 + 1);
        return leftCount + rightCount;
      }
    }

    private void readNodeData(boolean isLeft) throws IOException {
      leafBlockFPStack[level] = leafBlockFPStack[level - 1];
      if (isLeft == false) {
        // read leaf block FP delta
        leafBlockFPStack[level] += innerNodes.readVLong();
      }

      if (isLeafNode() == false) {
        System.arraycopy(
            negativeDeltas,
            (level - 1) * config.numIndexDims,
            negativeDeltas,
            level * config.numIndexDims,
            config.numIndexDims);
        negativeDeltas[level * config.numIndexDims + (splitDimsPos[level - 1] / config.bytesPerDim)] = isLeft;

        if (splitValuesStack[level] == null) {
          splitValuesStack[level] = splitValuesStack[level - 1].clone();
        } else {
          System.arraycopy(
              splitValuesStack[level - 1],
              0,
              splitValuesStack[level],
              0,
              config.packedIndexBytesLength);
        }

        // read split dim, prefix, firstDiffByteDelta encoded as int:
        int code = innerNodes.readVInt();
        int splitDim = code % config.numIndexDims;
        splitDimsPos[level] = splitDim * config.bytesPerDim;
        code /= config.numIndexDims;
        int prefix = code % (1 + config.bytesPerDim);
        int suffix = config.bytesPerDim - prefix;

        if (suffix > 0) {
          int firstDiffByteDelta = code / (1 + config.bytesPerDim);
          if (negativeDeltas[level * config.numIndexDims + splitDim]) {
            firstDiffByteDelta = -firstDiffByteDelta;
          }
          int oldByte =
              splitValuesStack[level][splitDim * config.bytesPerDim + prefix] & 0xFF;
          splitValuesStack[level][splitDim * config.bytesPerDim + prefix] =
              (byte) (oldByte + firstDiffByteDelta);
          innerNodes.readBytes(
              splitValuesStack[level],
                  splitDim * config.bytesPerDim + prefix + 1,
              suffix - 1);
        } else {
          // our split value is == last split value in this dim, which can happen when there are
          // many duplicate values
        }

        int leftNumBytes;
        if (nodeID * 2 < leafNodeOffset) {
          leftNumBytes = innerNodes.readVInt();
        } else {
          leftNumBytes = 0;
        }
        rightNodePositions[level] = Math.toIntExact(innerNodes.getFilePointer()) + leftNumBytes;
      }
    }

    private int getTreeDepth(int numLeaves) {
      // First +1 because all the non-leave nodes makes another power
      // of 2; e.g. to have a fully balanced tree with 4 leaves you
      // need a depth=3 tree:

      // Second +1 because MathUtil.log computes floor of the logarithm; e.g.
      // with 5 leaves you need a depth=4 tree:
      return MathUtil.log(numLeaves, 2) + 2;
    }

    private void visitDocValuesNoCardinality(
        int[] commonPrefixLengths,
        byte[] scratchDataPackedValue,
        byte[] scratchMinIndexPackedValue,
        byte[] scratchMaxIndexPackedValue,
        IndexInput in,
        BKDReaderDocIDSetIterator scratchIterator,
        int count,
        PointValues.IntersectVisitor visitor)
        throws IOException {
      readCommonPrefixes(commonPrefixLengths, scratchDataPackedValue, in);

      if (config.numIndexDims != 1 && version >= BKDWriter.VERSION_LEAF_STORES_BOUNDS) {
        byte[] minPackedValue = scratchMinIndexPackedValue;
        System.arraycopy(
            scratchDataPackedValue, 0, minPackedValue, 0, config.packedIndexBytesLength);
        byte[] maxPackedValue = scratchMaxIndexPackedValue;
        // Copy common prefixes before reading adjusted box
        System.arraycopy(minPackedValue, 0, maxPackedValue, 0, config.packedIndexBytesLength);
        readMinMax(commonPrefixLengths, minPackedValue, maxPackedValue, in);

        // The index gives us range of values for each dimension, but the actual range of values
        // might be much more narrow than what the index told us, so we double check the relation
        // here, which is cheap yet might help figure out that the block either entirely matches
        // or does not match at all. This is especially more likely in the case that there are
        // multiple dimensions that have correlation, ie. splitting on one dimension also
        // significantly changes the range of values in another dimension.
        PointValues.Relation r = visitor.compare(minPackedValue, maxPackedValue);
        if (r == PointValues.Relation.CELL_OUTSIDE_QUERY) {
          return;
        }
        visitor.grow(count);

        if (r == PointValues.Relation.CELL_INSIDE_QUERY) {
          for (int i = 0; i < count; ++i) {
            visitor.visit(scratchIterator.docIDs[i]);
          }
          return;
        }
      } else {
        visitor.grow(count);
      }

      int compressedDim = readCompressedDim(in);

      if (compressedDim == -1) {
        visitUniqueRawDocValues(scratchDataPackedValue, scratchIterator, count, visitor);
      } else {
        visitCompressedDocValues(
            commonPrefixLengths,
            scratchDataPackedValue,
            in,
            scratchIterator,
            count,
            visitor,
            compressedDim);
      }
    }

    private void visitDocValuesWithCardinality(
        int[] commonPrefixLengths,
        byte[] scratchDataPackedValue,
        byte[] scratchMinIndexPackedValue,
        byte[] scratchMaxIndexPackedValue,
        IndexInput in,
        BKDReaderDocIDSetIterator scratchIterator,
        int count,
        PointValues.IntersectVisitor visitor)
        throws IOException {

      readCommonPrefixes(commonPrefixLengths, scratchDataPackedValue, in);
      int compressedDim = readCompressedDim(in);
      if (compressedDim == -1) {
        // all values are the same
        visitor.grow(count);
        visitUniqueRawDocValues(scratchDataPackedValue, scratchIterator, count, visitor);
      } else {
        if (config.numIndexDims != 1) {
          byte[] minPackedValue = scratchMinIndexPackedValue;
          System.arraycopy(
              scratchDataPackedValue, 0, minPackedValue, 0, config.packedIndexBytesLength);
          byte[] maxPackedValue = scratchMaxIndexPackedValue;
          // Copy common prefixes before reading adjusted box
          System.arraycopy(minPackedValue, 0, maxPackedValue, 0, config.packedIndexBytesLength);
          readMinMax(commonPrefixLengths, minPackedValue, maxPackedValue, in);

          // The index gives us range of values for each dimension, but the actual range of values
          // might be much more narrow than what the index told us, so we double check the relation
          // here, which is cheap yet might help figure out that the block either entirely matches
          // or does not match at all. This is especially more likely in the case that there are
          // multiple dimensions that have correlation, ie. splitting on one dimension also
          // significantly changes the range of values in another dimension.
          PointValues.Relation r = visitor.compare(minPackedValue, maxPackedValue);
          if (r == PointValues.Relation.CELL_OUTSIDE_QUERY) {
            return;
          }
          visitor.grow(count);

          if (r == PointValues.Relation.CELL_INSIDE_QUERY) {
            for (int i = 0; i < count; ++i) {
              visitor.visit(scratchIterator.docIDs[i]);
            }
            return;
          }
        } else {
          visitor.grow(count);
        }
        if (compressedDim == -2) {
          // low cardinality values
          visitSparseRawDocValues(
              commonPrefixLengths, scratchDataPackedValue, in, scratchIterator, count, visitor);
        } else {
          // high cardinality
          visitCompressedDocValues(
              commonPrefixLengths,
              scratchDataPackedValue,
              in,
              scratchIterator,
              count,
              visitor,
              compressedDim);
        }
      }
    }

    private void readMinMax(
        int[] commonPrefixLengths, byte[] minPackedValue, byte[] maxPackedValue, IndexInput in)
        throws IOException {
      for (int dim = 0; dim < config.numIndexDims; dim++) {
        int prefix = commonPrefixLengths[dim];
        in.readBytes(
            minPackedValue, dim * config.bytesPerDim + prefix, config.bytesPerDim - prefix);
        in.readBytes(
            maxPackedValue, dim * config.bytesPerDim + prefix, config.bytesPerDim - prefix);
      }
    }

    // read cardinality and point
    private void visitSparseRawDocValues(
        int[] commonPrefixLengths,
        byte[] scratchPackedValue,
        IndexInput in,
        BKDReaderDocIDSetIterator scratchIterator,
        int count,
        PointValues.IntersectVisitor visitor)
        throws IOException {
      int i;
      for (i = 0; i < count; ) {
        int length = in.readVInt();
        for (int dim = 0; dim < config.numDims; dim++) {
          int prefix = commonPrefixLengths[dim];
          in.readBytes(
              scratchPackedValue, dim * config.bytesPerDim + prefix, config.bytesPerDim - prefix);
        }
        scratchIterator.reset(i, length);
        visitor.visit(scratchIterator, scratchPackedValue);
        i += length;
      }
      if (i != count) {
        throw new CorruptIndexException(
            "Sub blocks do not add up to the expected count: " + count + " != " + i, in);
      }
    }

    // point is under commonPrefix
    private void visitUniqueRawDocValues(
        byte[] scratchPackedValue,
        BKDReaderDocIDSetIterator scratchIterator,
        int count,
        PointValues.IntersectVisitor visitor)
        throws IOException {
      scratchIterator.reset(0, count);
      visitor.visit(scratchIterator, scratchPackedValue);
    }

    private void visitCompressedDocValues(
        int[] commonPrefixLengths,
        byte[] scratchPackedValue,
        IndexInput in,
        BKDReaderDocIDSetIterator scratchIterator,
        int count,
        PointValues.IntersectVisitor visitor,
        int compressedDim)
        throws IOException {
      // the byte at `compressedByteOffset` is compressed using run-length compression,
      // other suffix bytes are stored verbatim
      final int compressedByteOffset =
          compressedDim * config.bytesPerDim + commonPrefixLengths[compressedDim];
      commonPrefixLengths[compressedDim]++;
      int i;
      for (i = 0; i < count; ) {
        scratchPackedValue[compressedByteOffset] = in.readByte();
        final int runLen = Byte.toUnsignedInt(in.readByte());
        for (int j = 0; j < runLen; ++j) {
          for (int dim = 0; dim < config.numDims; dim++) {
            int prefix = commonPrefixLengths[dim];
            in.readBytes(
                scratchPackedValue, dim * config.bytesPerDim + prefix, config.bytesPerDim - prefix);
          }
          visitor.visit(scratchIterator.docIDs[i + j], scratchPackedValue);
        }
        i += runLen;
      }
      if (i != count) {
        throw new CorruptIndexException(
            "Sub blocks do not add up to the expected count: " + count + " != " + i, in);
      }
    }

    private int readCompressedDim(IndexInput in) throws IOException {
      int compressedDim = in.readByte();
      if (compressedDim < -2
          || compressedDim >= config.numDims
          || (version < BKDWriter.VERSION_LOW_CARDINALITY_LEAVES && compressedDim == -2)) {
        throw new CorruptIndexException("Got compressedDim=" + compressedDim, in);
      }
      return compressedDim;
    }

    private void readCommonPrefixes(
        int[] commonPrefixLengths, byte[] scratchPackedValue, IndexInput in) throws IOException {
      for (int dim = 0; dim < config.numDims; dim++) {
        int prefix = in.readVInt();
        commonPrefixLengths[dim] = prefix;
        if (prefix > 0) {
          in.readBytes(scratchPackedValue, dim * config.bytesPerDim, prefix);
        }
        // System.out.println("R: " + dim + " of " + numDims + " prefix=" + prefix);
      }
    }

    @Override
    public String toString() {
      return "nodeID=" + nodeID;
    }
  }

  /** Reusable {@link DocIdSetIterator} to handle low cardinality leaves. */
  private static class BKDReaderDocIDSetIterator extends DocIdSetIterator {

    private int idx;
    private int length;
    private int offset;
    private int docID;
    final int[] docIDs;

    public BKDReaderDocIDSetIterator(int maxPointsInLeafNode) {
      this.docIDs = new int[maxPointsInLeafNode];
    }

    @Override
    public int docID() {
      return docID;
    }

    private void reset(int offset, int length) {
      this.offset = offset;
      this.length = length;
      assert offset + length <= docIDs.length;
      this.docID = -1;
      this.idx = 0;
    }

    @Override
    public int nextDoc() throws IOException {
      if (idx == length) {
        docID = DocIdSetIterator.NO_MORE_DOCS;
      } else {
        docID = docIDs[offset + idx];
        idx++;
      }
      return docID;
    }

    @Override
    public int advance(int target) throws IOException {
      return slowAdvance(target);
    }

    @Override
    public long cost() {
      return length;
    }
  }
}
