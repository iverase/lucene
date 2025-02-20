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
package org.apache.lucene.sandbox.codecs.quantization;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;

/** Default implementation of {@link IVFVectorsWriter}. It uses lucene  {@link KMeans} algoritm to
 * partition the vector space, and then stores the centroids an posting list in a sequential
 * fashion. */
public class DefaultIVFVectorsWriter extends IVFVectorsWriter {

  private final int vectorPerCluster;

  public DefaultIVFVectorsWriter(
      SegmentWriteState state, FlatVectorsWriter rawVectorDelegate, int vectorPerCluster)
      throws IOException {
    super(state, rawVectorDelegate);
    this.vectorPerCluster = vectorPerCluster;
  }

  @Override
  protected PostingLists buildPostingLists(FieldInfo fieldInfo, FloatVectorValues floatVectorValues)
      throws IOException {
    if (floatVectorValues.size() == 0) {
      return new PostingLists() {
        @Override
        public int size() {
          return 0;
        }

        @Override
        public boolean hasNext() {
          return false;
        }

        @Override
        public PostingList next() {
          return null;
        }
      };
    }
    final KMeans.Results kMeans =
        KMeans.cluster(
            floatVectorValues,
            fieldInfo.getVectorSimilarityFunction(),
            ((floatVectorValues.size() - 1) / vectorPerCluster) + 1);
    final int[] clusterOffsets = new int[kMeans.centroidsSize().length];
    int offset = 0;
    for (int i = 0; i < clusterOffsets.length; i++) {
      clusterOffsets[i] = offset;
      offset += kMeans.centroidsSize()[i];
    }
    final int[] clusters = new int[kMeans.vectorCentroids().length];
    for (int i = 0; i < floatVectorValues.size(); i++) {
      final short centroid = kMeans.vectorCentroids()[i];
      clusters[clusterOffsets[centroid]++] = i;
    }
    offset = 0;
    for (int i = 0; i < clusterOffsets.length; i++) {
      clusterOffsets[i] = offset;
      offset += kMeans.centroidsSize()[i];
    }
    return new PostingLists() {
      int i = 0;

      @Override
      public int size() {
        return kMeans.centroids().length;
      }

      @Override
      public boolean hasNext() {
        return i < kMeans.centroids().length;
      }

      @Override
      public PostingList next() {
        assert hasNext();
        final int finalI = i++;
        return new PostingList() {
          int j = 0;

          @Override
          public int size() {
            return kMeans.centroidsSize()[finalI];
          }

          @Override
          public float[] centroid() {
            return kMeans.centroids()[finalI];
          }

          @Override
          public int nextOrd() {
            assert hasNext();
            int offset = clusterOffsets[finalI] + j++;
            return clusters[offset];
          }

          @Override
          public boolean hasNext() {
            return j < kMeans.centroidsSize()[finalI];
          }
        };
      }
    };
  }

  @Override
  protected PostingListWithFileOffset[] writePostingLists(
      IndexOutput output,
      FieldInfo fieldInfo,
      FloatVectorValues floatVectorValues,
      PostingLists postingLists)
      throws IOException {
    final ByteBuffer buffer =
        ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    final PostingListWithFileOffset[] clustersAndOffsets = new PostingListWithFileOffset[postingLists.size()];
    for (int i = 0; i < postingLists.size(); i++) {
      final PostingList postingList = postingLists.next();
      clustersAndOffsets[i] =
          new PostingListWithFileOffset(postingList.centroid(), output.getFilePointer());
      output.writeVInt(postingList.size());
      while (postingList.hasNext()) {
        final int ord = postingList.nextOrd();
        output.writeInt(floatVectorValues.ordToDoc(ord));
        buffer.asFloatBuffer().put(floatVectorValues.vectorValue(ord));
        output.writeBytes(buffer.array(), buffer.array().length);
      }
    }
    return clustersAndOffsets;
  }

  @Override
  protected void writeCentroidsAndOffsets(
      IndexOutput output, FieldInfo fieldInfo, PostingListWithFileOffset[] postingListWithFileOffsets)
      throws IOException {
    final ByteBuffer buffer =
        ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    output.writeInt(vectorPerCluster);
    output.writeInt(postingListWithFileOffsets.length);
    for (PostingListWithFileOffset postingListWithFileOffset : postingListWithFileOffsets) {
      output.writeVLong(postingListWithFileOffset.fileOffset());
      buffer.asFloatBuffer().put(postingListWithFileOffset.centroid());
      output.writeBytes(buffer.array(), buffer.array().length);
    }
  }
}
