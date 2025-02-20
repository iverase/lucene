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
import java.util.Arrays;

import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.PriorityQueue;

/**
 * Default implementation of {@link IVFVectorsReader}. It scores the posting lists centroids
 * using brute force and then scores the top ones using the posting list.
 *
 * @lucene.experimental
 */
public class DefaultIVFVectorsReader extends IVFVectorsReader {

  public DefaultIVFVectorsReader(SegmentReadState state, FlatVectorsReader rawVectorsReader)
      throws IOException {
    super(state, rawVectorsReader);
  }

  @Override
  protected PostingListScorer readPostingListScorer(FieldInfo info, IndexInput input)
      throws IOException {
    final int vectorsPerPostingList = input.readInt();
    final  int numCentroids = input.readInt();
    final PostingListWithFileOffset[] centroidAndOffset = new PostingListWithFileOffset[numCentroids];
    for (int i = 0; i < numCentroids; i++) {
      final long fileOffset = input.readVLong();
      final float[] vector = new float[info.getVectorDimension()];
      input.readFloats(vector, 0, vector.length);
      centroidAndOffset[i] = new PostingListWithFileOffset(vector, fileOffset);
    }
    return (fieldInfo, knnCollector, target) -> {
      // TODO: improve the heuristic here. It does not work they there are many deleted documents or
      // restrictive filter.
      final int postingListsToScore = Math.min(numCentroids, (knnCollector.k() * 100) / vectorsPerPostingList);
      final PriorityQueue<PostingListWithFileOffsetWithScore> pq =
          new PriorityQueue<>(postingListsToScore) {
            @Override
            protected boolean lessThan(PostingListWithFileOffsetWithScore a, PostingListWithFileOffsetWithScore b) {
              return a.score() < b.score();
            }
          };
      for (PostingListWithFileOffset PostingListWithFileOffset : centroidAndOffset) {
        float score =
            fieldInfo.getVectorSimilarityFunction().compare(PostingListWithFileOffset.centroid(), target);
        pq.add(new PostingListWithFileOffsetWithScore(PostingListWithFileOffset, score));
      }

      final PostingListWithFileOffsetWithScore[] topCentroids = new PostingListWithFileOffsetWithScore[postingListsToScore];
      for (int i = 1; i <= postingListsToScore; i++) {
        topCentroids[postingListsToScore - i] = pq.pop();
      }
      return Arrays.stream(topCentroids).iterator();
    };
  }

  @Override
  protected boolean scorePostingList(
      FieldInfo fieldInfo,
      IndexInput input,
      float[] target,
      KnnCollector knnCollector,
      Bits acceptDocs,
      long fileOffset,
      float[] scratch)
      throws IOException {
    input.seek(fileOffset);
    int vectors = input.readVInt();
    for (int i = 0; i < vectors; i++) {
      final int docId = input.readInt();
      if (acceptDocs != null && acceptDocs.get(docId) == false) {
        input.skipBytes(Float.BYTES * fieldInfo.getVectorDimension());
        continue;
      }
      knnCollector.incVisitedCount(1);
      input.readFloats(scratch, 0, scratch.length);
      knnCollector.collect(docId, fieldInfo.getVectorSimilarityFunction().compare(scratch, target));
      if (knnCollector.earlyTerminated()) {
        return true;
      }
    }
    return false;
  }
}
