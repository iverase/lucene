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
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99ScalarQuantizedVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;

/**
 * Codec format for Inverted File Vector indexes. This index expects to break the dimensional space into clusters and
 * assign each vector to a cluster generating a posting list of vectors. Clusters are represented by centroids.
 *<p>
 * THe index is searcher by looking for the closest centroids to our vector query and then scoring the vectors in the
 * posting list of the closest centroids.
 *
 * @lucene.experimental
 */
public class IVFVectorsFormat extends KnnVectorsFormat {

  public static final String NAME = "IVFVectorsFormat";
  static final String IVF_META_EXTENSION = "mivf";
  static final String IVF_INDEX_EXTENSION = "ivf";

  public static final int VERSION_START = 0;
  public static final int VERSION_CURRENT = VERSION_START;

  private static final FlatVectorsFormat rawVectorFormat =
      new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());

  private static final int DEFAULT_VECTORS_PER_CLUSTER = 1000;

  private final int vectorPerCluster;

  public IVFVectorsFormat(int vectorPerCluster) {
    super(NAME);
    this.vectorPerCluster = vectorPerCluster;
  }

  /** Constructs a format using the given graph construction parameters and scalar quantization. */
  public IVFVectorsFormat() {
    this(DEFAULT_VECTORS_PER_CLUSTER);
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new DefaultIVFVectorsWriter(
        state, rawVectorFormat.fieldsWriter(state), vectorPerCluster);
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new DefaultIVFVectorsReader(state, rawVectorFormat.fieldsReader(state));
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return 1024;
  }

  @Override
  public String toString() {
    return "IVFVectorFormat";
  }
}
