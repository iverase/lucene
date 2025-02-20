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

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.SIMILARITY_FUNCTIONS;

import java.io.IOException;
import java.util.Iterator;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.ReadAdvice;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;


/**
 * Base codec class for reading Inverted File Vector indexes. The
 * centroids and offsets lists are read using the method {@link #readPostingListScorer(FieldInfo, IndexInput)}. The
 * generated {@link PostingListScorer} is then used to score the centroids and return the top scoring ones.
 * <p>
 * THe posting list of vectors is then read using the method
 * {@link #scorePostingList(FieldInfo, IndexInput, float[], KnnCollector, Bits, long, float[])}.
 *  <p>
 *  It is only implemented vor the {@link VectorEncoding#FLOAT32} encoding. For the {@link VectorEncoding#BYTE}
 *  it uses brute force for scoring the vectors.
 *
 * @lucene.experimental
 */
public abstract class IVFVectorsReader extends KnnVectorsReader {

  private final IndexInput ivfIndex;
  private final SegmentReadState state;
  private final FieldInfos fieldInfos;
  private final IntObjectHashMap<FieldEntry> fields;
  private final FlatVectorsReader rawVectorsReader;

  protected IVFVectorsReader(SegmentReadState state, FlatVectorsReader rawVectorsReader)
      throws IOException {
    this.state = state;
    this.fieldInfos = state.fieldInfos;
    this.rawVectorsReader = rawVectorsReader;
    this.fields = new IntObjectHashMap<>();
    String meta =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, IVFVectorsFormat.IVF_META_EXTENSION);

    int versionMeta = -1;
    boolean success = false;
    try (ChecksumIndexInput ivfMeta = state.directory.openChecksumInput(meta)) {
      Throwable priorE = null;
      try {
        versionMeta =
            CodecUtil.checkIndexHeader(
                ivfMeta,
                IVFVectorsFormat.NAME,
                IVFVectorsFormat.VERSION_START,
                IVFVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix);
        readFields(ivfMeta);
      } catch (Throwable exception) {
        priorE = exception;
      } finally {
        CodecUtil.checkFooter(ivfMeta, priorE);
      }
      ivfIndex =
          openDataInput(
              state,
              versionMeta,
              IVFVectorsFormat.IVF_INDEX_EXTENSION,
              IVFVectorsFormat.NAME,
              //  we need to jump between posting lists
              state.context.withReadAdvice(ReadAdvice.RANDOM));
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  /**
   * Reads the generated posting lists metadata in {@link IVFVectorsWriter#writeCentroidsAndOffsets(IndexOutput, FieldInfo, IVFVectorsWriter.PostingListWithFileOffset[])}
   *  and creates a {@link PostingListScorer}
   */
  protected abstract PostingListScorer readPostingListScorer(FieldInfo info, IndexInput input)
      throws IOException;

  /**
   * Scores the posting list of vectors generated in
   * {@link IVFVectorsWriter#writePostingLists(IndexOutput, FieldInfo, FloatVectorValues, IVFVectorsWriter.PostingLists)}
   * for the given target vector. The method should return true if there is no more scoring required.
   */
  protected abstract boolean scorePostingList(
      FieldInfo fieldInfo,
      IndexInput input,
      float[] target,
      KnnCollector knnCollector,
      Bits acceptDocs,
      long fileOffset,
      float[] scratch)
      throws IOException;

  private static IndexInput openDataInput(
      SegmentReadState state,
      int versionMeta,
      String fileExtension,
      String codecName,
      IOContext context)
      throws IOException {
    final String fileName =
        IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, fileExtension);
    final IndexInput in = state.directory.openInput(fileName, context);
    boolean success = false;
    try {
      final int versionVectorData =
          CodecUtil.checkIndexHeader(
              in,
              codecName,
              IVFVectorsFormat.VERSION_START,
              IVFVectorsFormat.VERSION_CURRENT,
              state.segmentInfo.getId(),
              state.segmentSuffix);
      if (versionMeta != versionVectorData) {
        throw new CorruptIndexException(
            "Format versions mismatch: meta="
                + versionMeta
                + ", "
                + codecName
                + "="
                + versionVectorData,
            in);
      }
      CodecUtil.retrieveChecksum(in);
      success = true;
      return in;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(in);
      }
    }
  }

  private void readFields(ChecksumIndexInput meta) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      final FieldInfo info = fieldInfos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }
      fields.put(info.number, readField(meta, info));
    }
  }

  private FieldEntry readField(IndexInput input, FieldInfo info) throws IOException {
    final VectorEncoding vectorEncoding = readVectorEncoding(input);
    final VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
    if (similarityFunction != info.getVectorSimilarityFunction()) {
      throw new IllegalStateException(
          "Inconsistent vector similarity function for field=\""
              + info.name
              + "\"; "
              + similarityFunction
              + " != "
              + info.getVectorSimilarityFunction());
    }
    return new FieldEntry(similarityFunction, vectorEncoding, readPostingListScorer(info, input));
  }

  private static VectorSimilarityFunction readSimilarityFunction(DataInput input)
      throws IOException {
    final int i = input.readInt();
    if (i < 0 || i >= SIMILARITY_FUNCTIONS.size()) {
      throw new IllegalArgumentException("invalid distance function: " + i);
    }
    return SIMILARITY_FUNCTIONS.get(i);
  }

  private static VectorEncoding readVectorEncoding(DataInput input) throws IOException {
    final  int encodingId = input.readInt();
    if (encodingId < 0 || encodingId >= VectorEncoding.values().length) {
      throw new CorruptIndexException("Invalid vector encoding id: " + encodingId, input);
    }
    return VectorEncoding.values()[encodingId];
  }

  @Override
  public final void checkIntegrity() throws IOException {
    rawVectorsReader.checkIntegrity();
    CodecUtil.checksumEntireFile(ivfIndex);
  }

  @Override
  public final FloatVectorValues getFloatVectorValues(String field) throws IOException {
    return rawVectorsReader.getFloatVectorValues(field);
  }

  @Override
  public final ByteVectorValues getByteVectorValues(String field) throws IOException {
    return rawVectorsReader.getByteVectorValues(field);
  }

  @Override
  public final void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    final FieldInfo fieldInfo = state.fieldInfos.fieldInfo(field);
    if (fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32) == false) {
      rawVectorsReader.search(field, target, knnCollector, acceptDocs);
      return;
    }
    final FieldEntry fieldEntry = fields.get(fieldInfo.number);
    final Iterator<PostingListWithFileOffsetWithScore> topPostingLists =
        fieldEntry.postingListScorer.scorePostingList(fieldInfo, knnCollector, target);
    final float[] vector = new float[fieldInfo.getVectorDimension()];
    while (topPostingLists.hasNext()) {
      final PostingListWithFileOffsetWithScore next = topPostingLists.next();
      if (next.score() < knnCollector.minCompetitiveSimilarity()
          || scorePostingList(
              fieldInfo, ivfIndex, target, knnCollector, acceptDocs, next.postingListWithFileOffset().fileOffset(), vector)) {
        return;
      }
    }
  }

  @Override
  public final void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs)
      throws IOException {
    final FieldInfo fieldInfo = state.fieldInfos.fieldInfo(field);
    final ByteVectorValues values = rawVectorsReader.getByteVectorValues(field);
    for (int i = 0; i < values.size(); i++) {
      final float score =
          fieldInfo.getVectorSimilarityFunction().compare(target, values.vectorValue(i));
      knnCollector.collect(values.ordToDoc(i), score);
      if (knnCollector.earlyTerminated()) {
        return;
      }
    }
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(ivfIndex, rawVectorsReader);
  }

  private record FieldEntry(
      VectorSimilarityFunction similarityFunction,
      VectorEncoding vectorEncoding,
      PostingListScorer postingListScorer) {}

  /**
   * Scorer the posting list centroid against a target vector. The scorer should return the top scoring
   * posting lists.
   */
  protected interface PostingListScorer {
    /** Score the centroid of the posting lists against the target vector */
    Iterator<PostingListWithFileOffsetWithScore> scorePostingList(
        FieldInfo fieldInfo, KnnCollector knnCollector, float[] target) throws IOException;
  }

  /** A record containing the centroid and the index offset for a posting list with the given score */
  protected record PostingListWithFileOffsetWithScore(PostingListWithFileOffset postingListWithFileOffset, float score) {}

  /** A record containing the centroid and the index offset for a posting list */
  protected record PostingListWithFileOffset(float[] centroid, long fileOffset) {}


}
