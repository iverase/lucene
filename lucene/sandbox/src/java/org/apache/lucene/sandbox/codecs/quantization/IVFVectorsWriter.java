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
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;

/**
 * Base codec class for writing Inverted File Vector indexes. The
 * posting lists are created using the method {@link #buildPostingLists(FieldInfo, FloatVectorValues)}.
 * <p>
 * Each posting list is then stored in the index file using the method
 *  {@link #writePostingLists(IndexOutput, FieldInfo, FloatVectorValues, PostingLists)} and the centroids
 *  with an offset to the index files are stored in the meta file using
 *  {@link #writeCentroidsAndOffsets(IndexOutput, FieldInfo, PostingListWithFileOffset[])}.
 *  <p>
 *  It is only implemented vor the {@link VectorEncoding#FLOAT32} encoding. For the {@link VectorEncoding#BYTE}
 *  it only stores the vectors.
 *
 * @lucene.experimental
 */
public abstract class IVFVectorsWriter extends KnnVectorsWriter {

  private final List<FieldWriter> fieldWriters = new ArrayList<>();
  private final IndexOutput ivfIndex;
  private final IndexOutput ivfMeta;
  private final FlatVectorsWriter rawVectorDelegate;

  protected IVFVectorsWriter(SegmentWriteState state, FlatVectorsWriter rawVectorDelegate)
      throws IOException {
    this.rawVectorDelegate = rawVectorDelegate;
    final String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, IVFVectorsFormat.IVF_META_EXTENSION);

    final String ivfFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, IVFVectorsFormat.IVF_INDEX_EXTENSION);
    boolean success = false;
    try {
      ivfIndex = state.directory.createOutput(ivfFileName, state.context);
      CodecUtil.writeIndexHeader(
          ivfIndex,
          IVFVectorsFormat.NAME,
          IVFVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      ivfMeta = state.directory.createOutput(metaFileName, state.context);
      CodecUtil.writeIndexHeader(
          ivfMeta,
          IVFVectorsFormat.NAME,
          IVFVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(this);
      }
    }
  }

  /** Partition the dimensional space generating a list of posting lists represented by a centroid.*/
  protected abstract PostingLists buildPostingLists(
      FieldInfo fieldInfo, FloatVectorValues floatVectorValues) throws IOException;

  /** Write the posting list into the index file. These posting lists will be read using
   * {@link IVFVectorsReader#scorePostingList(FieldInfo, IndexInput, float[], KnnCollector, Bits, long, float[])}. */
  protected abstract PostingListWithFileOffset[] writePostingLists(
          IndexOutput out, FieldInfo fieldInfo, FloatVectorValues floatVectorValues, PostingLists postingLists)
          throws IOException;

  /**
   * Write the centroids and the offsets to the meta file. The centroids will be read using the method
   * {@link IVFVectorsReader#readPostingListScorer(FieldInfo, IndexInput)}.
   */
  protected abstract void writeCentroidsAndOffsets(
      IndexOutput out, FieldInfo fieldInfo, PostingListWithFileOffset[] postingListWithFileOffsets)
      throws IOException;



  @Override
  public final KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    final FlatFieldVectorsWriter<?> rawVectorDelegate = this.rawVectorDelegate.addField(fieldInfo);
    if (fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32)) {
      @SuppressWarnings("unchecked")
      final FlatFieldVectorsWriter<float[]> floatWriter =
          (FlatFieldVectorsWriter<float[]>) rawVectorDelegate;
      fieldWriters.add(new FieldWriter(fieldInfo, floatWriter));
    }
    return rawVectorDelegate;
  }

  @Override
  public final void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    rawVectorDelegate.flush(maxDoc, sortMap);
    for (FieldWriter fieldWriter : fieldWriters) {
      // build a float vector values with random access
      final FloatVectorValues floatVectorValues =
          getFloatVectorValues(fieldWriter.fieldInfo, fieldWriter.delegate, maxDoc);
      // build posting lists
      final PostingLists postingLists = buildPostingLists(fieldWriter.fieldInfo, floatVectorValues);
      // write posting lists
      if (postingLists.size() > 0) {
        final PostingListWithFileOffset[] postingListWithFileOffsets =
            writePostingLists(ivfIndex, fieldWriter.fieldInfo, floatVectorValues, postingLists);
        writeMeta(fieldWriter.fieldInfo, postingListWithFileOffsets);
      } else {
        writeMeta(fieldWriter.fieldInfo, new PostingListWithFileOffset[0]);
      }
    }
  }

  private static FloatVectorValues getFloatVectorValues(
      FieldInfo fieldInfo, FlatFieldVectorsWriter<float[]> fieldVectorsWriter, int maxDoc)
      throws IOException {
    List<float[]> vectors = fieldVectorsWriter.getVectors();
    if (vectors.size() == maxDoc) {
      return FloatVectorValues.fromFloats(vectors, fieldInfo.getVectorDimension());
    }
    final DocIdSetIterator iterator = fieldVectorsWriter.getDocsWithFieldSet().iterator();
    final int[] docIds = new int[vectors.size()];
    for (int i = 0; i < docIds.length; i++) {
      docIds[i] = iterator.nextDoc();
    }
    assert iterator.nextDoc() == NO_MORE_DOCS;
    return new FloatVectorValues() {
      @Override
      public float[] vectorValue(int ord) {
        return vectors.get(ord);
      }

      @Override
      public FloatVectorValues copy() {
        return this;
      }

      @Override
      public int dimension() {
        return fieldInfo.getVectorDimension();
      }

      @Override
      public int size() {
        return vectors.size();
      }

      @Override
      public int ordToDoc(int ord) {
        return docIds[ord];
      }
    };
  }

  @Override
  public final void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    if (fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32)) {
      final int numVectors;
      String name = null;
      boolean success = false;
      // build a float vector values with random access. In order to do that we dump the vectors to
      // a temporary file
      // and write the docID follow by the vector
      try (IndexOutput out =
          mergeState.segmentInfo.dir.createTempOutput(
              mergeState.segmentInfo.name, "ivf_", IOContext.DEFAULT)) {
        name = out.getName();
        numVectors =
            writeFloatVectorValues(
                fieldInfo, out, MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState));
        success = true;
      } finally {
        if (success == false && name != null) {
          IOUtils.deleteFilesIgnoringExceptions(mergeState.segmentInfo.dir, name);
        }
      }
      try (IndexInput in = mergeState.segmentInfo.dir.openInput(name, IOContext.DEFAULT)) {
        final FloatVectorValues floatVectorValues =
            getFloatVectorValues(fieldInfo, in.randomAccessSlice(0, in.length()), numVectors);
        final PostingLists postingLists = buildPostingLists(fieldInfo, floatVectorValues);
        if (postingLists.size() > 0) {
          final PostingListWithFileOffset[] postingListWithFileOffsets =
              writePostingLists(ivfIndex, fieldInfo, floatVectorValues, postingLists);
          writeMeta(fieldInfo, postingListWithFileOffsets);
        } else {
          writeMeta(fieldInfo, new PostingListWithFileOffset[0]);
        }
      } finally {
        IOUtils.deleteFilesIgnoringExceptions(mergeState.segmentInfo.dir, name);
      }
    }
    rawVectorDelegate.mergeOneField(fieldInfo, mergeState);
  }

  private static FloatVectorValues getFloatVectorValues(
      FieldInfo fieldInfo, RandomAccessInput randomAccessInput, int numVectors) {
    final long length = (long) Float.BYTES * fieldInfo.getVectorDimension() + Integer.BYTES;
    final float[] vector = new float[fieldInfo.getVectorDimension()];
    return new FloatVectorValues() {
      @Override
      public float[] vectorValue(int ord) throws IOException {
        for (int i = 0; i < vector.length; i++) {
          vector[i] =
              Float.intBitsToFloat(randomAccessInput.readInt(ord * length + i * Float.BYTES));
        }
        return vector;
      }

      @Override
      public FloatVectorValues copy() {
        return this;
      }

      @Override
      public int dimension() {
        return fieldInfo.getVectorDimension();
      }

      @Override
      public int size() {
        return numVectors;
      }

      @Override
      public int ordToDoc(int ord) {
        try {
          return randomAccessInput.readInt(ord * length);
        } catch (IOException e) {
          throw new UncheckedIOException(e);
        }
      }
    };
  }

  private static int writeFloatVectorValues(
      FieldInfo fieldInfo, IndexOutput out, FloatVectorValues floatVectorValues)
      throws IOException {
    int numVectors = 0;
    final ByteBuffer buffer =
        ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    final KnnVectorValues.DocIndexIterator iterator = floatVectorValues.iterator();
    for (int docV = iterator.nextDoc(); docV != NO_MORE_DOCS; docV = iterator.nextDoc()) {
      numVectors++;
      float[] vector = floatVectorValues.vectorValue(iterator.index());
      out.writeInt(iterator.docID());
      buffer.asFloatBuffer().put(vector);
      out.writeBytes(buffer.array(), buffer.array().length);
    }
    return numVectors;
  }

  private void writeMeta(FieldInfo field, PostingListWithFileOffset[] postingListWithFileOffsets)
      throws IOException {
    ivfMeta.writeInt(field.number);
    ivfMeta.writeInt(field.getVectorEncoding().ordinal());
    ivfMeta.writeInt(distFuncToOrd(field.getVectorSimilarityFunction()));
    writeCentroidsAndOffsets(ivfMeta, field, postingListWithFileOffsets);
  }

  private static int distFuncToOrd(VectorSimilarityFunction func) {
    for (int i = 0; i < SIMILARITY_FUNCTIONS.size(); i++) {
      if (SIMILARITY_FUNCTIONS.get(i).equals(func)) {
        return (byte) i;
      }
    }
    throw new IllegalArgumentException("invalid distance function: " + func);
  }

  @Override
  public final void finish() throws IOException {
    rawVectorDelegate.finish();
    if (ivfIndex != null) {
      CodecUtil.writeFooter(ivfIndex);
    }
    if (ivfMeta != null) {
      // write end of fields marker
      ivfMeta.writeInt(-1);
      CodecUtil.writeFooter(ivfMeta);
    }
  }

  @Override
  public final void close() throws IOException {
    IOUtils.close(rawVectorDelegate, ivfIndex, ivfMeta);
  }

  @Override
  public final long ramBytesUsed() {
    return rawVectorDelegate.ramBytesUsed();
  }

  private record FieldWriter(FieldInfo fieldInfo, FlatFieldVectorsWriter<float[]> delegate) {}

  /** A record containing the centroid and the index offset for a posting list */
  protected record PostingListWithFileOffset(float[] centroid, long fileOffset) {}

  /** A posting list iterator with a known size */
  protected interface PostingLists extends Iterator<PostingList> {
    /** Return the number of posting lists */
    int size();
  }

  /** A posting list with the vector s it contains. It provides the ords that can be used
   * against the provided {@link FloatVectorValues}  */
  protected interface PostingList {
    /** Return the number of vectors in the posting list */
    int size();
    /** The centroid of the posting list */
    float[] centroid();
    /** If there is next vector in the posting list */
    boolean hasNext();
    /** Return the ord of the next vector in the posting list */
    int nextOrd();
  }
}
