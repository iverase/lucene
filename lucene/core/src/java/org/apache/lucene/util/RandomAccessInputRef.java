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
import org.apache.lucene.store.RandomAccessInput;

/**
 * Represents a RandomAccessInput, as a slice (offset + length) into an existing RandomAccessInput.
 * The {@link #bytes} member should never be null. In many ways, it is the off-heap equivalent of a
 * {@link BytesRef}.
 *
 * @see BytesRef
 */
public final class RandomAccessInputRef {

  /** The contents of the RandomAccessInput. */
  public RandomAccessInput bytes;

  /** Offset of first valid byte. */
  public long offset;

  /** Length of used bytes. */
  public int length;

  public RandomAccessInputRef(RandomAccessInput bytes) {
    this(bytes, 0, 0);
  }

  public RandomAccessInputRef(RandomAccessInput bytes, long offset, int length) {
    this.bytes = bytes;
    this.offset = offset;
    this.length = length;
  }

  /**
   * Interprets stored bytes as UTF-8 bytes, returning the resulting string. May throw an {@link
   * AssertionError} or a {@link RuntimeException} if the data is not well-formed UTF-8.
   */
  public String utf8ToString() throws IOException {
    final char[] ref = new char[length];
    final int len = UnicodeUtil.UTF8toUTF16(bytes, offset, length, ref);
    return new String(ref, 0, len);
  }

  /**
   * Creates a new BytesRef that points to a copy of the bytes from <code>input</code> starting at
   * offset for length.
   *
   * <p>The returned BytesRef will have a offset of zero.
   */
  public static BytesRef toBytesRef(RandomAccessInputRef input) throws IOException {
    final byte[] bytes = new byte[input.length];
    input.bytes.readBytes(input.offset, bytes, 0, input.length);
    return new BytesRef(bytes, 0, input.length);
  }

  /** Checks the validity of the RandomAccessInputRef. */
  public boolean isValid() throws IOException {
    if (bytes == null) {
      throw new IllegalStateException("bytes is null");
    }
    if (length < 0) {
      throw new IllegalStateException("length is negative: " + length);
    }
    if (length != 0 && length > bytes.length()) {
      throw new IllegalStateException(
          "length is out of bounds: " + length + ",bytes.length=" + bytes.length());
    }
    if (offset < 0) {
      throw new IllegalStateException("offset is negative: " + offset);
    }
    if (offset > bytes.length()) {
      throw new IllegalStateException(
          "offset out of bounds: " + offset + ",bytes.length=" + bytes.length());
    }
    if (offset + length < 0) {
      throw new IllegalStateException(
          "offset+length is negative: offset=" + offset + ",length=" + length);
    }
    if (offset + length > bytes.length()) {
      throw new IllegalStateException(
          "offset+length out of bounds: offset="
              + offset
              + ",length="
              + length
              + ",bytes.length="
              + bytes.length());
    }
    return true;
  }
}
