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

import org.apache.lucene.index.PointValues;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.DataOutput;

final class BKDForUtil {

  private final int[] tmp;

  BKDForUtil(int maxPointsInLeaf) {
    tmp = new int[maxPointsInLeaf * 3 / 4];
  }

  void encode16(int len, int[] ints, DataOutput out) throws IOException {
    final int halfLen = len >>> 1;
    for (int i = 0; i < halfLen; ++i) {
      ints[i] = ints[halfLen + i] | (ints[i] << 16);
    }
    for (int i = 0; i < halfLen; i++) {
      out.writeInt(ints[i]);
    }
    if ((len & 1) == 1) {
      out.writeShort((short) ints[len - 1]);
    }
  }

  void encode32(int off, int len, int[] ints, DataOutput out) throws IOException {
    for (int i = 0; i < len; i++) {
      out.writeInt(ints[off + i]);
    }
  }

  void encode24(int off, int len, int[] ints, DataOutput out) throws IOException {
    final int quarterLen = len >>> 2;
    final int quarterLen3 = quarterLen * 3;
    for (int i = 0; i < quarterLen3; ++i) {
      tmp[i] = ints[off + i] << 8;
    }
    for (int i = 0; i < quarterLen; i++) {
      final int longIdx = off + i + quarterLen3;
      tmp[i] |= (ints[longIdx] >>> 16) & 0xFF;
      tmp[i + quarterLen] |= (ints[longIdx] >>> 8) & 0xFF;
      tmp[i + quarterLen * 2] |= ints[longIdx] & 0xFF;
    }
    for (int i = 0; i < quarterLen3; ++i) {
      out.writeInt(tmp[i]);
    }

    final int remainder = len & 0x3;
    for (int i = 0; i < remainder; i++) {
      out.writeInt(ints[quarterLen * 4 + i]);
    }
  }

  void decode16(DataInput in, int[] ints, int len, final int base) throws IOException {
    final int halfLen = len >>> 1;
    in.readInts(ints, 0, halfLen);
    for (int i = 0; i < halfLen; ++i) {
      int l = ints[i];
      ints[i] = (l >>> 16) + base;
      ints[halfLen + i] = (l & 0xFFFF) + base;
    }
    if ((len & 1) == 1) {
      ints[len - 1] = Short.toUnsignedInt(in.readShort()) + base;
    }
  }

  void decode24(DataInput in, int[] ints, int len) throws IOException {
    final int quarterLen = len >>> 2;
    final int quarterLen3 = quarterLen * 3;
    in.readInts(tmp, 0, quarterLen3);
    for (int i = 0; i < quarterLen3; ++i) {
      ints[i] = tmp[i] >>> 8;
    }
    for (int i = 0; i < quarterLen; i++) {
      ints[i + quarterLen3] =
              ((tmp[i] << 16) & 0xFFFFFF)
                      | ((tmp[i + quarterLen] << 8) & 0xFFFF)
                      | (tmp[i + quarterLen * 2] & 0xFF);
    }
    in.readInts(ints, quarterLen << 2, len & 0x3);
  }
}