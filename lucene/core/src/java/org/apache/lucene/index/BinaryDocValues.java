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

package org.apache.lucene.index;

import java.io.IOException;
import org.apache.lucene.util.RandomAccessInputRef;

/** A per-document binary value. */
public abstract class BinaryDocValues extends DocValuesIterator {

  /** Sole constructor. (For invocation by subclass constructors, typically implicit.) */
  protected BinaryDocValues() {}

  /**
   * Returns the binary value for the current document ID. The returned instance might be reused
   * across calls, therefore the result should be fully consumed before moving to the next doc. It
   * is illegal to call this method after {@link #advanceExact(int)} returned {@code false}.
   *
   * @return the binary value as a {@link RandomAccessInputRef}
   */
  public abstract RandomAccessInputRef randomAccessInputValue() throws IOException;
}
