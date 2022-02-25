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

package org.apache.lucene.search;

import java.io.IOException;
import java.util.Objects;
import org.apache.lucene.index.LeafReaderContext;

/**
 * A {@link Query} that matches documents that contain a {@link
 * org.apache.lucene.document.KnnVectorField}.
 */
public class KnnVectorFieldExistsQuery extends Query {

  private final String field;

  /** Create a query that will match documents which have a value for the given {@code field}. */
  public KnnVectorFieldExistsQuery(String field) {
    this.field = Objects.requireNonNull(field);
  }

  public String getField() {
    return field;
  }

  @Override
  public boolean equals(Object other) {
    return sameClassAs(other) && field.equals(((KnnVectorFieldExistsQuery) other).field);
  }

  @Override
  public int hashCode() {
    return 31 * classHash() + field.hashCode();
  }

  @Override
  public String toString(String field) {
    return "KnnVectorFieldExistsQuery [field=" + this.field + "]";
  }

  @Override
  public void visit(QueryVisitor visitor) {
    if (visitor.acceptField(field)) {
      visitor.visitLeaf(this);
    }
  }

  @Override
  public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {
    return new ConstantScoreWeight(this, boost) {
      @Override
      public Scorer scorer(LeafReaderContext context) throws IOException {
        DocIdSetIterator iterator = context.reader().getVectorValues(field);
        if (iterator == null) {
          return null;
        }
        return new ConstantScoreScorer(this, score(), scoreMode, iterator);
      }

      @Override
      public boolean isCacheable(LeafReaderContext ctx) {
        return true;
      }
    };
  }
}
