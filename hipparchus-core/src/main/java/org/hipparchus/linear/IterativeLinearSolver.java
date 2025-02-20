/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * This is not the original file distributed by the Apache Software Foundation
 * It has been modified by the Hipparchus project
 */
package org.hipparchus.linear;

import org.hipparchus.exception.LocalizedCoreFormats;
import org.hipparchus.exception.MathIllegalArgumentException;
import org.hipparchus.exception.MathIllegalStateException;
import org.hipparchus.exception.NullArgumentException;
import org.hipparchus.util.IterationManager;
import org.hipparchus.util.MathUtils;

/**
 * This abstract class defines an iterative solver for the linear system A
 * &middot; x = b. In what follows, the <em>residual</em> r is defined as r = b
 * - A &middot; x, where A is the linear operator of the linear system, b is the
 * right-hand side vector, and x the current estimate of the solution.
 *
 */
public abstract class IterativeLinearSolver {

    /** The object in charge of managing the iterations. */
    private final IterationManager manager;

    /**
     * Creates a new instance of this class, with default iteration manager.
     *
     * @param maxIterations the maximum number of iterations
     */
    protected IterativeLinearSolver(final int maxIterations) {
        this.manager = new IterationManager(maxIterations);
    }

    /**
     * Creates a new instance of this class, with custom iteration manager.
     *
     * @param manager the custom iteration manager
     * @throws NullArgumentException if {@code manager} is {@code null}
     */
    protected IterativeLinearSolver(final IterationManager manager)
        throws NullArgumentException {
        MathUtils.checkNotNull(manager);
        this.manager = manager;
    }

    /**
     * Performs all dimension checks on the parameters of
     * {@link #solve(RealLinearOperator, RealVector, RealVector) solve} and
     * {@link #solveInPlace(RealLinearOperator, RealVector, RealVector) solveInPlace},
     * and throws an exception if one of the checks fails.
     *
     * @param a the linear operator A of the system
     * @param b the right-hand side vector
     * @param x0 the initial guess of the solution
     * @throws NullArgumentException if one of the parameters is {@code null}
     * @throws MathIllegalArgumentException if {@code a} is not square
     * @throws MathIllegalArgumentException if {@code b} or {@code x0} have
     * dimensions inconsistent with {@code a}
     */
    protected static void checkParameters(final RealLinearOperator a,
        final RealVector b, final RealVector x0) throws
        MathIllegalArgumentException, NullArgumentException {
        MathUtils.checkNotNull(a);
        MathUtils.checkNotNull(b);
        MathUtils.checkNotNull(x0);
        if (a.getRowDimension() != a.getColumnDimension()) {
            throw new MathIllegalArgumentException(LocalizedCoreFormats.NON_SQUARE_OPERATOR,
                                                   a.getRowDimension(), a.getColumnDimension());
        }
        if (b.getDimension() != a.getRowDimension()) {
            throw new MathIllegalArgumentException(LocalizedCoreFormats.DIMENSIONS_MISMATCH,
                                                   b.getDimension(), a.getRowDimension());
        }
        if (x0.getDimension() != a.getColumnDimension()) {
            throw new MathIllegalArgumentException(LocalizedCoreFormats.DIMENSIONS_MISMATCH,
                                                   x0.getDimension(), a.getColumnDimension());
        }
    }

    /**
     * Returns the iteration manager attached to this solver.
     *
     * @return the manager
     */
    public IterationManager getIterationManager() {
        return manager;
    }

    /**
     * Returns an estimate of the solution to the linear system A &middot; x =
     * b.
     *
     * @param a the linear operator A of the system
     * @param b the right-hand side vector
     * @return a new vector containing the solution
     * @throws NullArgumentException if one of the parameters is {@code null}
     * @throws MathIllegalArgumentException if {@code a} is not square
     * @throws MathIllegalArgumentException if {@code b} has dimensions
     * inconsistent with {@code a}
     * @throws MathIllegalStateException at exhaustion of the iteration count,
     * unless a custom
     * {@link org.hipparchus.util.Incrementor.MaxCountExceededCallback callback}
     * has been set at construction of the {@link IterationManager}
     */
    public RealVector solve(final RealLinearOperator a, final RealVector b)
        throws MathIllegalArgumentException, NullArgumentException, MathIllegalStateException {
        MathUtils.checkNotNull(a);
        final RealVector x = new ArrayRealVector(a.getColumnDimension());
        x.set(0.);
        return solveInPlace(a, b, x);
    }

    /**
     * Returns an estimate of the solution to the linear system A &middot; x =
     * b.
     *
     * @param a the linear operator A of the system
     * @param b the right-hand side vector
     * @param x0 the initial guess of the solution
     * @return a new vector containing the solution
     * @throws NullArgumentException if one of the parameters is {@code null}
     * @throws MathIllegalArgumentException if {@code a} is not square
     * @throws MathIllegalArgumentException if {@code b} or {@code x0} have
     * dimensions inconsistent with {@code a}
     * @throws MathIllegalStateException at exhaustion of the iteration count,
     * unless a custom
     * {@link org.hipparchus.util.Incrementor.MaxCountExceededCallback callback}
     * has been set at construction of the {@link IterationManager}
     */
    public RealVector solve(RealLinearOperator a, RealVector b, RealVector x0)
        throws MathIllegalArgumentException, NullArgumentException, MathIllegalStateException {
        MathUtils.checkNotNull(x0);
        return solveInPlace(a, b, x0.copy());
    }

    /**
     * Returns an estimate of the solution to the linear system A &middot; x =
     * b. The solution is computed in-place (initial guess is modified).
     *
     * @param a the linear operator A of the system
     * @param b the right-hand side vector
     * @param x0 initial guess of the solution
     * @return a reference to {@code x0} (shallow copy) updated with the
     * solution
     * @throws NullArgumentException if one of the parameters is {@code null}
     * @throws MathIllegalArgumentException if {@code a} is not square
     * @throws MathIllegalArgumentException if {@code b} or {@code x0} have
     * dimensions inconsistent with {@code a}
     * @throws MathIllegalStateException at exhaustion of the iteration count,
     * unless a custom
     * {@link org.hipparchus.util.Incrementor.MaxCountExceededCallback callback}
     * has been set at construction of the {@link IterationManager}
     */
    public abstract RealVector solveInPlace(RealLinearOperator a, RealVector b,
        RealVector x0) throws MathIllegalArgumentException, NullArgumentException, MathIllegalStateException;
}
