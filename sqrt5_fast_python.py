#########################################################################
#       Copyright (C) 2010-2012 William Stein <wstein@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#
#                  http://www.gnu.org/licenses/
#########################################################################

# This file contains code that is really part of sqrt5_fast.pyx, but
# does not need to be in Cython.

#######################################################################################
# Compute local splitting at 2 by successive lifting argument.
#######################################################################################
"""
This is on pages 9-10 of my research notes on 2010-11-07.

The mod-2 splitting must be done differently because I,J do *not*
generate the icosian ring over O_F, because of the 2's in the
denominator.

The icosian ring has generators given above, call them w0,w1,w2,w3.  Using linear algebra
we find that w0^2=w0-1, w1^2=-1, w2^2=-1, w3^2=-1.  Moreover, and most importantly, we have
letting g=(1+sqrt(5))/2, that 

      w0 = (g-1)*(2*g*w3 - w1 - w2 - w1*w2)
      w3 = g*w0 + w2*w1

Thus mod 2, w0 and w3 are completely determined by w1 and w2.
Also, because of the 2 coefficient of w3 in the first equation,
modulo 2^i, w0 and w3 are completely determined if we know w1, w2, and w3 mod 2^(i-1).

We have several functions below.  The one called find_mod2_splitting iterates
through all of M_2(O/2O) finding the possibilities for w1 and w2, i.e., matrices
with square minus 1.  For each, we let w0 and w3 be the corresponding matrices
given by the above formulas, verify that we get four independent matrices, and
that the minpoly conditions on w0 and w3 hold.  We thus construct a matrix
algebra Alg over F_4 satisfying all the above relations, and get a map R -->> Alg.

[[I have some worry though -- what if somehow not *all* the right relations hold?!]]

Next we lift from mod p^(i-1) to mod p^i.  This uses the following algebra.
Suppose A^2 = -1 (mod 2^(i-1)) and (A+2^(i-1)*B)^2 = -1 (mod 2^i).
Then A^2+2^(i-1)(AB+BA) + 2^(2(i-1))*B^2=-1 (mod 2^i).
Thus (-1+2^(i-1)*C) + 2^i*(AB+BA) = -1 (mod 2^i)
So we need C + AB + BA = 0 (mod 2), where C = (A^2+1)/2^i in Mat(O/2O).

To find the B's with this property, we just loop through Mat(O/2O), and test
the condition C+AB+BA in Mat(O/2O).

TODO: This implementation is slow.  This is since matrix arithmetic is
generic, and generic matrix arithmetic is double dog slow (100 times
too slow).  I think the algorithm itself is solid.
"""

from sage.matrix.all import MatrixSpace, matrix
from sage.misc.all import cached_function
from sage.rings.all import ZZ

from sqrt5_fast import ResidueRing
from sqrt5 import F, B, icosian_ring_gens

@cached_function
def find_mod2_splitting():
    """
    Return elements w0, w1, w2, w3 that determine a mod-2 splitting
    of the Hamilton quaternion algebra over Q(sqrt(5)).

    OUTPUT:

    - four 2x2 matrices over the residue field of Q(sqrt(5)) of order 4.
    
    EXAMPLES::

        sage: import sage.modular.hilbert.sqrt5_fast_python
        sage: w = sage.modular.hilbert.sqrt5_fast_python.find_mod2_splitting(); w
        (
        [       1     abar]  [       0 abar + 1]  [abar + 1     abar]  [1 1]
        [abar + 1        0], [    abar        0], [    abar abar + 1], [0 1]
        )
        sage: w[1]^2, w[2]^2
        (
        [1 0]  [1 0]
        [0 1], [0 1]
        )
        sage: span([vector(z.list()) for z in w]).dimension()
        4
    """
    P = F.primes_above(2)[0]
    k = P.residue_field()
    M = MatrixSpace(k,2)
    V = k**4
    g = k.gen() # image of golden ratio

    m1 = M(-1)
    sqrt_minus_1 = [(w, V(w.list())) for w in M if w*w == m1]
    one = M(1)
    v_one = V(one.list())
    for w1, v1 in sqrt_minus_1:
        for w2, v2  in sqrt_minus_1:
            w0 = (g-1)*(w1+w2 - w1*w2)
            w3 = g*w0 + w2*w1
            if w0*w0 != w0 - 1:
                continue
            if w3*w3 != -1:
                continue
            if V.span([w0.list(),v1,v2,w3.list()]).dimension() == 4:
                return w0, w1, w2, w3

def matrix_lift(A):
    """
    Return the matrix obtained by calling R.lift on each entry of A,
    where R is the base ring of A.

    INPUT:

    - `A` - matrix

    OUTPUT:

    - matrix
    
    EXAMPLES::

        sage: import sage.modular.hilbert.sqrt5_fast_python
        sage: B = sage.modular.hilbert.sqrt5_fast_python.matrix_lift(matrix(GF(7),2,[1..4]))
        sage: B.parent()
        Full MatrixSpace of 2 by 2 dense matrices over Integer Ring    
    """
    R = A.base_ring() 
    return matrix(A.nrows(),A.ncols(),[R.lift(x) for x in A.list()])

@cached_function
def find_mod2pow_splitting(i):
    """
    Return elements w0, w1, w2, w3 that determine a mod-`2^i` splitting
    of the Hamilton quaternion algebra over Q(sqrt(5)).

    INPUT:

    - `i` -- positive integer

    OUTPUT:

    - four 2x2 matrices over a residue ring of Q(sqrt(5)) of characteristic a power of 2
    
    EXAMPLES::

        sage: import sage.modular.hilbert.sqrt5_fast_python
        sage: w = sage.modular.hilbert.sqrt5_fast_python.find_mod2pow_splitting(1); w
        (
        [    1     g]  [    0 1 + g]  [1 + g     g]  [1 1]
        [1 + g     0], [    g     0], [    g 1 + g], [0 1]
        )
        sage: w = sage.modular.hilbert.sqrt5_fast_python.find_mod2pow_splitting(2); w
        (
        [      3 2 + 3*g]  [      0 1 + 3*g]  [3 + 3*g     3*g]
        [  1 + g       2], [      g       0], [    3*g   1 + g],
        [3 + 2*g 3 + 2*g]
        [      2 1 + 2*g]
        )
        sage: w[1]^2
        [3 0]
        [0 3]
        sage: w[2]^2
        [3 0]
        [0 3]
        sage: w = sage.modular.hilbert.sqrt5_fast_python.find_mod2pow_splitting(4)
        sage: w[1]^2
        [15  0]
        [ 0 15]
        sage: w[2]^2
        [15  0]
        [ 0 15]
        sage: w[1], w[2]
        (
        [       0 1 + 15*g]  [15 + 15*g  12 + 7*g]
        [       g        0], [ 4 + 11*g     1 + g]
        )

    TESTS:

    The power must be positive::

        sage: sage.modular.hilbert.sqrt5_fast_python.find_mod2pow_splitting(0)
        Traceback (most recent call last):
        ...
        ValueError: i must be positive    
    """
    P = F.primes_above(2)[0]
    i = ZZ(i)
    if i <= 0:
        raise ValueError, "i must be positive"
    
    if i == 1:
        R = ResidueRing(P, 1)
        M = MatrixSpace(R,2)
        return tuple([M(matrix_lift(a).list()) for a in find_mod2_splitting()])

    R = ResidueRing(P, i)
    M = MatrixSpace(R,2)
    # arbitrary lift
    wbar = [M(matrix_lift(a).list()) for a in find_mod2pow_splitting(i-1)]

    # Find lifts of wbar[1] and wbar[2] that have square -1
    k = P.residue_field()
    Mk = MatrixSpace(k, 2)
    
    t = 2**(i-1)
    s = M(t)

    L = []
    for j in [1,2]:
        C = Mk(matrix_lift(wbar[j]**2 + M(1)) / t)
        A = Mk(matrix_lift(wbar[j]))
        # Find all matrices B in Mk such that AB+BA=C.
        L.append([wbar[j]+s*M(matrix_lift(B)) for B in Mk if A*B + B*A == C])

    g = M(F.gen())
    t = M(t)
    two = M(2)
    ginv = M(F.gen()**(-1))
    for w1 in L[0]:
        for w2 in L[1]:
            w0 = ginv*(two*g*wbar[3] -w1 -w2 - w1*w2)
            w3 = g*w0 + w2*w1
            if w0*w0 != w0 - M(1):
                continue
            if w3*w3 != M(-1):
                continue
            return w0, w1, w2, w3
        
    raise ValueError




