#########################################################################
#       Copyright (C) 2010-2012 William Stein <wstein@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#
#                  http://www.gnu.org/licenses/
#########################################################################


"""
Hilbert Modular Forms over Q(sqrt(5)) of Weight (2,2)

AUTHORS:

- William Stein (2010, 2011, 2012)
"""

from sage.misc.cachefunc import cached_method
from sage.misc.all import verbose
from sage.rings.all import Integer, prime_divisors, QQ, next_prime, ZZ
from sage.matrix.all import matrix
from sage.structure.all import Sequence

from sqrt5 import F, O_F
from sqrt5_prime import primes_of_bounded_norm
from sqrt5_fast import IcosiansModP1ModN
from sqrt5_tables import ideals_of_norm, PrimesCoprimeTo, sqrt5_ideal

    

# We define the following new class instead of trying to use the code
# in sage.modular.hecke, which has too much baggage and assumptions
# about the base field.

class Space(object):
    """
    Abstract space of modular forms.

    EXAMPLES::

        sage: from sage.modular.hilbert.sqrt5_hmf import Space
        sage: S = Space(); type(S)
        <class 'sage.modular.hilbert.sqrt5_hmf.Space'>    
    """
    def __cmp__(self, right):
        """
        Compares self to a Space instance right based on level,
        dimension, and underlying vector space.  
        
        EXAMPLES::

        We test out various uses of comparisons that use all the
        relevant properties::

            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: H1 = QuaternionicModule(3 * F.prime_above(31)); H2 = QuaternionicModule(5)
            sage: H2 < H1
            True
            sage: H1 > H2
            True
            sage: D = H1.decomposition(6)
            sage: D[0].dimension(); D[-1].dimension()
            1
            2
            sage: D[-1] > D[0]
            True
            sage: D[0].dimension(), D[1].dimension()
            (1, 1)
            sage: D[1].vector_space() > D[0].vector_space()
            True
        """
        if not isinstance(right, Space):
            raise NotImplementedError
        return cmp((self.level(), self.dimension(), self.vector_space()),
                   (right.level(), right.dimension(), right.vector_space()))

    def subspace(self, V):
        """
        Return subspace of underlying vector space.
        
        INPUT:

        - `V` -- subspace of vector space

        EXAMPLES:

        This must be implemented in a derived class.::

            sage: from sage.modular.hilbert.sqrt5_hmf import Space
            sage: S = Space()
            sage: S.subspace(QQ^2)
            Traceback (most recent call last):
            ...
            NotImplementedError
        """
        raise NotImplementedError
    
    def vector_space(self):
        """
        Return underlying vector space.  

        OUTPUT:

        - vector space

        EXAMPLES::

        This must be implemented in the derived class.::
        
            sage: from sage.modular.hilbert.sqrt5_hmf import Space
            sage: Space().vector_space()        
            Traceback (most recent call last):
            ...
            NotImplementedError
        """
        raise NotImplementedError
    
    def basis(self):
        """
        Return basis for underlying vector space.  Only works if
        vector_space() is implemented.

        OUTPUT:

        - basis for vector space

        EXAMPLES::
        
            sage: from sage.modular.hilbert.sqrt5_hmf import Space
            sage: Space().basis()        
            Traceback (most recent call last):
            ...
            NotImplementedError
        """
        return self.vector_space().basis()

    def new_subspace(self, p=None):
        """
        Return (p-)"new" subspace of this space.

        This is the kernel of the degeneracy map to level
        self.level()/p, or the intersection of all degeneracy maps
        when p is None.

        WARNING: This space contains what should properly be called
        the new subspace, but may have some overlap as an anemic Hecke
        module with 

        INPUT:
        
        - `p` -- None (default) or a prime divisor of the level

        OUTPUT:

        - subspace of this space

        EXAMPLES::

        We make a space of level a product of 2 split primes and (2)::
        
            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: P = F.prime_above(31); Q = F.prime_above(11); R = F.prime_above(2)
            sage: H = QuaternionicModule(P*Q*R); H
            Quaternionic module of dimension 32, level 2*a-38 (of norm 1364=2^2*11*31) over QQ(sqrt(5))

        The full new space::
        
            sage: N = H.new_subspace(); N
            Subspace of dimension 22 of Quaternionic module of dimension 32, level 2*a-38 (of norm 1364=2^2*11*31) over QQ(sqrt(5))

        The new subspace for each prime divisor of the level::
        
            sage: N_P = H.new_subspace(P); N_P
            Subspace of dimension 31 of Quaternionic module of dimension 32, level 2*a-38 (of norm 1364=2^2*11*31) over QQ(sqrt(5))
            sage: N_Q = H.new_subspace(Q); N_Q
            Subspace of dimension 28 of Quaternionic module of dimension 32, level 2*a-38 (of norm 1364=2^2*11*31) over QQ(sqrt(5))
            sage: N_R = H.new_subspace(R); N_R
            Subspace of dimension 24 of Quaternionic module of dimension 32, level 2*a-38 (of norm 1364=2^2*11*31) over QQ(sqrt(5))
            sage: N_P.intersection(N_Q).intersection(N_R) == N
            True

        An example that illustrates that the "new" and old subspaces can have
        a common system of Hecke eigenvalues, at least for the Hecke operators
        of index coprime to the level::

            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule, PrimesCoprimeTo
            sage: N1 = F.prime_above(31); N2 = 2*N1
            sage: D2 = QuaternionicModule(N2).new_subspace().decomposition(10); D2
            [
            Subspace of dimension 1...
            Subspace of dimension 1...
            ]
            sage: D1 = QuaternionicModule(N1).new_subspace().decomposition(10); D1
            [
            Subspace of dimension 1...
            Subspace of dimension 1...
            ]

        Finally we list the systems of eigenvalues on these
        1-dimensional spaces, noting that they are not all distinct::
        
            sage: [D1[0].hecke_matrix(p) for p in PrimesCoprimeTo(N2, 40)]
            [[-2], [2], [-4], [4], [4], [-4], [-2], [-2], [8]]
            sage: [D2[1].hecke_matrix(p) for p in PrimesCoprimeTo(N2, 40)]
            [[-2], [2], [-4], [4], [4], [-4], [-2], [-2], [8]]            
            sage: [D1[1].hecke_matrix(p) for p in PrimesCoprimeTo(N2, 40)]   # eisenstein
            [[6], [10], [12], [12], [20], [20], [30], [30], [32]]
            sage: [D2[0].hecke_matrix(p) for p in PrimesCoprimeTo(N2, 40)]   # truly new
            [[0], [-2], [0], [-6], [2], [2], [6], [0], [-4]]

        At the prime 2, the systems of eigenvalues do differ.  For
        D1[0], we have that at the prime 2, the eigenvalue is -3.
        However, the eigenvalue must be 1 or -1 for the curve
        corresponding to D2[1], since that curve has multiplicative
        reduction at 2::

            sage: D1[0].hecke_matrix(2)
            [-3]
        """
        V = self.degeneracy_matrix(p).kernel()
        return self.subspace(V)

    def decomposition(self, B):
        """
        Return Hecke decomposition of self using Hecke operators T_p
        coprime to the level with norm(p) <= B.

        INPUT:

        - `B` -- positive integer

        OUTPUT:

        - sorted Sequence of subspaces of self

        EXAMPLES::

            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: H = QuaternionicModule(F.prime_above(31))
            sage: H.decomposition(10)
            [
            Subspace of dimension 1 of Quaternionic module of dimension 2, level 5*a-2 (of norm 31=31) over QQ(sqrt(5)),
            Subspace of dimension 1 of Quaternionic module of dimension 2, level 5*a-2 (of norm 31=31) over QQ(sqrt(5))
            ]
            sage: H.decomposition(2)
            [
            Quaternionic module of dimension 2, level 5*a-2 (of norm 31=31) over QQ(sqrt(5))
            ]

            sage: H = QuaternionicModule(3 * F.prime_above(31)); H
            Quaternionic module of dimension 6, level 15*a-6 (of norm 279=3^2*31) over QQ(sqrt(5))
            sage: H.decomposition(10)
            [
            Subspace of dimension 1 ...,
            Subspace of dimension 1 ...,
            Subspace of dimension 1 ...,
            Subspace of dimension 1 ...,
            Subspace of dimension 2 ...
            ]            
        """
        primes = PrimesCoprimeTo(self.level(), B)
        if len(primes) == 0:
            D = [self]
        else:
            T = self.hecke_matrix(primes.next())
            D = T.decomposition()
            while len([X for X in D if not X[1]]) > 0 and len(primes) > 0:
                p = primes.next()
                verbose('Norm(p) = %s'%p.norm())
                T = self.hecke_matrix(p)
                D2 = []
                for X in D:
                    if X[1]:
                        D2.append(X)
                    else:
                        for Z in T.decomposition_of_subspace(X[0]):
                            D2.append(Z)
                D = D2
            D = [self.subspace(X[0]) for X in D]
            D.sort()
            
        S = Sequence(D, immutable=True, cr=True, universe=int, check=False)
        return S

    def new_decomposition(self):
        """
        Return complete irreducible Hecke decomposition of "new"
        subspace of self.  

        OUTPUT:

        - sorted Sequence of subspaces of self

        EXAMPLES::

            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: H = QuaternionicModule(3 * F.prime_above(31)); H
            Quaternionic module of dimension 6, level 15*a-6 (of norm 279=3^2*31) over QQ(sqrt(5))
            sage: H.new_decomposition()
            [
            Subspace of dimension 1 ...,
            Subspace of dimension 1 ...,
            Subspace of dimension 1 ...,
            Subspace of dimension 1 ...
            ]        
        """
        V = self.degeneracy_matrix().kernel()
        primes = PrimesCoprimeTo(self.level())
        p = primes.next()
        T = self.hecke_matrix(p)
        D = T.decomposition_of_subspace(V)
        
        while len([X for X in D if not X[1]]) > 0:
            p = primes.next()
            verbose('Norm(p) = %s'%p.norm())
            T = self.hecke_matrix(p)
            D2 = []
            for X in D:
                if X[1]:
                    D2.append(X)
                else:
                    for Z in T.decomposition_of_subspace(X[0]):
                        D2.append(Z)
            D = D2
        D = [self.subspace(X[0]) for X in D]
        D.sort()
        S = Sequence(D, immutable=True, cr=True, universe=int, check=False)
        return S

class QuaternionicModule(Space):
    """
    QQ-vector space isomorphic as a Hecke module to a QQ-subspace of a
    Quaternionic module over Q(sqrt(5)) of parallel weight 2 and some
    level.

    EXAMPLES::

        sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
        sage: H = QuaternionicModule(100); H
        Quaternionic module of dimension 250, level 100 (of norm 10000=2^4*5^4) over QQ(sqrt(5))
        sage: type(H)
        <class 'sage.modular.hilbert.sqrt5_hmf.QuaternionicModule'>
        sage: QuaternionicModule(F.prime_above(5) * 31)
        Quaternionic module of dimension 104, level -62*a+31 (of norm 4805=5*31^2) over QQ(sqrt(5))
    """
    def __init__(self, level):
        """
        INPUT:

        - ``level`` -- an ideal or element of ZZ[(1+sqrt(5))/2].

        TESTS::

            sage: H = sage.modular.hilbert.sqrt5_hmf.QuaternionicModule(3); H
            Quaternionic module of dimension 1, level 3 (of norm 9=3^2) over QQ(sqrt(5))
            sage: loads(dumps(H)) == H
            True
        """
        self._level = sqrt5_ideal(level)
        self._gen = self._level.gens_reduced()[0]
        self._icosians_mod_p1 = IcosiansModP1ModN(self._level)
        self._dimension = self._icosians_mod_p1.cardinality()
        self._vector_space = QQ**self._dimension
        self._hecke_matrices = {}
        self._degeneracy_matrices = {}

    def __repr__(self):
        """
        EXAMPLES::

            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: QuaternionicModule(F.prime_above(5) * 31).__repr__()
            'Quaternionic module of dimension 104, level -62*a+31 (of norm 4805=5*31^2) over QQ(sqrt(5))'
        """
        return "Quaternionic module of dimension %s, level %s (of norm %s=%s) over QQ(sqrt(5))"%(
            self._dimension, str(self._gen).replace(' ',''),
            self._level.norm(), str(self._level.norm().factor()).replace(' ',''))

    def intersection(self, M):
        """
        Return intsection of self and other.  Since self is an ambient
        module, this only makes sense when M is a subspace of self, in
        which case M is returned.  Otherwise, a TypeError is raised.

        INPUT:

        - `M` -- (sub)space of Quaternionic module

        OUTPUT::

        - (sub)space of Quaternionic module

        EXAMPLES::

            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: H = QuaternionicModule(2 * F.prime_above(31))
            sage: D = H.decomposition(10)

        The two cases where intersection just returns the object we
        are intersecting self with::
        
            sage: H.intersection(H) is H
            True
            sage: H.intersection(D[0]) is D[0]
            True

        Everything else will just raise a TypeError::
        
            sage: H2 = QuaternionicModule(F.prime_above(31))
            sage: H.intersection(H2)
            Traceback (most recent call last):
            ...
            TypeError
            sage: H.intersection(0)
            Traceback (most recent call last):
            ...
            TypeError
            sage: H.intersection(H2.decomposition(10)[0])
            Traceback (most recent call last):
            ...
            TypeError
        """
        if isinstance(M, QuaternionicModule):
            if self != M:
                raise TypeError
            return self
        if isinstance(M, QuaternionicModuleSubspace):
            if self != M.ambient():
                raise TypeError
            return M
        raise TypeError

    def level(self):
        """
        Return the level of self.

        OUTPUT:

        - ``level`` -- ideal

        EXAMPLES::

            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: I = 2*F.prime_above(31); I
            Fractional ideal (10*a - 4)
            sage: QuaternionicModule(I).level()
            Fractional ideal (10*a - 4)        
        """
        return self._level
 
    def vector_space(self):
        """
        Return underlying vector space.  This is an ambient vector space over QQ.

        EXAMPLES::

            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: H = QuaternionicModule(F.prime_above(101)); H
            Quaternionic module of dimension 3, level 9*a-4 (of norm 101=101) over QQ(sqrt(5))
            sage: H.vector_space()
            Vector space of dimension 3 over Rational Field        
        """
        return self._vector_space

    def weight(self):
        """
        Return the weight, which is (2,2).

        EXAMPLES::

            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: H = QuaternionicModule(F.prime_above(101)); H.weight()
            (2, 2)        
        """
        return (Integer(2),Integer(2))

    def dimension(self):
        """
        Return the dimension of this space.

        EXAMPLES::

            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: H = QuaternionicModule(2*F.prime_above(101)); H
            Quaternionic module of dimension 9, level 18*a-8 (of norm 404=2^2*101) over QQ(sqrt(5))
            sage: H.dimension()
            9        
        """
        return self._dimension

    def hecke_matrix(self, n):
        """
        Return the matrix of the `n`-th Hecke operator.

        This is only implemented when `n` is a prime ideal that is
        coprime to the level, though the notion of Hecke operator
        is defined for any nonzero ideal `n`.
        
        INPUT:

        - `n` -- nonzero prime ideal of ring of integers of
          QQ(sqrt(5))

        OUTPUT:

        - a matrix over the rational numbers with integer entries
        
        EXAMPLES::

            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: H = QuaternionicModule(F.prime_above(31)); H
            Quaternionic module of dimension 2, level 5*a-2 (of norm 31=31) over QQ(sqrt(5))
            sage: H.hecke_matrix(2)
            [0 5]
            [3 2]
            sage: P = F.prime_above(5); P
            Fractional ideal (-2*a + 1)
            sage: H.hecke_matrix(P)
            [1 5]
            [3 3]

        At least the prime does not have to be coprime to the norm of the level::
        
            sage: v = F.primes_above(31)
            sage: H = QuaternionicModule(v[0]); H
            Quaternionic module of dimension 2, level 5*a-2 (of norm 31=31) over QQ(sqrt(5))
            sage: H.hecke_matrix(v[1])
            [17 15]
            [ 9 23]            

        The input must be nonzero, or you get a ValueError::
        
            sage: H.hecke_matrix(F.ideal(0))
            Traceback (most recent call last):
            ...
            ValueError: n must be nonzero

        We illustrate some shortcomings of this function::
        
            sage: H.hecke_matrix(P^2)
            Traceback (most recent call last):
            ...
            NotImplementedError: n must be prime
            sage: H.hecke_matrix(F.prime_above(31))
            Traceback (most recent call last):
            ...
            NotImplementedError: n must be coprime to the level

        You may also use T as an alias for hecke_matrix::

            sage: H.T(3)
            [5 5]
            [3 7]        
        """
        # I'm not using @cached_method, since I want to ensure that
        # the input "n" is properly normalized.  I also want it
        # to be transparent to see which matrices have been computed,
        # to clear the cache, etc.
        n = sqrt5_ideal(n)
        if n.is_zero():
            raise ValueError, "n must be nonzero"
        if not n.is_prime():
            raise NotImplementedError, "n must be prime"
        if not self.level().is_coprime(n):
            raise NotImplementedError, "n must be coprime to the level"
        if self._hecke_matrices.has_key(n):
            return self._hecke_matrices[n]
        t = self._icosians_mod_p1.hecke_matrix(n)
        t.set_immutable()
        self._hecke_matrices[n] = t
        return t

    T = hecke_matrix

    def degeneracy_matrix(self, p=None):
        """
        Map from self to QuaterniocModule of level self/p.
        
        EXAMPLES::

            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: H = QuaternionicModule(2*F.prime_above(31)); H
            Quaternionic module of dimension 4, level 10*a-4 (of norm 124=2^2*31) over QQ(sqrt(5))
            sage: H.degeneracy_matrix(2)
            [1 0]
            [0 1]
            [0 1]
            [0 1]
            sage: H.degeneracy_matrix(F.prime_above(31))
            [1]
            [1]
            [1]
            [1]
            sage: H.degeneracy_matrix()
            [1 0 1]
            [0 1 1]
            [0 1 1]
            [0 1 1]
            sage: H.degeneracy_matrix() is H.degeneracy_matrix()
            False
            sage: H.degeneracy_matrix(2) is H.degeneracy_matrix(2)
            True        
        """
        if self.level().is_prime():
            return matrix(QQ, self.dimension(), 0, sparse=True)
        if self._degeneracy_matrices.has_key(p):
            return self._degeneracy_matrices[p]
        if p is None:
            A = None
            for p in prime_divisors(self._level):
                A = self.degeneracy_matrix(p) if A is None else A.augment(self.degeneracy_matrix(p))
            A.set_immutable()
            self._degeneracy_matrices[None] = A
            return A
        p = sqrt5_ideal(p)
        if self._degeneracy_matrices.has_key(p):
            return self._degeneracy_matrices[p]
        d = self._icosians_mod_p1.degeneracy_matrix(p)
        d.set_immutable()
        self._degeneracy_matrices[p] = d
        return d
                
    def __cmp__(self, other):
        """

        TESTS::

        Create some spaces::
        
            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: H = QuaternionicModule(2*F.prime_above(31))
            sage: A = H.decomposition(10)[-1]; B = H.subspace(H.vector_space())
            sage: C = QuaternionicModule(F.prime_above(31))

        Do some consistency checks::
        
            sage: H > A     # A is a submodule
            True
            sage: A < H     # symmetry
            True
            sage: type(B)   # B is of a different type, but is the same space
            <class 'sage.modular.hilbert.sqrt5_hmf.QuaternionicModuleSubspace'>
            sage: H == B    
            True
            sage: B < H     # not strict containment
            False
            sage: B <= H
            True
            sage: H == C    # C has a completely different (smaller) level
            False
            sage: C < H
            True
            sage: cmp(H, '5')  # can't compare to just anything
            Traceback (most recent call last):
            ...
            NotImplementedError
        """
        if isinstance(other, QuaternionicModuleSubspace):
            if other.ambient() != self:
                return cmp(self, other.ambient())
            else:
                return cmp(self.dimension(), other.dimension())
        if not isinstance(other, QuaternionicModule):
            raise NotImplementedError
        # first sort by norms, since Sage ideals sort stupidly.
        return cmp((self._level.norm(), self._level), (other._level.norm(), other._level))

    def subspace(self, V):
        """
        Return the subspace of self defined by the vector space V,
        which we consider as a subspace of self.vector_space().

        WARNING: We do not require that V be invariant under the Hecke
        operators; if it is not, you may get an error when computing
        Hecke operators.
        
        INPUT:

        - `V` -- vector space over QQ, which is assumed invariant
          under the Hecke operators

        EXAMPLES::

            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: H = QuaternionicModule(2*F.prime_above(31)); D = H.decomposition(10); D
            [
            Subspace of dimension 1 of ...
            Subspace of dimension 1 of ...
            Subspace of dimension 2 of ...
            ]
            sage: H.subspace(H.vector_space())
            Subspace of dimension 4 of ...
            sage: H.subspace(D[0].vector_space())
            Subspace of dimension 1 of ...
            sage: H.subspace(D[0].vector_space() + D[1].vector_space())
            Subspace of dimension 2 of ...
            sage: J = H.subspace(D[0].vector_space() + D[1].vector_space()); J.hecke_matrix(3).fcp()
            (x - 10) * (x + 2)
            sage: D[0].hecke_matrix(3).charpoly(), D[1].hecke_matrix(3).charpoly()
            (x + 2, x - 10)

        Next we give invalid input in various ways::

            sage: H.subspace(QQ^10)                 # wrong degree
            Traceback (most recent call last):
            ...
            ValueError: V must be a subspace of the vector space underlying H
            sage: H.subspace(GF(7)^H.dimension())   # over wrong field
            Traceback (most recent call last):
            ...
            ValueError: V must have base field QQ
            sage: H.subspace(ZZ^H.dimension())      # a module
            Traceback (most recent call last):
            ...
            TypeError: V must be a vector space
        """
        return QuaternionicModuleSubspace(self, V)
    
    def rational_newforms(self):
        """
        Return the newforms with QQ-rational Hecke eigenvalues.

        Conjecturally, these correspond to the isogeny classes of
        elliptic curves over Q(sqrt(5)) having conductor self.level().

        WARNING/TODO: This relies on an unproven (but surely correct)
        bound to determine whether a system of Hecke eigenvalues is
        really old.
        
        EXAMPLES::


        The smallest level example::
        
            sage: from sage.modular.hilbert.sqrt5_hmf import F, QuaternionicModule
            sage: H = QuaternionicModule(F.prime_above(31)); D = H.rational_newforms(); D
            [
            Rational newform number 0...
            ]
            sage: f = D[0]; f
            Rational newform number 0 over QQ(sqrt(5)) in Quaternionic module of dimension 2, level 5*a-2 (of norm 31=31) over QQ(sqrt(5))

        Notice that computing `a_P` at the bad primes `P` isn't implemented
        (it just gives '?')::
        
            sage: f.aplist(50)
            [-3, -2, 2, -4, 4, 4, -4, -2, -2, 8, '?', -6, -6, 2]

        Another example of higher level::

            sage: H = QuaternionicModule(2*F.prime_above(31)); D = H.rational_newforms(); D
            [
            Rational newform number 0 ...
            ]
            sage: D[0].aplist(33)
            ['?', 0, -2, 0, -6, 2, 2, 6, 0, -4, '?']
        """
        primes = PrimesCoprimeTo(self._level)
        D = [X for X in self.new_decomposition() if X.dimension() == 1]

        # Have to get rid of the Eisenstein factor
        p = primes.next()
        while True:
            q = p.residue_field().cardinality() + 1
            E = [A for A in D if A.hecke_matrix(p)[0,0] == q]
            if len(E) == 0:
                break
            elif len(E) == 1:
                D = [A for A in D if A != E[0]]
                break
            else:
                p = primes.next()

        Z = []
        for number, X in enumerate(D):
            f = QuaternionicRationalNewform(X, number)
            try:
                # ensure that dual eigenspace is defined, i.e., that
                # newform really is new.
                f.dual_eigenspace()
                Z.append(f)
            except RuntimeError:
                pass
            
        return Sequence(Z, immutable=True, cr=True, universe=int, check=False)

from sage.modules.module import is_VectorSpace

class QuaternionicModuleSubspace(Space):
    def __init__(self, H, V):
        if not is_VectorSpace(V):
            raise TypeError, "V must be a vector space"
        if not isinstance(H, QuaternionicModule):
            raise TypeError, "H must be a QuaternionicModule"
        if V.base_ring() != QQ:
            raise ValueError,"V must have base field QQ"
        if H.dimension() != V.degree():
            raise ValueError, "V must be a subspace of the vector space underlying H"
        self._H = H
        self._V = V

    def __repr__(self):
        return "Subspace of dimension %s of %s"%(self._V.dimension(), self._H)

    def subspace(self, V):
        A = self.ambient()
        if not is_VectorSpace(V):
            raise TypeError, "V must be a vector space"
        if V.degree() != self.dimension():
            raise ValueError, "V must have degree the dimension of self"
        return A.subspace((V.basis_matrix() * self._V.basis_matrix()).row_module())

    def intersection(self, M):
        if isinstance(M, QuaternionicModule):
            assert self.ambient() == M
            return self
        if isinstance(M, QuaternionicModuleSubspace):
            assert self.ambient() == M.ambient()
            H = self.ambient()
            V = self.vector_space().intersection(M.vector_space())
            return QuaternionicModuleSubspace(H, V)
        raise TypeError

    def ambient(self):
        return self._H

    def vector_space(self):
        return self._V

    def hecke_matrix(self, n):
        return self._H.hecke_matrix(n).restrict(self._V)
    T = hecke_matrix

    def degeneracy_matrix(self, p):
        return self._H.degeneracy_matrix(p).restrict_domain(self._V)

    def level(self):
        return self._H.level()

    def dimension(self):
        return self._V.dimension()
        

class QuaternionicRationalNewform(object):
    """
    A subspace of the new subspace of a space of weight 2 Hilbert
    modular forms that (conjecturally) corresponds to an elliptic
    curve.
    """
    def __init__(self, S, number):
        """
        INPUT:
            - S -- subspace of a Quaternionic module
            - ``number`` -- nonnegative integer indicating some
              ordering among the factors of a given level.
        """
        self._S = S
        self._number = number

    def __repr__(self):
        """
        EXAMPLES::

            sage: from sage.modular.hilbert.sqrt5_hmf import QuaternionicModule, F
            sage: H = QuaternionicModule(F.prime_above(31)).rational_newforms()[0]
            sage: type(H)
            <class 'sage.modular.hilbert.sqrt5_hmf.QuaternionicRationalNewform'>
            sage: H.__repr__()
            'Isogeny class of elliptic curves over QQ(sqrt(5)) attached to form number 0 in Quaternionic module of dimension 2, level 5*a-2 (of norm 31=31) over QQ(sqrt(5))'
        """
        return "Rational newform number %s over QQ(sqrt(5)) in %s"%(self._number, self._S.ambient())

    def base_field(self):
        """
        Return the base field of this elliptic curve factor.

        OUTPUT:
            - the field Q(sqrt(5))
        
        EXAMPLES::
        
            sage: from sage.modular.hilbert.sqrt5_hmf import QuaternionicModule, F
            sage: H = QuaternionicModule(F.prime_above(31)).rational_newforms()[0]
            sage: H.base_field()
            Number Field in a with defining polynomial x^2 - x - 1
        """
        return F

    def conductor(self):
        """
        Return the conductor of this elliptic curve factor, which is
        the level of the Quaternionic module.

        OUTPUT:
            - ideal of the ring of integers of Q(sqrt(5))
        
        EXAMPLES::
        
        """
        return self._S.level()

    def ap(self, P):
        """
        Return the trace of Frobenius at the prime P, for a prime P of
        good reduction.

        INPUT:
            - `P` -- a prime ideal of the ring of integers of Q(sqrt(5)).
            
        OUTPUT:
            - an integer

        EXAMPLES::

            sage: from sage.modular.hilbert.sqrt5_hmf import QuaternionicModule, F
            sage: H = QuaternionicModule(F.primes_above(31)[0]).rational_newforms()[0]
            sage: H.ap(F.primes_above(11)[0])
            4
            sage: H.ap(F.prime_above(5))
            -2
            sage: H.ap(F.prime_above(7))
            2

        We check that the ap we compute here match with those of a known elliptic curve
        of this conductor::

            sage: a = F.0; E = EllipticCurve(F, [1,a+1,a,a,0])
            sage: E.conductor().norm()
            31
            sage: 11+1 - E.change_ring(F.primes_above(11)[0].residue_field()).cardinality()
            4
            sage: 5+1 - E.change_ring(F.prime_above(5).residue_field()).cardinality()
            -2
            sage: 49+1 - E.change_ring(F.prime_above(7).residue_field()).cardinality()
            2
        """
        if P.divides(self.conductor()):
            if (P*P).divides(self.conductor()):
                # It is 0, because the reduction is additive.
                return ZZ(0)
            else:
                # TODO: It is +1 or -1, but I do not yet know how to
                # compute which without using the L-function.
                return '?'
        else:
            return self._S.hecke_matrix(P)[0,0]

    @cached_method
    def dual_eigenspace(self, B=None):
        """
        Return 1-dimensional subspace of the dual of the ambient space
        with the same system of eigenvalues as self.  This is useful when
        computing a large number of `a_P`.

        If we can't find such a subspace using Hecke operators of norm
        less than B, then we raise a RuntimeError.  This should only happen
        if you set B way too small, or self is actually not new.
        
        INPUT:
            - B -- Integer or None; if None, defaults to a heuristic bound.
        """
        N = self.conductor()
        H = self._S.ambient()
        V = H.vector_space()
        if B is None:
            # TODO: This is a heuristic guess at a "Sturm bound"; it's the same
            # formula as the one over QQ for Gamma_0(N).  I have no idea if this
            # is correct or not yet. It is probably much too large. -- William Stein
            from sage.modular.all import Gamma0
            B = Gamma0(N.norm()).index()//6 + 1
        for P in primes_of_bounded_norm(B+1):
            P = P.sage_ideal()
            if V.dimension() == 1:
                return V
            if not P.divides(N):
                T = H.hecke_matrix(P).transpose()
                V = (T - self.ap(P)).kernel_on(V)
        raise RuntimeError, "unable to isolate 1-dimensional space"

    @cached_method
    def dual_eigenvector(self, B=None):
        # 1. compute dual eigenspace
        E = self.dual_eigenspace(B)
        assert E.dimension() == 1
        # 2. compute normalized integer eigenvector in dual
        return E.basis_matrix()._clear_denom()[0][0]

    def aplist(self, B, dual_bound=None, algorithm='dual'):
        """
        Return list of traces of Frobenius for all primes P of norm
        less than bound.  Use the function
        sage.modular.hilbert.sqrt5_prime.primes_of_bounded_norm(B)
        to get the corresponding primes.

        INPUT:
            - `B` -- a nonnegative integer
            - ``dual_bound`` -- default None; passed to dual_eigenvector function
            - ``algorithm`` -- 'dual' (default) or 'direct'

        OUTPUT:
             - a list of Sage integers

        EXAMPLES::

        We compute the aplists up to B=50::

            sage: from sage.modular.hilbert.sqrt5_hmf import QuaternionicModule, F
            sage: H = QuaternionicModule(F.primes_above(71)[1]).rational_newforms()[0]
            sage: v = H.aplist(50); v
            [-1, 0, -2, 0, 0, 2, -4, 6, -6, 8, 2, 6, 12, -4]

        This agrees with what we get using an elliptic curve of this
        conductor::
        
            sage: a = F.0; E = EllipticCurve(F, [a,a+1,a,a,0])
            sage: from psage.ellcurve.lseries.aplist_sqrt5 import aplist   # optional - psage
            sage: w = aplist(E, 50)                                        # optional - psage
            sage: v == w                                                   # optional - psage
            True

        We compare the output from the two algorithms up to norm 75::

            sage: H.aplist(75, algorithm='direct') == H.aplist(75, algorithm='dual')
            True
        """
        primes = [P.sage_ideal() for P in primes_of_bounded_norm(B)]
        if algorithm == 'direct':
            return [self.ap(P) for P in primes]
        elif algorithm == 'dual':
            v = self.dual_eigenvector(dual_bound)
            i = v.nonzero_positions()[0]
            c = v[i]
            I = self._S.ambient()._icosians_mod_p1
            N = self.conductor()
            aplist = []
            for P in primes:
                if P.divides(N):
                    ap = self.ap(P)
                else:
                    ap = (I.hecke_operator_on_basis_element(P,i).dot_product(v))/c
                aplist.append(ap)
            return aplist
        else:
            raise ValueError, "unknown algorithm '%s'"%algorithm

