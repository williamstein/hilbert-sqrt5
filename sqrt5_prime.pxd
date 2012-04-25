#########################################################################
#       Copyright (C) 2010-2012 William Stein <wstein@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#
#                  http://www.gnu.org/licenses/
#########################################################################

cdef class Prime:
    cdef public long p, r
    cdef bint first
    cpdef long norm(self)
    cpdef bint is_split(self)
    cpdef bint is_inert(self)
    cpdef bint is_ramified(self)
