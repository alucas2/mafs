macro_rules! implement_scalarops {
    ($V: ident, $S: ident) => {
        // Scalar + Vector
        impl std::ops::Add<$V> for $S {
            type Output = $V;

            #[inline]
            fn add(self, rhs: $V) -> $V {
                $V::splat(self).add_componentwise(rhs)
            }
        }

        // Scalar - Vector
        impl std::ops::Sub<$V> for $S {
            type Output = $V;

            #[inline]
            fn sub(self, rhs: $V) -> $V {
                $V::splat(self).sub_componentwise(rhs)
            }
        }

        // Scalar * Vector
        impl std::ops::Mul<$V> for $S {
            type Output = $V;

            #[inline]
            fn mul(self, rhs: $V) -> $V {
                $V::splat(self).mul_componentwise(rhs)
            }
        }

        // Scalar / Vector
        impl std::ops::Div<$V> for $S {
            type Output = $V;

            #[inline]
            fn div(self, rhs: $V) -> $V {
                $V::splat(self).div_componentwise(rhs)
            }
        }

        impl crate::traits::ScalarOps<$V> for $S {}
    };
}

macro_rules! implement_vecops {
    ($V: ident, $S: ident) => {
        // Zero
        impl Default for $V {
            #[inline]
            fn default() -> $V {
                $V::splat(num_traits::zero())
            }
        }

        // Vector + Vector
        impl std::ops::Add<$V> for $V {
            type Output = $V;

            #[inline]
            fn add(self, rhs: $V) -> $V {
                self.add_componentwise(rhs)
            }
        }

        // Vector += Vector
        impl std::ops::AddAssign<$V> for $V {
            #[inline]
            fn add_assign(&mut self, rhs: $V) {
                *self = self.add_componentwise(rhs)
            }
        }

        // Vector - Vector
        impl std::ops::Sub<$V> for $V {
            type Output = $V;

            #[inline]
            fn sub(self, rhs: $V) -> $V {
                self.sub_componentwise(rhs)
            }
        }

        // Vector -= Vector
        impl std::ops::SubAssign<$V> for $V {
            #[inline]
            fn sub_assign(&mut self, rhs: $V) {
                *self = self.sub_componentwise(rhs)
            }
        }

        // Vector * Vector
        impl std::ops::Mul<$V> for $V {
            type Output = $V;

            #[inline]
            fn mul(self, rhs: $V) -> $V {
                self.mul_componentwise(rhs)
            }
        }

        // Vector *= Vector
        impl std::ops::MulAssign<$V> for $V {
            #[inline]
            fn mul_assign(&mut self, rhs: $V) {
                *self = self.mul_componentwise(rhs)
            }
        }

        // Vector / Vector
        impl std::ops::Div<$V> for $V {
            type Output = $V;

            #[inline]
            fn div(self, rhs: $V) -> $V {
                self.div_componentwise(rhs)
            }
        }

        // Vector /= Vector
        impl std::ops::DivAssign<$V> for $V {
            #[inline]
            fn div_assign(&mut self, rhs: $V) {
                *self = self.div_componentwise(rhs)
            }
        }

        // Vector + Scalar
        impl std::ops::Add<$S> for $V {
            type Output = $V;

            #[inline]
            fn add(self, rhs: $S) -> $V {
                self.add_componentwise(Self::splat(rhs))
            }
        }

        // Vector += Scalar
        impl std::ops::AddAssign<$S> for $V {
            #[inline]
            fn add_assign(&mut self, rhs: $S) {
                *self = self.add_componentwise(Self::splat(rhs))
            }
        }

        // Vector - Scalar
        impl std::ops::Sub<$S> for $V {
            type Output = $V;

            #[inline]
            fn sub(self, rhs: $S) -> $V {
                self.sub_componentwise(Self::splat(rhs))
            }
        }

        // Vector -= Scalar
        impl std::ops::SubAssign<$S> for $V {
            #[inline]
            fn sub_assign(&mut self, rhs: $S) {
                *self = self.sub_componentwise(Self::splat(rhs))
            }
        }

        // Vector * Scalar
        impl std::ops::Mul<$S> for $V {
            type Output = $V;

            #[inline]
            fn mul(self, rhs: $S) -> $V {
                self.mul_componentwise(Self::splat(rhs))
            }
        }

        // Vector *= Scalar
        impl std::ops::MulAssign<$S> for $V {
            #[inline]
            fn mul_assign(&mut self, rhs: $S) {
                *self = self.mul_componentwise(Self::splat(rhs))
            }
        }

        // Vector / Scalar
        impl std::ops::Div<$S> for $V {
            type Output = $V;

            #[inline]
            fn div(self, rhs: $S) -> $V {
                self.div_componentwise(Self::splat(rhs))
            }
        }

        // Vector /= Scalar
        impl std::ops::DivAssign<$S> for $V {
            #[inline]
            fn div_assign(&mut self, rhs: $S) {
                *self = self.div_componentwise(Self::splat(rhs))
            }
        }

        // -Vector
        impl std::ops::Neg for $V {
            type Output = $V;

            #[inline]
            fn neg(self) -> $V {
                $V::splat(num_traits::zero()).sub_componentwise(self)
            }
        }

        // Vector[index]
        impl std::ops::Index<usize> for $V {
            type Output = $S;

            #[inline]
            fn index(&self, idx: usize) -> &$S {
                &self.as_array()[idx]
            }
        }

        // Vector[index]
        impl std::ops::IndexMut<usize> for $V {
            #[inline]
            fn index_mut(&mut self, idx: usize) -> &mut $S {
                &mut self.as_mut_array()[idx]
            }
        }

        // Vector == Vector
        impl PartialEq<$V> for $V {
            fn eq(&self, rhs: &$V) -> bool {
                self.eq_reduce(*rhs)
            }
        }

        impl crate::traits::VecOps<$S> for $V {}
    };
}

macro_rules! implement_matops {
    ($M: ident, $V: ident, $S: ident) => {
        // Matrix + Matrix
        impl std::ops::Add<$M> for $M {
            type Output = $M;

            #[inline]
            fn add(self, rhs: $M) -> $M {
                self.add_componentwise(rhs)
            }
        }

        // Matrix += Matrix
        impl std::ops::AddAssign<$M> for $M {
            #[inline]
            fn add_assign(&mut self, rhs: $M) {
                *self = self.add_componentwise(rhs)
            }
        }

        // Matrix - Matrix
        impl std::ops::Sub<$M> for $M {
            type Output = $M;

            #[inline]
            fn sub(self, rhs: $M) -> $M {
                self.sub_componentwise(rhs)
            }
        }

        // Matrix -= Matrix
        impl std::ops::SubAssign<$M> for $M {
            #[inline]
            fn sub_assign(&mut self, rhs: $M) {
                *self = self.sub_componentwise(rhs)
            }
        }

        // Matrix * Vector
        impl std::ops::Mul<$V> for $M {
            type Output = $V;

            #[inline]
            fn mul(self, rhs: $V) -> $V {
                self.mul_vector(rhs)
            }
        }

        // Matrix * Matrix
        impl std::ops::Mul<$M> for $M {
            type Output = $M;

            #[inline]
            fn mul(self, rhs: $M) -> $M {
                self.mul_matrix(rhs)
            }
        }

        // Matrix *= Matrix
        impl std::ops::MulAssign<$M> for $M {
            #[inline]
            fn mul_assign(&mut self, rhs: $M) {
                *self = self.mul_matrix(rhs)
            }
        }

        // -Matrix
        impl std::ops::Neg for $M {
            type Output = $M;

            #[inline]
            fn neg(self) -> $M {
                $M::splat(num_traits::zero()).sub_componentwise(self)
            }
        }

        // Matrix[index]
        impl std::ops::Index<usize> for $M {
            type Output = $V;

            #[inline]
            fn index(&self, idx: usize) -> &$V {
                &self.as_array()[idx]
            }
        }

        // Matrix[index]
        impl std::ops::IndexMut<usize> for $M {
            #[inline]
            fn index_mut(&mut self, idx: usize) -> &mut $V {
                &mut self.as_mut_array()[idx]
            }
        }

        impl crate::traits::MatOps<$S, $V> for $M {}
    };
}
