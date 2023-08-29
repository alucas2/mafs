use num_traits::float::Float;
use std::ops::{Add, AddAssign, Div, DivAssign, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

#[rustfmt::skip]
/// Operators where the left operand is a scalar and the right operand is a vector.
pub trait ScalarOps<V>:
    Copy
    + Add<V, Output = V>
    + Sub<V, Output = V>
    + Mul<V, Output = V>
    + Div<V, Output = V>
{}

#[rustfmt::skip]
/// Operators where the left operand is a vector and the right operand is either a vector or a scalar.
pub trait VecOps<S>:
    Copy
    + Default
    + Add<Self, Output = Self> + AddAssign<Self>
    + Sub<Self, Output = Self> + SubAssign<Self>
    + Mul<Self, Output = Self> + MulAssign<Self>
    + Div<Self, Output = Self> + DivAssign<Self>
    + Add<S, Output = Self> + AddAssign<Self>
    + Sub<S, Output = Self> + SubAssign<Self>
    + Mul<S, Output = Self> + MulAssign<S>
    + Div<S, Output = Self> + DivAssign<S>
    + Neg<Output = Self>
    + IndexMut<usize, Output = S>
    + PartialEq<Self>
{}

#[rustfmt::skip]
/// Operators where the left operand is a matrix and the right operand is either a matrix, a vector or a scalar.
pub trait MatOps<S, V>:
    Copy
    + Default
    + Add<Self, Output = Self> + AddAssign<Self>
    + Sub<Self, Output = Self> + SubAssign<Self>
    + Mul<V, Output = V>
    + Mul<Self, Output = Self> + MulAssign<Self>
    + Neg<Output = Self>
    + IndexMut<usize, Output = V>
    + PartialEq<Self>
{}

/// Methods on two-dimensional vectors.
///
/// - `S` is the type of the vector's components.
pub trait Vec2<S>
where
    Self: VecOps<S>,
    S: Float + ScalarOps<Self>,
{
    // --------------- Required methods ---------------

    /// Create a new two-dimensional vector.
    fn new(x: S, y: S) -> Self;

    /// Convert to an array.
    /// Can also use the indexing operator `[]`.
    fn as_array(&self) -> &[S; 2];

    /// Convert to a mutable array.
    /// Can also use the indexing operator`[]`.
    fn as_mut_array(&mut self) -> &mut [S; 2];

    /// Add component by component.
    /// Can also use the `+` operator.
    fn add_componentwise(&self, rhs: Self) -> Self;

    /// Subtract component by component.
    /// Can also use the `-` operator.
    fn sub_componentwise(&self, rhs: Self) -> Self;

    /// Multiply component by component.
    /// Can also use the `*` operator.
    fn mul_componentwise(&self, rhs: Self) -> Self;

    /// Divide component by component.
    /// Can also use the `/` operator.
    fn div_componentwise(&self, rhs: Self) -> Self;

    /// For each lane, select the smallest component of the two.
    fn min_componentwise(&self, rhs: Self) -> Self;

    /// For each lane, select the largest component of the two.
    fn max_componentwise(&self, rhs: Self) -> Self;

    /// Round down all components to an integer value.
    fn floor(&self) -> Self;

    /// Smallest of the four components.
    fn min_reduce(&self) -> S;

    /// Largest of the four components.
    fn max_reduce(&self) -> S;

    /// Equality of a vector to another on all components.
    fn eq_reduce(&self, rhs: Self) -> bool;

    /// Dot product.
    fn dot(&self, rhs: Self) -> S;

    // --------------- Provided methods ---------------

    /// Create a two-dimensional vector all with equal components.
    fn splat(value: S) -> Self {
        Self::new(value, value)
    }

    /// Norm of this vector.
    fn norm(&self) -> S {
        self.dot(*self).sqrt()
    }

    /// Divide by the norm to obain a normalized vector.
    fn normalize(&self) -> Self {
        self.div(Self::splat(self.norm()))
    }
}

/// Methods on four-dimensional vectors.
///
/// - `S` is the type of the vector's components.
pub trait Vec4<S>
where
    Self: VecOps<S>,
    S: Float,
{
    // --------------- Required methods ---------------

    /// Create a new two-dimensional vector.
    fn new(x: S, y: S, y: S, z: S) -> Self;

    /// Convert to an array.
    /// Can also use the indexing operator `[]`.
    fn as_array(&self) -> &[S; 4];

    /// Convert to a mutable array.
    /// Can also use the indexing operator`[]`.
    fn as_mut_array(&mut self) -> &mut [S; 4];

    /// Add component by component.
    /// Can also use the `+` operator.
    fn add_componentwise(&self, rhs: Self) -> Self;

    /// Subtract component by component.
    /// Can also use the `-` operator.
    fn sub_componentwise(&self, rhs: Self) -> Self;

    /// Multiply component by component.
    /// Can also use the `*` operator.
    fn mul_componentwise(&self, rhs: Self) -> Self;

    /// Divide component by component.
    /// Can also use the `/` operator.
    fn div_componentwise(&self, rhs: Self) -> Self;

    /// For each lane, select the smallest component of the two.
    fn min_componentwise(&self, rhs: Self) -> Self;

    /// For each lane, select the largest component of the two.
    fn max_componentwise(&self, rhs: Self) -> Self;

    /// Round down all components to an integer value.
    fn floor(&self) -> Self;

    /// Smallest of the four components.
    fn min_reduce(&self) -> S;

    /// Largest of the four components.
    fn max_reduce(&self) -> S;

    /// Equality of a vector to another on all components.
    fn eq_reduce(&self, rhs: Self) -> bool;

    /// Dot product.
    fn dot(&self, rhs: Self) -> S;

    /// Cross product.
    /// The fourth component of the operands is ignored and the fourth component of the result will be zero.
    fn cross(&self, rhs: Self) -> Self;

    // --------------- Provided methods ---------------

    /// Create a two-dimensional vector with all equal components.
    fn splat(value: S) -> Self {
        Self::new(value, value, value, value)
    }

    /// Norm of this vector.
    fn norm(&self) -> S {
        self.dot(*self).sqrt()
    }

    /// Divide by the norm to obain a normalized vector.
    fn normalize(&self) -> Self {
        self.div(Self::splat(self.norm()))
    }

    /// Create a point in 3D space, i.e. the fourth component is 1.
    fn point(x: S, y: S, z: S) -> Self {
        Self::new(x, y, z, S::one())
    }

    /// Create a direction in 3D space, i.e. the fourth component is 0.
    fn direction(x: S, y: S, z: S) -> Self {
        Self::new(x, y, z, S::zero())
    }
}

/// Methods on a 4x4 matrices.
///
/// - `S` is the type of the matrix's components.
/// - `V` is the type of the matrix's columns.
pub trait Mat4<S, V>
where
    Self: MatOps<S, V>,
    S: Float,
    V: Vec4<S>,
{
    // --------------- Required methods ---------------

    /// Create a new 4x4 matrix from its four columns.
    fn from_columns(x: V, y: V, z: V, w: V) -> Self;

    /// Convert to an array.
    /// Can also use the indexing operator `[]`.
    fn as_array(&self) -> &[V; 4];

    /// Convert to a mutable array.
    /// Can also use the indexing operator `[]`.
    fn as_mut_array(&mut self) -> &mut [V; 4];

    /// Multiply this matrix with a vector.
    /// Can also use the `*` operator.
    fn mul_vector(&self, rhs: V) -> V;

    /// Transpose.
    fn transpose(&self) -> Self;

    // --------------- Provided methods ---------------

    /// Create a new 4x4 matrix with all equal components.
    fn splat(value: S) -> Self {
        Self::from_columns(
            V::splat(value),
            V::splat(value),
            V::splat(value),
            V::splat(value),
        )
    }

    /// Create a new 4x4 matrix from its four rows
    fn from_rows(r0: [S; 4], r1: [S; 4], r2: [S; 4], r3: [S; 4]) -> Self {
        Self::from_columns(
            V::new(r0[0], r1[0], r2[0], r3[0]),
            V::new(r0[1], r1[1], r2[1], r3[1]),
            V::new(r0[2], r1[2], r2[2], r3[2]),
            V::new(r0[3], r1[3], r2[3], r3[3]),
        )
    }

    /// Identity matrix.
    fn identity() -> Self {
        Self::from_columns(
            V::new(S::one(), S::zero(), S::zero(), S::zero()),
            V::new(S::zero(), S::one(), S::zero(), S::zero()),
            V::new(S::zero(), S::zero(), S::one(), S::zero()),
            V::new(S::zero(), S::zero(), S::zero(), S::one()),
        )
    }

    /// Add component by component.
    /// Can also use the `+` operator.
    fn add_componentwise(&self, rhs: Self) -> Self {
        Self::from_columns(
            self[0] + rhs[0],
            self[1] + rhs[1],
            self[2] + rhs[2],
            self[3] + rhs[3],
        )
    }

    /// Subtract component by component.
    /// Can also use the `-` operator.
    fn sub_componentwise(&self, rhs: Self) -> Self {
        Self::from_columns(
            self[0] - rhs[0],
            self[1] - rhs[1],
            self[2] - rhs[2],
            self[3] - rhs[3],
        )
    }

    /// Multiply this matrix with another matrix.
    /// Can also use the `*` operator.
    fn mul_matrix(&self, rhs: Self) -> Self {
        Self::from_columns(
            self.mul_vector(rhs[0]),
            self.mul_vector(rhs[1]),
            self.mul_vector(rhs[2]),
            self.mul_vector(rhs[3]),
        )
    }

    /// Assume that this matrix is a rotation+translation matrix and computes its inverse.
    /// If this matrix is not a rotation+translation, the result will be nonsense.
    fn inverse_se3(&self) -> Self {
        let mut m = *self;
        let p = m[3];
        m[3] = V::new(S::zero(), S::zero(), S::zero(), S::one());
        m = m.transpose(); // Inverse the rotation
        m[3] = -m.mul_vector(p); // Inverse the translation
        m[3][3] = S::one();
        m
    }
}
