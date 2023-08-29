use crate::Vec2;

/// 2D vector with single precision.
///
/// The components are laid out in this order: `[x, y]`. This struct is aligned to 4 bytes, not 8.
///
/// This struct is here for consistency and does not explicitly use SIMD instructions.
/// Internally, it is just an array of two floats.
///
/// ## Examples
///
/// ```
/// use mafs::{Vec2, Fvec2};
///
/// // Construction
/// let a = Fvec2::new(2.0, 3.0);
/// let b = Fvec2::new(6.0, 9.0);
/// let c = Fvec2::splat(0.0); // Set all four components to the same value
///
/// // Arithmetics
/// assert_eq!(a + b, Fvec2::new(8.0, 12.0));
/// assert_eq!(a - b, Fvec2::new(-4.0, -6.0));
/// assert_eq!(a * b, Fvec2::new(12.0, 27.0));
/// assert_eq!(b / a, Fvec2::new(3.0, 3.0));
///
/// // Euclidian norm
/// assert_eq!(a.norm(), 13.0f32.sqrt());
/// assert_eq!(a.normalize().norm(), 1.0);
///
/// // Specialized operations
/// assert_eq!(a.dot(b), 39.0);
/// assert_eq!(b.dot(a), a.dot(b));
/// assert_eq!(Fvec2::new(-0.5, 0.5).floor(), Fvec2::new(-1.0, 0.0));
///
/// // Comparisons
/// assert_eq!(a.min_componentwise(b), Fvec2::new(2.0, 3.0));
/// assert_eq!(a.max_componentwise(b), Fvec2::new(6.0, 9.0));
///
/// // Reduction
/// assert_eq!(a.min_reduce(), 2.0);
/// assert_eq!(b.max_reduce(), 9.0);
/// ```
#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable, bytemuck::Pod))]
pub struct Fvec2 {
    pub(crate) inner: [f32; 2],
}

impl std::fmt::Debug for Fvec2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_array().fmt(f)
    }
}

impl Vec2<f32> for Fvec2 {
    #[inline]
    fn new(x: f32, y: f32) -> Fvec2 {
        Fvec2 { inner: [x, y] }
    }

    #[inline]
    fn as_array(&self) -> &[f32; 2] {
        &self.inner
    }

    #[inline]
    fn as_mut_array(&mut self) -> &mut [f32; 2] {
        &mut self.inner
    }

    #[inline]
    fn add_componentwise(&self, rhs: Fvec2) -> Fvec2 {
        Fvec2 {
            inner: [self.inner[0] + rhs.inner[0], self.inner[1] + rhs.inner[1]],
        }
    }

    #[inline]
    fn sub_componentwise(&self, rhs: Fvec2) -> Fvec2 {
        Fvec2 {
            inner: [self.inner[0] - rhs.inner[0], self.inner[1] - rhs.inner[1]],
        }
    }

    #[inline]
    fn mul_componentwise(&self, rhs: Fvec2) -> Fvec2 {
        Fvec2 {
            inner: [self.inner[0] * rhs.inner[0], self.inner[1] * rhs.inner[1]],
        }
    }

    #[inline]
    fn div_componentwise(&self, rhs: Fvec2) -> Fvec2 {
        Fvec2 {
            inner: [self.inner[0] / rhs.inner[0], self.inner[1] / rhs.inner[1]],
        }
    }

    #[inline]
    fn min_componentwise(&self, rhs: Fvec2) -> Fvec2 {
        Fvec2 {
            inner: [
                self.inner[0].min(rhs.inner[0]),
                self.inner[1].min(rhs.inner[1]),
            ],
        }
    }

    #[inline]
    fn max_componentwise(&self, rhs: Fvec2) -> Fvec2 {
        Fvec2 {
            inner: [
                self.inner[0].max(rhs.inner[0]),
                self.inner[1].max(rhs.inner[1]),
            ],
        }
    }

    #[inline]
    fn floor(&self) -> Fvec2 {
        Fvec2 {
            inner: [self.inner[0].floor(), self.inner[1].floor()],
        }
    }

    #[inline]
    fn min_reduce(&self) -> f32 {
        self.inner[0].min(self.inner[1])
    }

    #[inline]
    fn max_reduce(&self) -> f32 {
        self.inner[0].max(self.inner[1])
    }

    #[inline]
    fn eq_reduce(&self, rhs: Fvec2) -> bool {
        self.inner[0] == rhs.inner[0] && self.inner[1] == rhs.inner[1]
    }

    #[inline]
    fn dot(&self, rhs: Fvec2) -> f32 {
        self.inner[0] * rhs.inner[0] + self.inner[1] * rhs.inner[1]
    }
}

implement_scalarops!(Fvec2, f32);
implement_vecops!(Fvec2, f32);
