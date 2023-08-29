use crate::Vec2;
use std::arch::x86_64::*;

/// 2D vector with double precision
///
/// The components are laid out in this order: `[x, y]`. This struct is aligned to 16 bytes.
///
/// ## Examples
///
/// ```
/// use mafs::{Vec2, Dvec2};
///
/// // Construction
/// let a = Dvec2::new(2.0, 3.0);
/// let b = Dvec2::new(6.0, 9.0);
/// let c = Dvec2::splat(0.0); // Set all four components to the same value
///
/// // Arithmetics
/// assert_eq!(a + b, Dvec2::new(8.0, 12.0));
/// assert_eq!(a - b, Dvec2::new(-4.0, -6.0));
/// assert_eq!(a * b, Dvec2::new(12.0, 27.0));
/// assert_eq!(b / a, Dvec2::new(3.0, 3.0));
///
/// // Euclidian norm
/// assert_eq!(a.norm(), 13.0f64.sqrt());
/// assert_eq!(a.normalize().norm(), 1.0);
///
/// // Specialized operations
/// assert_eq!(a.dot(b), 39.0);
/// assert_eq!(b.dot(a), a.dot(b));
/// assert_eq!(Dvec2::new(-0.5, 0.5).floor(), Dvec2::new(-1.0, 0.0));
///
/// // Comparisons
/// assert_eq!(a.min_componentwise(b), Dvec2::new(2.0, 3.0));
/// assert_eq!(a.max_componentwise(b), Dvec2::new(6.0, 9.0));
///
/// // Reduction
/// assert_eq!(a.min_reduce(), 2.0);
/// assert_eq!(b.max_reduce(), 9.0);
/// ```
#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable, bytemuck::Pod))]
pub struct Dvec2 {
    pub(crate) inner: __m128d,
}

impl std::fmt::Debug for Dvec2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_array().fmt(f)
    }
}

impl Vec2<f64> for Dvec2 {
    #[inline]
    fn new(x: f64, y: f64) -> Dvec2 {
        unsafe {
            // The order is reversed!
            Dvec2 {
                inner: _mm_set_pd(y, x),
            }
        }
    }

    #[inline]
    fn as_array(&self) -> &[f64; 2] {
        unsafe { &*(self as *const Dvec2 as *const [f64; 2]) }
    }

    #[inline]
    fn as_mut_array(&mut self) -> &mut [f64; 2] {
        unsafe { &mut *(self as *mut Dvec2 as *mut [f64; 2]) }
    }

    #[inline]
    fn add_componentwise(&self, rhs: Dvec2) -> Dvec2 {
        unsafe {
            Dvec2 {
                inner: _mm_add_pd(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn sub_componentwise(&self, rhs: Dvec2) -> Dvec2 {
        unsafe {
            Dvec2 {
                inner: _mm_sub_pd(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn mul_componentwise(&self, rhs: Dvec2) -> Dvec2 {
        unsafe {
            Dvec2 {
                inner: _mm_mul_pd(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn div_componentwise(&self, rhs: Dvec2) -> Dvec2 {
        unsafe {
            Dvec2 {
                inner: _mm_div_pd(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn min_componentwise(&self, rhs: Dvec2) -> Dvec2 {
        unsafe {
            Dvec2 {
                inner: _mm_min_pd(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn max_componentwise(&self, rhs: Dvec2) -> Dvec2 {
        unsafe {
            Dvec2 {
                inner: _mm_max_pd(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn floor(&self) -> Dvec2 {
        unsafe {
            Dvec2 {
                inner: _mm_floor_pd(self.inner),
            }
        }
    }

    #[inline]
    fn min_reduce(&self) -> f64 {
        unsafe {
            let perm = _mm_permute_pd::<1>(self.inner);
            let reduce = _mm_min_pd(self.inner, perm);
            _mm_cvtsd_f64(reduce)
        }
    }

    #[inline]
    fn max_reduce(&self) -> f64 {
        unsafe {
            let perm = _mm_permute_pd::<1>(self.inner);
            let reduce = _mm_max_pd(self.inner, perm);
            _mm_cvtsd_f64(reduce)
        }
    }

    #[inline]
    fn eq_reduce(&self, rhs: Dvec2) -> bool {
        unsafe {
            let mask = _mm_cmpeq_pd(self.inner, rhs.inner);
            let reduce = _mm_movemask_epi8(std::mem::transmute(mask));
            reduce == 0xffff
        }
    }

    #[inline]
    fn dot(&self, rhs: Dvec2) -> f64 {
        unsafe {
            let prod = _mm_mul_pd(self.inner, rhs.inner);
            let reduce64 = _mm_add_sd(prod, _mm_permute_pd::<1>(prod));
            _mm_cvtsd_f64(reduce64)
        }
    }
}

implement_scalarops!(Dvec2, f64);
implement_vecops!(Dvec2, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq_works() {
        let a = Dvec2::new(1.0, 2.0);
        let b = Dvec2::new(1.0, 2.0);
        let c = Dvec2::new(5.0, 6.0);
        assert_eq!(a == b, true);
        assert_eq!(b == a, true);
        assert_eq!(a == a, true);
        assert_eq!(a == c, false);

        let d = Dvec2::new(0.0, -0.0);
        let e = Dvec2::new(0.0, 0.0);
        assert_eq!(d == e, true);

        let f = Dvec2::new(f64::NAN, f64::NAN);
        assert_eq!(f == f, false);
    }
}
