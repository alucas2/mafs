use crate::Vec4;
use std::arch::x86_64::*;

/// 4D vector with single precision
///
/// The components are laid out in this order: `[x, y, z, w]`. This struct is aligned to 16 bytes.
///
/// ## Examples
///
/// ```
/// use mafs::{Vec4, Fvec4};
///
/// // Construction
/// let a = Fvec4::new(2.0, 3.0, 5.0, 6.0);
/// let b = Fvec4::new(6.0, 9.0, 2.5, 3.0);
/// let c = Fvec4::splat(0.0); // Set all four components to the same value
///
/// let p = Fvec4::point(1.0, 2.0, 3.0);
/// assert_eq!(p[3], 1.0); // Fourth component of a point is one
///
/// let d = Fvec4::direction(1.0, 2.0, 3.0);
/// assert_eq!(d[3], 0.0); // Fourth component of a direction is zero
///
/// // Arithmetics
/// assert_eq!(a + b, Fvec4::new(8.0, 12.0, 7.5, 9.0));
/// assert_eq!(a - b, Fvec4::new(-4.0, -6.0, 2.5, 3.0));
/// assert_eq!(a * b, Fvec4::new(12.0, 27.0, 12.5, 18.0));
/// assert_eq!(b / a, Fvec4::new(3.0, 3.0, 0.5, 0.5));
///
/// // Euclidian norm
/// assert_eq!(a.norm(), 74.0f32.sqrt());
/// assert_eq!(a.normalize().norm(), 0.99999994); // hmmmm
///
/// // Specialized operations
/// assert_eq!(a.dot(b), 69.5);
/// assert_eq!(b.dot(a), a.dot(b));
/// assert_eq!(a.cross(b), Fvec4::new(-37.5, 25.0, 0.0, 0.0));
/// assert_eq!(b.cross(a), -a.cross(b));
/// assert_eq!(Fvec4::new(-0.5, 0.5, 2.9, 0.0).floor(), Fvec4::new(-1.0, 0.0, 2.0, 0.0));
///
/// // Comparisons
/// assert_eq!(a.min_componentwise(b), Fvec4::new(2.0, 3.0, 2.5, 3.0));
/// assert_eq!(a.max_componentwise(b), Fvec4::new(6.0, 9.0, 5.0, 6.0));
///
/// // Reduction
/// assert_eq!(a.min_reduce(), 2.0);
/// assert_eq!(b.max_reduce(), 9.0);
/// ```
#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable, bytemuck::Pod))]
pub struct Fvec4 {
    pub(crate) inner: __m128,
}

impl std::fmt::Debug for Fvec4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_array().fmt(f)
    }
}

impl Vec4<f32> for Fvec4 {
    #[inline]
    fn new(x: f32, y: f32, z: f32, w: f32) -> Fvec4 {
        unsafe {
            Fvec4 {
                inner: _mm_set_ps(w, z, y, x),
            }
        }
    }

    #[inline]
    fn as_array(&self) -> &[f32; 4] {
        unsafe { &*(self as *const Fvec4 as *const [f32; 4]) }
    }

    #[inline]
    fn as_mut_array(&mut self) -> &mut [f32; 4] {
        unsafe { &mut *(self as *mut Fvec4 as *mut [f32; 4]) }
    }

    #[inline]
    fn add_componentwise(&self, rhs: Fvec4) -> Fvec4 {
        unsafe {
            Fvec4 {
                inner: _mm_add_ps(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn sub_componentwise(&self, rhs: Fvec4) -> Fvec4 {
        unsafe {
            Fvec4 {
                inner: _mm_sub_ps(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn mul_componentwise(&self, rhs: Fvec4) -> Fvec4 {
        unsafe {
            Fvec4 {
                inner: _mm_mul_ps(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn div_componentwise(&self, rhs: Fvec4) -> Fvec4 {
        unsafe {
            Fvec4 {
                inner: _mm_div_ps(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn min_componentwise(&self, rhs: Fvec4) -> Fvec4 {
        unsafe {
            Fvec4 {
                inner: _mm_min_ps(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn max_componentwise(&self, rhs: Fvec4) -> Fvec4 {
        unsafe {
            Fvec4 {
                inner: _mm_max_ps(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn floor(&self) -> Fvec4 {
        unsafe {
            Fvec4 {
                inner: _mm_floor_ps(self.inner),
            }
        }
    }

    #[inline]
    fn min_reduce(&self) -> f32 {
        unsafe {
            let reduce64 = _mm_min_ps(self.inner, _mm_permute_ps::<0b_11_10>(self.inner));
            let reduce32 = _mm_min_ss(reduce64, _mm_permute_ps::<1>(reduce64));
            _mm_cvtss_f32(reduce32)
        }
    }

    #[inline]
    fn max_reduce(&self) -> f32 {
        unsafe {
            let reduce64 = _mm_max_ps(self.inner, _mm_permute_ps::<0b_11_10>(self.inner));
            let reduce32 = _mm_max_ss(reduce64, _mm_permute_ps::<1>(reduce64));
            _mm_cvtss_f32(reduce32)
        }
    }

    #[inline]
    fn eq_reduce(&self, rhs: Fvec4) -> bool {
        unsafe {
            let mask = _mm_cmpeq_ps(self.inner, rhs.inner);
            let reduce = _mm_movemask_epi8(std::mem::transmute(mask));
            reduce == 0xffff
        }
    }

    #[inline]
    fn dot(&self, rhs: Fvec4) -> f32 {
        unsafe {
            let prod = _mm_mul_ps(self.inner, rhs.inner);
            let reduce64 = _mm_add_ps(prod, _mm_permute_ps::<0b_11_10>(prod));
            let reduce32 = _mm_add_ss(reduce64, _mm_permute_ps::<1>(reduce64));
            _mm_cvtss_f32(reduce32)
        }
    }

    #[inline]
    fn cross(&self, rhs: Fvec4) -> Self {
        unsafe {
            // Permutation (1, 2, 0, 3) = 0b_11_00_10_01
            let left = _mm_mul_ps(self.inner, _mm_permute_ps::<0b_11_00_10_01>(rhs.inner));
            let right = _mm_mul_ps(rhs.inner, _mm_permute_ps::<0b_11_00_10_01>(self.inner));
            let result = _mm_permute_ps::<0b_11_00_10_01>(_mm_sub_ps(left, right));
            Fvec4 { inner: result }
        }
    }
}

implement_vecops!(Fvec4, f32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq_works() {
        let a = Fvec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Fvec4::new(1.0, 2.0, 3.0, 4.0);
        let c = Fvec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a == b, true);
        assert_eq!(b == a, true);
        assert_eq!(a == a, true);
        assert_eq!(a == c, false);

        let d = Fvec4::new(0.0, -0.0, 0.0, -0.0);
        let e = Fvec4::new(0.0, 0.0, -0.0, -0.0);
        assert_eq!(d == e, true);

        let f = Fvec4::new(f32::NAN, f32::NAN, f32::NAN, f32::NAN);
        assert_eq!(f == f, false);
    }
}
