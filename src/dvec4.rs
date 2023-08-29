use crate::Vec4;
use std::arch::x86_64::*;

/// 4D vector with double precision
///
/// The components are laid out in this order: `[x, y, z, w]`. This struct is aligned to 32 bytes.
///
/// ## Examples
///
/// ```
/// use mafs::{Vec4, Dvec4};
///
/// // Construction
/// let a = Dvec4::new(2.0, 3.0, 5.0, 6.0);
/// let b = Dvec4::new(6.0, 9.0, 2.5, 3.0);
/// let c = Dvec4::splat(0.0); // Set all four components to the same value
///
/// let p = Dvec4::point(1.0, 2.0, 3.0);
/// assert_eq!(p[3], 1.0); // Fourth component of a point is one
///
/// let d = Dvec4::direction(1.0, 2.0, 3.0);
/// assert_eq!(d[3], 0.0); // Fourth component of a direction is zero
///
/// // Arithmetics
/// assert_eq!(a + b, Dvec4::new(8.0, 12.0, 7.5, 9.0));
/// assert_eq!(a - b, Dvec4::new(-4.0, -6.0, 2.5, 3.0));
/// assert_eq!(a * b, Dvec4::new(12.0, 27.0, 12.5, 18.0));
/// assert_eq!(b / a, Dvec4::new(3.0, 3.0, 0.5, 0.5));
///
/// // Euclidian norm
/// assert_eq!(a.norm(), 74.0f64.sqrt());
/// assert_eq!(a.normalize().norm(), 1.0); // hmmmm
///
/// // Specialized operations
/// assert_eq!(a.dot(b), 69.5);
/// assert_eq!(b.dot(a), a.dot(b));
/// assert_eq!(a.cross(b), Dvec4::new(-37.5, 25.0, 0.0, 0.0));
/// assert_eq!(b.cross(a), -a.cross(b));
/// assert_eq!(Dvec4::new(-0.5, 0.5, 2.9, 0.0).floor(), Dvec4::new(-1.0, 0.0, 2.0, 0.0));
///
/// // Comparisons
/// assert_eq!(a.min_componentwise(b), Dvec4::new(2.0, 3.0, 2.5, 3.0));
/// assert_eq!(a.max_componentwise(b), Dvec4::new(6.0, 9.0, 5.0, 6.0));
///
/// // Reduction
/// assert_eq!(a.min_reduce(), 2.0);
/// assert_eq!(b.max_reduce(), 9.0);
/// ```
#[repr(C)]
#[derive(Copy, Clone)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable, bytemuck::Pod))]
pub struct Dvec4 {
    pub(crate) inner: __m256d,
}

impl std::fmt::Debug for Dvec4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_array().fmt(f)
    }
}

impl Vec4<f64> for Dvec4 {
    #[inline]
    fn new(x: f64, y: f64, z: f64, w: f64) -> Dvec4 {
        unsafe {
            Dvec4 {
                inner: _mm256_set_pd(w, z, y, x),
            }
        }
    }

    #[inline]
    fn as_array(&self) -> &[f64; 4] {
        unsafe { &*(self as *const Dvec4 as *const [f64; 4]) }
    }

    #[inline]
    fn as_mut_array(&mut self) -> &mut [f64; 4] {
        unsafe { &mut *(self as *mut Dvec4 as *mut [f64; 4]) }
    }

    #[inline]
    fn add_componentwise(&self, rhs: Dvec4) -> Dvec4 {
        unsafe {
            Dvec4 {
                inner: _mm256_add_pd(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn sub_componentwise(&self, rhs: Dvec4) -> Dvec4 {
        unsafe {
            Dvec4 {
                inner: _mm256_sub_pd(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn mul_componentwise(&self, rhs: Dvec4) -> Dvec4 {
        unsafe {
            Dvec4 {
                inner: _mm256_mul_pd(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn div_componentwise(&self, rhs: Dvec4) -> Dvec4 {
        unsafe {
            Dvec4 {
                inner: _mm256_div_pd(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn min_componentwise(&self, rhs: Dvec4) -> Dvec4 {
        unsafe {
            Dvec4 {
                inner: _mm256_min_pd(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn max_componentwise(&self, rhs: Dvec4) -> Dvec4 {
        unsafe {
            Dvec4 {
                inner: _mm256_max_pd(self.inner, rhs.inner),
            }
        }
    }

    #[inline]
    fn floor(&self) -> Dvec4 {
        unsafe {
            Dvec4 {
                inner: _mm256_floor_pd(self.inner),
            }
        }
    }

    #[inline]
    fn min_reduce(&self) -> f64 {
        unsafe {
            let reduce128 = _mm_min_pd(
                _mm256_castpd256_pd128(self.inner),
                _mm256_extractf128_pd::<1>(self.inner),
            );
            let reduce64 = _mm_min_sd(reduce128, _mm_permute_pd::<1>(reduce128));
            _mm_cvtsd_f64(reduce64)
        }
    }

    #[inline]
    fn max_reduce(&self) -> f64 {
        unsafe {
            let reduce128 = _mm_max_pd(
                _mm256_castpd256_pd128(self.inner),
                _mm256_extractf128_pd::<1>(self.inner),
            );
            let reduce64 = _mm_max_sd(reduce128, _mm_permute_pd::<1>(reduce128));
            _mm_cvtsd_f64(reduce64)
        }
    }

    #[inline]
    fn eq_reduce(&self, rhs: Dvec4) -> bool {
        unsafe {
            let mask = _mm256_cmp_pd::<_CMP_EQ_OQ>(self.inner, rhs.inner);
            let reduce = _mm256_movemask_epi8(std::mem::transmute(mask));
            reduce as u32 == 0xffffffff
        }
    }

    #[inline]
    fn dot(&self, rhs: Dvec4) -> f64 {
        unsafe {
            let prod = _mm256_mul_pd(self.inner, rhs.inner);
            let reduce128 = _mm_add_pd(
                _mm256_castpd256_pd128(prod),
                _mm256_extractf128_pd::<1>(prod),
            );
            let reduce64 = _mm_add_sd(reduce128, _mm_permute_pd::<1>(reduce128));
            _mm_cvtsd_f64(reduce64)
        }
    }

    #[inline]
    fn cross(&self, rhs: Dvec4) -> Dvec4 {
        unsafe {
            // Permutation (1, 2, 0, 3) = 0b_11_00_10_01
            let left = _mm256_mul_pd(
                self.inner,
                _mm256_permute4x64_pd::<0b_11_00_10_01>(rhs.inner),
            );
            let right = _mm256_mul_pd(
                rhs.inner,
                _mm256_permute4x64_pd::<0b_11_00_10_01>(self.inner),
            );
            let result = _mm256_permute4x64_pd::<0b_11_00_10_01>(_mm256_sub_pd(left, right));
            Dvec4 { inner: result }
        }
    }
}

implement_scalarops!(Dvec4, f64);
implement_vecops!(Dvec4, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq_works() {
        let a = Dvec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Dvec4::new(1.0, 2.0, 3.0, 4.0);
        let c = Dvec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(a == b, true);
        assert_eq!(b == a, true);
        assert_eq!(a == a, true);
        assert_eq!(a == c, false);

        let d = Dvec4::new(0.0, -0.0, 0.0, -0.0);
        let e = Dvec4::new(0.0, 0.0, -0.0, -0.0);
        assert_eq!(d == e, true);

        let f = Dvec4::new(f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        assert_eq!(f == f, false);
    }
}
