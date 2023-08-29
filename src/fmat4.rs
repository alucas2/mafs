use crate::{Fvec4, Mat4};
use std::arch::x86_64::*;

/// 4x4 matrix with double precision
///
/// It has the same layout as `[Fvec4; 4]`, so it is aligned to 16 bytes.
///
/// ## Examples
///
/// ```
/// use mafs::{Mat4, Fmat4, Vec4, Fvec4};
///
/// // Construction
/// let m1 = Fmat4::from_columns(
///     Fvec4::new(1.0, 2.0, 3.0, 4.0),
///     Fvec4::new(5.0, 6.0, 7.0, 8.0),
///     Fvec4::new(9.0, 10.0, 11.0, 12.0),
///     Fvec4::new(13.0, 14.0, 15.0, 16.0),
/// );
/// let m2 = Fmat4::from_columns(
///     Fvec4::new(17.0, 18.0, 19.0, 20.0),
///     Fvec4::new(21.0, 22.0, 23.0, 24.0),
///     Fvec4::new(25.0, 26.0, 27.0, 28.0),
///     Fvec4::new(29.0, 30.0, 31.0, 32.0),
/// );
///
/// // Matrix-Vector arithmetics
/// let v = Fvec4::new(17.0, 18.0, 19.0, 20.0);
/// assert_eq!(m1 * v, Fvec4::new(538.0, 612.0, 686.0, 760.0));
///
/// // Matrix-Matrix arithmetics
/// assert_eq!(m1 + m2, Fmat4::from_columns(
///     Fvec4::new(18.0, 20.0, 22.0, 24.0),
///     Fvec4::new(26.0, 28.0, 30.0, 32.0),
///     Fvec4::new(34.0, 36.0, 38.0, 40.0),
///     Fvec4::new(42.0, 44.0, 46.0, 48.0),
/// ));
/// assert_eq!(m1 - m2, Fmat4::from_columns(
///     Fvec4::new(-16.0, -16.0, -16.0, -16.0),
///     Fvec4::new(-16.0, -16.0, -16.0, -16.0),
///     Fvec4::new(-16.0, -16.0, -16.0, -16.0),
///     Fvec4::new(-16.0, -16.0, -16.0, -16.0),
/// ));
/// assert_eq!(m1 * m2, Fmat4::from_columns(
///     Fvec4::new(538.0, 612.0, 686.0, 760.0),
///     Fvec4::new(650.0, 740.0, 830.0, 920.0),
///     Fvec4::new(762.0, 868.0, 974.0, 1080.0),
///     Fvec4::new(874.0, 996.0, 1118.0, 1240.0),
/// ));
///
/// // Transpose
/// assert_eq!(m1.transpose(), Fmat4::from_columns(
///     Fvec4::new(1.0, 5.0, 9.0, 13.0),
///     Fvec4::new(2.0, 6.0, 10.0, 14.0),
///     Fvec4::new(3.0, 7.0, 11.0, 15.0),
///     Fvec4::new(4.0, 8.0, 12.0, 16.0),
/// ));
///
/// // Inverse of transformation matrices
/// let rotation_matrix = Fmat4::from_columns(
///     Fvec4::new(1.0,          0.0,           0.0, 0.0),
///     Fvec4::new(0.0, 1.0f32.cos(), -1.0f32.sin(), 0.0),
///     Fvec4::new(0.0, 1.0f32.sin(),  1.0f32.cos(), 0.0),
///     Fvec4::new(0.0,          0.0,           0.0, 1.0),
/// );
/// assert_eq!(rotation_matrix.inverse_se3(), rotation_matrix.transpose());
/// let rotation_and_translation = Fmat4::from_columns(
///     Fvec4::new( 0.6666666666666666,  0.6666666666666666, -0.3333333333333333, 0.0),
///     Fvec4::new(-0.3333333333333333,  0.6666666666666666,  0.6666666666666666, 0.0),
///     Fvec4::new( 0.6666666666666666, -0.3333333333333333,  0.6666666666666666, 0.0),
///     Fvec4::new(               -4.0,                 5.0,                 6.0, 1.0),
/// );
/// assert_eq!(rotation_and_translation.inverse_se3(), Fmat4::from_columns(
///     Fvec4::new(0.6666667, -0.33333334, 0.6666667, 0.0),
///     Fvec4::new(0.6666667, 0.6666667, -0.33333334, 0.0),
///     Fvec4::new(-0.33333334, 0.6666667, 0.6666667, 0.0),
///     Fvec4::new(1.3333334, -8.666667, 0.33333337, 1.0),    
/// ));
/// ```
#[repr(C)]
#[derive(Copy, Clone, Default, PartialEq)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable, bytemuck::Pod))]
pub struct Fmat4 {
    pub(crate) inner: [Fvec4; 4],
}

impl std::fmt::Debug for Fmat4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_array().fmt(f)
    }
}

impl Mat4<f32, Fvec4> for Fmat4 {
    #[inline]
    fn from_columns(x: Fvec4, y: Fvec4, z: Fvec4, w: Fvec4) -> Fmat4 {
        Fmat4 {
            inner: [x, y, z, w],
        }
    }

    #[inline]
    fn as_array(&self) -> &[Fvec4; 4] {
        &self.inner
    }

    #[inline]
    fn as_mut_array(&mut self) -> &mut [Fvec4; 4] {
        &mut self.inner
    }

    #[inline]
    fn mul_vector(&self, rhs: Fvec4) -> Fvec4 {
        unsafe {
            let mut result = _mm_mul_ps(
                self.inner[0].inner,
                _mm_permute_ps::<0b_00_00_00_00>(rhs.inner),
            );
            result = _mm_fmadd_ps(
                self.inner[1].inner,
                _mm_permute_ps::<0b_01_01_01_01>(rhs.inner),
                result,
            );
            result = _mm_fmadd_ps(
                self.inner[2].inner,
                _mm_permute_ps::<0b_10_10_10_10>(rhs.inner),
                result,
            );
            result = _mm_fmadd_ps(
                self.inner[3].inner,
                _mm_permute_ps::<0b_11_11_11_11>(rhs.inner),
                result,
            );
            Fvec4 { inner: result }
        }
    }

    #[inline]
    fn transpose(&self) -> Fmat4 {
        unsafe {
            let c0 = _mm_unpacklo_ps(self.inner[0].inner, self.inner[1].inner);
            let c1 = _mm_unpackhi_ps(self.inner[0].inner, self.inner[1].inner);
            let c2 = _mm_unpacklo_ps(self.inner[2].inner, self.inner[3].inner);
            let c3 = _mm_unpackhi_ps(self.inner[2].inner, self.inner[3].inner);
            let d0 = _mm_movelh_ps(c0, c2);
            let d1 = _mm_movehl_ps(c2, c0);
            let d2 = _mm_movelh_ps(c1, c3);
            let d3 = _mm_movehl_ps(c3, c1);

            Fmat4::from_columns(
                Fvec4 { inner: d0 },
                Fvec4 { inner: d1 },
                Fvec4 { inner: d2 },
                Fvec4 { inner: d3 },
            )
        }
    }
}

implement_matops!(Fmat4, Fvec4, f32);
