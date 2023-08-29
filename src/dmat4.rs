use crate::{Dvec4, Mat4};
use std::arch::x86_64::*;

/// 4x4 matrix with double precision
///
/// It has the same layout as `[Dvec4; 4]`, so it is aligned to 32 bytes.
///
/// ## Examples
///
/// ```
/// use mafs::{Mat4, Dmat4, Vec4, Dvec4};
///
/// // Construction
/// let m1 = Dmat4::from_columns(
///     Dvec4::new(1.0, 2.0, 3.0, 4.0),
///     Dvec4::new(5.0, 6.0, 7.0, 8.0),
///     Dvec4::new(9.0, 10.0, 11.0, 12.0),
///     Dvec4::new(13.0, 14.0, 15.0, 16.0),
/// );
/// let m2 = Dmat4::from_columns(
///     Dvec4::new(17.0, 18.0, 19.0, 20.0),
///     Dvec4::new(21.0, 22.0, 23.0, 24.0),
///     Dvec4::new(25.0, 26.0, 27.0, 28.0),
///     Dvec4::new(29.0, 30.0, 31.0, 32.0),
/// );
///
/// // Matrix-Vector arithmetics
/// let v = Dvec4::new(17.0, 18.0, 19.0, 20.0);
/// assert_eq!(m1 * v, Dvec4::new(538.0, 612.0, 686.0, 760.0));
///
/// // Matrix-Matrix arithmetics
/// assert_eq!(m1 + m2, Dmat4::from_columns(
///     Dvec4::new(18.0, 20.0, 22.0, 24.0),
///     Dvec4::new(26.0, 28.0, 30.0, 32.0),
///     Dvec4::new(34.0, 36.0, 38.0, 40.0),
///     Dvec4::new(42.0, 44.0, 46.0, 48.0),
/// ));
/// assert_eq!(m1 - m2, Dmat4::from_columns(
///     Dvec4::new(-16.0, -16.0, -16.0, -16.0),
///     Dvec4::new(-16.0, -16.0, -16.0, -16.0),
///     Dvec4::new(-16.0, -16.0, -16.0, -16.0),
///     Dvec4::new(-16.0, -16.0, -16.0, -16.0),
/// ));
/// assert_eq!(m1 * m2, Dmat4::from_columns(
///     Dvec4::new(538.0, 612.0, 686.0, 760.0),
///     Dvec4::new(650.0, 740.0, 830.0, 920.0),
///     Dvec4::new(762.0, 868.0, 974.0, 1080.0),
///     Dvec4::new(874.0, 996.0, 1118.0, 1240.0),
/// ));
///
/// // Transpose
/// assert_eq!(m1.transpose(), Dmat4::from_columns(
///     Dvec4::new(1.0, 5.0, 9.0, 13.0),
///     Dvec4::new(2.0, 6.0, 10.0, 14.0),
///     Dvec4::new(3.0, 7.0, 11.0, 15.0),
///     Dvec4::new(4.0, 8.0, 12.0, 16.0),
/// ));
///
/// // Inverse of transformation matrices
/// let rotation_matrix = Dmat4::from_columns(
///     Dvec4::new(1.0,          0.0,           0.0, 0.0),
///     Dvec4::new(0.0, 1.0f64.cos(), -1.0f64.sin(), 0.0),
///     Dvec4::new(0.0, 1.0f64.sin(),  1.0f64.cos(), 0.0),
///     Dvec4::new(0.0,          0.0,           0.0, 1.0),
/// );
/// assert_eq!(rotation_matrix.inverse_se3(), rotation_matrix.transpose());
/// let rotation_and_translation = Dmat4::from_columns(
///     Dvec4::new( 0.6666666666666666,  0.6666666666666666, -0.3333333333333333, 0.0),
///     Dvec4::new(-0.3333333333333333,  0.6666666666666666,  0.6666666666666666, 0.0),
///     Dvec4::new( 0.6666666666666666, -0.3333333333333333,  0.6666666666666666, 0.0),
///     Dvec4::new(               -4.0,                 5.0,                 6.0, 1.0),
/// );
/// assert_eq!(rotation_and_translation.inverse_se3(), Dmat4::from_columns(
///     Dvec4::new( 0.6666666666666666, -0.3333333333333333,  0.6666666666666666, 0.0),
///     Dvec4::new( 0.6666666666666666,  0.6666666666666666, -0.3333333333333333, 0.0),
///     Dvec4::new(-0.3333333333333333,  0.6666666666666666,  0.6666666666666666, 0.0),
///     Dvec4::new( 1.3333333333333333,  -8.666666666666666, 0.33333333333333326, 1.0),
/// ));
/// ```
#[repr(C)]
#[derive(Copy, Clone, Default, PartialEq)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Zeroable, bytemuck::Pod))]
pub struct Dmat4 {
    pub(crate) inner: [Dvec4; 4],
}

impl std::fmt::Debug for Dmat4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.as_array().fmt(f)
    }
}

impl Mat4<f64, Dvec4> for Dmat4 {
    #[inline]
    fn from_columns(x: Dvec4, y: Dvec4, z: Dvec4, w: Dvec4) -> Dmat4 {
        Dmat4 {
            inner: [x, y, z, w],
        }
    }

    #[inline]
    fn as_array(&self) -> &[Dvec4; 4] {
        &self.inner
    }

    #[inline]
    fn as_mut_array(&mut self) -> &mut [Dvec4; 4] {
        &mut self.inner
    }

    #[inline]
    fn mul_vector(&self, rhs: Dvec4) -> Dvec4 {
        unsafe {
            let mut result = _mm256_mul_pd(
                self.inner[0].inner,
                _mm256_permute4x64_pd::<0b_00_00_00_00>(rhs.inner),
            );
            result = _mm256_fmadd_pd(
                self.inner[1].inner,
                _mm256_permute4x64_pd::<0b_01_01_01_01>(rhs.inner),
                result,
            );
            result = _mm256_fmadd_pd(
                self.inner[2].inner,
                _mm256_permute4x64_pd::<0b_10_10_10_10>(rhs.inner),
                result,
            );
            result = _mm256_fmadd_pd(
                self.inner[3].inner,
                _mm256_permute4x64_pd::<0b_11_11_11_11>(rhs.inner),
                result,
            );
            Dvec4 { inner: result }
        }
    }

    #[inline]
    fn transpose(&self) -> Dmat4 {
        unsafe {
            let c0 = _mm256_unpacklo_pd(self.inner[0].inner, self.inner[1].inner);
            let c1 = _mm256_unpackhi_pd(self.inner[0].inner, self.inner[1].inner);
            let c2 = _mm256_unpacklo_pd(self.inner[2].inner, self.inner[3].inner);
            let c3 = _mm256_unpackhi_pd(self.inner[2].inner, self.inner[3].inner);
            let d0 = _mm256_permute2f128_pd::<0b_00_10_00_00>(c0, c2);
            let d1 = _mm256_permute2f128_pd::<0b_00_10_00_00>(c1, c3);
            let d2 = _mm256_permute2f128_pd::<0b_00_11_00_01>(c0, c2);
            let d3 = _mm256_permute2f128_pd::<0b_00_11_00_01>(c1, c3);

            Dmat4::from_columns(
                Dvec4 { inner: d0 },
                Dvec4 { inner: d1 },
                Dvec4 { inner: d2 },
                Dvec4 { inner: d3 },
            )
        }
    }
}

implement_matops!(Dmat4, Dvec4, f64);
