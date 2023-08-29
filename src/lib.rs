//! # A Tiny SIMD Vector Crate 🏹
//!
//! 🚩 **Works only on the x86_64 CPU architecture with the AVX2 and FMA extensions!**
//!
//! They can be enabled by putting these lines inside `.cargo/config.toml`, located either at the root of your
//! project or in the installation directory of cargo:
//! ```toml
//! [build]
//! rustflags = ["-Ctarget-feature=+avx2,+fma"]
//! ```
//!
//! ## Highlights
//!
//! - The motivation behind this crate is to provide fast vectors for small computer graphics projects.
//! - Straightforward implementation without macro trickery. Explore the source code to see how everything is done!
//!
//! ## Data types
//!
//! - Double precision:
//!     - [`Dvec2`] - 2D vector
//!     - [`Dvec4`] - 4D vector
//!     - [`Dmat4`] - 4x4 matrix
//! - Single precision:
//!     - [`Fvec2`] - 2D vector (this one is not SIMD)
//!     - [`Fvec4`] - 4D vetcor
//!     - [`Fmat4`] - 4x4 matrix
//!
//! ## Available operations
//!
//! - Arithmetics (`+`, `-`, `*` and `/`):
//!     - Add, subtract, multiply or divide two vectors, or a vector with a scalar, componentwise.
//!     - Add or subtract two matrices.
//!     - Multiply two matrices.
//!     - Multiply a matrix by a vector.
//! - Methods:
//!     - Operations on one vector: componentwise floor
//!     - Operations on two vectors: dot product, cross product, componentwise minimum and maximum.
//!     - Reduce a single vector: minimun and maximum across all components.
//!     - Invert a matrix that belongs to SE(3), i.e. a transformation matrix.
//!     - Transpose a matrix.
//!
//! ## Crate features
//!
//! - Enable the crate feature `bytemuck` to mark all vectors as *Plain Old Data*.

#[macro_use]
mod private_macros;

#[cfg(not(any(
    doc,
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma",
    )
)))]
compile_error!(
    "
This crate only works on x86_64 and requires the following extensions: AVX2 and FMA.
They can be enabled by adding this in `config.toml`:

[build]
rustflags = [\"-Ctarget-feature=+avx2,+fma\"]
"
);

mod traits;
pub use traits::{Mat4, Vec2, Vec4};

mod dvec2;
pub use dvec2::*;

mod dvec4;
pub use dvec4::*;

mod dmat4;
pub use dmat4::*;

mod fvec4;
pub use fvec4::*;

mod fvec2;
pub use fvec2::*;

mod fmat4;
pub use fmat4::*;

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn sizes() {
        assert_eq!(size_of::<Fvec2>(), 8);
        assert_eq!(size_of::<Dvec2>(), 16);

        assert_eq!(size_of::<Fvec4>(), 16);
        assert_eq!(size_of::<Dvec4>(), 32);

        assert_eq!(size_of::<Fmat4>(), 64);
        assert_eq!(size_of::<Dmat4>(), 128);
    }

    #[test]
    fn aligns() {
        assert_eq!(align_of::<Fvec2>(), 4); // <- small exception here
        assert_eq!(align_of::<Dvec2>(), 16);

        assert_eq!(align_of::<Fvec4>(), 16);
        assert_eq!(align_of::<Dvec4>(), 32);

        assert_eq!(align_of::<Fmat4>(), 16);
        assert_eq!(align_of::<Dmat4>(), 32);
    }
}
