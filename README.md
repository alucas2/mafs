 # A Tiny SIMD Vector Crate 🏹

 🚩 **Works only on the x86_64 CPU architecture with the AVX2 and FMA extensions!**

 They can be enabled by putting these lines inside `.cargo/config.toml`, located either at the root of your
 project or in the installation directory of cargo:
 ```toml
 [build]
 rustflags = ["-Ctarget-feature=+avx2,+fma"]
 ```

 ## Highlights

 - The motivation behind this crate is to provide fast vectors for small computer graphics projects.
 - Straightforward implementation without macro trickery. Explore the source code to see how everything is done!

 ## Data types

 - Double precision:
     - `Dvec2` - 2D vector
     - `Dvec4` - 4D vector
     - `Dmat4` - 4x4 matrix
 - Single precision:
     - `Fvec2` - 2D vector (this one is not SIMD)
     - `Fvec4` - 4D vetcor
     - `Fmat4` - 4x4 matrix

 ## Available operations

 - Arithmetics (`+`, `-`, `*` and `/`):
     - Add, subtract, multiply or divide two vectors, or a vector with a scalar, componentwise.
     - Add or subtract two matrices.
     - Multiply two matrices.
     - Multiply a matrix by a vector.
 - Methods:
     - Operations on one vector: componentwise floor
     - Operations on two vectors: dot product, cross product, componentwise minimum and maximum.
     - Reduce a single vector: minimun and maximum across all components.
     - Invert a matrix that belongs to SE(3), i.e. a transformation matrix.
     - Transpose a matrix.

 ## Crate features

 - Enable the crate feature `bytemuck` to mark all vectors as *Plain Old Data*.