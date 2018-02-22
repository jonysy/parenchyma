/// Different algorithms to compute the gradient with respect to the filter.
#[derive(Clone, Copy, Debug)]
pub enum ConvBackwardDataAlgo {
    /// Attempt to automatically find the best algorithm of all the other available ones.
    Auto,
    /// Compute the convolution as matrix product without forming the matrix that holds the input data.
    ///
    /// Does not need any memory workspace.
    ///
    /// The results are deterministic.
    ImplicitGemm,
    /// Compute the convolution as sum of matrix product without forming the matrix that holds the input data.
    ///
    /// Does not need any memory workspace.
    ///
    /// The results are non-deterministic.
    ImplicitGemmSum,
    /// Compute the convolution as Fast-Fourier Transform.
    ///
    /// Needs a significant memory workspace.
    ///
    /// The results are deterministic.
    Fft,
    /// Compute the convolution as Fast-Fourier Transform with 32x32 tiles.
    ///
    /// Needs a significant memory workspace.
    ///
    /// The results are deterministic.
    FftTiling,
}

/// Different algorithms to compute the gradient with respect to the filter.
#[derive(Clone, Copy, Debug)]
pub enum ConvBackwardFilterAlgo {
    /// Attempt to automatically find the best algorithm of all the other available ones.
    Auto,
    /// Compute the convolution as matrix product without forming the matrix that holds the input data.
    ///
    /// Does not need any memory workspace.
    ///
    /// The results are deterministic.
    ImplicitGemm,
    /// Compute the convolution as sum of matrix product without forming the matrix that holds the input data.
    ///
    /// Does not need any memory workspace.
    ///
    /// The results are non-deterministic.
    ImplicitGemmSum,
    /// Similar to `ImplicitGEMMSum` but needs some workspace to precompile the implicit indices.
    ///
    /// The results are non-deterministic.
    ImplicitPrecompiledGemmSum,
    /// Compute the convolution as Fast-Fourier Transform.
    ///
    /// Needs a significant memory workspace.
    ///
    /// The results are deterministic.
    Fft,
}

/// Different algorithms to compute the convolution forward algorithm.
#[derive(Clone, Copy, Debug)]
pub enum ConvForwardAlgo {
    /// Attempt to automatically find the best algorithm of all the other available ones.
    Auto,
    /// Compute the convolution as explicit matrix product.
    ///
    /// Needs a significant memory workspace.
    Gemm,
    /// Compute the convolution as matrix product without forming the matrix that holds the input data.
    ///
    /// Does not need any memory workspace.
    ImplicitGemm,
    /// Similar to `ImplicitGEMM` but needs some workspace to precompile the implicit indices.
    ImplicitPrecompiledGemm,
    /// Compute the convolution as Fast-Fourier Transform.
    ///
    /// Needs a significant memory workspace.
    Fft,
    /// Compute the convolution as Fast-Fourier Transform with 32x32 tiles.
    ///
    /// Needs a significant memory workspace.
    FftTiling,
    /// Compute the convolution without implicit or explicit matrix-multiplication. **Do not try to use this**.
    ///
    /// Listed in cuDNN docs but cuDNN does not provide a implementation.
    Direct,
}