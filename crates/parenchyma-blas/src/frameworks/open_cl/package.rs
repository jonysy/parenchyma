use ocl;
use std::ffi::CString;
use parenchyma::error::Result;
use parenchyma::frameworks::OpenCLContext;

// const WGS: usize = 64;
// const WGS1: usize = 64;
// const WGS2: usize = 64;

// /// Caches instances of `Kernel`
// #[derive(Debug)]
// pub struct OpenCLPackage {
//     pub(in super) program: ocl::Program,
//     asum: [ocl::Kernel; 2],
//     pub(in super) axpy: ocl::Kernel,
//     copy: ocl::Kernel,
//     dot: [ocl::Kernel; 2],
//     nrm2: [ocl::Kernel; 2],
//     scal: ocl::Kernel,
//     swap: ocl::Kernel,

//     gemm_direct: Gemm,
// }

// #[derive(Debug)]
// pub struct Gemm {
//     tt: ocl::Kernel,
//     tn: ocl::Kernel,
//     nt: ocl::Kernel,
//     nn: ocl::Kernel,
// }

/// Caches instances of `Kernel`
#[derive(Debug)]
pub struct OpenCLPackage {
    pub(in frameworks::open_cl) program: ocl::Program,
}

impl OpenCLPackage {
    pub fn compile(cx: &mut OpenCLContext<()>) -> Result<OpenCLPackage> {
        let program = cx.program(vec![
            CString::new(include_str!("source/common.cl")).unwrap(),

            CString::new(include_str!("source/level1/level1.cl")).unwrap(),
            CString::new(include_str!("source/level1/xasum.cl")).unwrap(),
            CString::new(include_str!("source/level1/xaxpy.cl")).unwrap(),
            CString::new(include_str!("source/level1/xcopy.cl")).unwrap(),
            CString::new(include_str!("source/level1/xdot.cl")).unwrap(),
            CString::new(include_str!("source/level1/xnrm2.cl")).unwrap(),
            CString::new(include_str!("source/level1/xscal.cl")).unwrap(),
            CString::new(include_str!("source/level1/xswap.cl")).unwrap(),

            CString::new(include_str!("source/level3/level3.cl")).unwrap(),
            CString::new(include_str!("source/level3/xgemm_direct_part1.cl")).unwrap(),
            CString::new(include_str!("source/level3/xgemm_direct_part2.cl")).unwrap(),
            CString::new(include_str!("source/level3/xgemm_direct_part3.cl")).unwrap(),
        ])?;

        // Ok(OpenCLPackage {
        //     asum: [ocl::Kernel::new("Xasum", &program)?, ocl::Kernel::new("XasumEpilogue", &program)?],
        //     axpy: ocl::Kernel::new("Xaxpy", &program)?, 
        //     copy: ocl::Kernel::new("Xcopy", &program)?, 
        //     dot: [ocl::Kernel::new("Xdot", &program)?, ocl::Kernel::new("XdotEpilogue", &program)?],
        //     nrm2: [ocl::Kernel::new("Xnrm2", &program)?, ocl::Kernel::new("Xnrm2Epilogue", &program)?],
        //     scal: ocl::Kernel::new("Xscal", &program)?,
        //     swap: ocl::Kernel::new("Xswap", &program)?,

        //     gemm_direct: Gemm {
        //         tt: ocl::Kernel::new("XgemmDirectTT", &program)?,
        //         tn: ocl::Kernel::new("XgemmDirectTN", &program)?,
        //         nt: ocl::Kernel::new("XgemmDirectNT", &program)?,
        //         nn: ocl::Kernel::new("XgemmDirectNN", &program)?,
        //     },

        //     program,
        // })

        Ok(OpenCLPackage { program })
    }
}