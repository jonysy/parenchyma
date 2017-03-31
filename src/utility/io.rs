//! Useful functions for writing and reading data to and from tensors

use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

// //     // let mut a = [1.3, 4.5, 6.7, 3.4];
// //     // let b     = [1.3, 4.5, 6.6, 3.4];

// //     // chyma::io::write(x, .., &b)?;
// //     // chyma::io::write_on(backend, x, .., &b)?;
// pub fn write<T, R>(x: &mut [T], x_range: R, data: &[T]) where T: Copy, R: RangeArgument {
//     let length = x.len();
//     let start = x_range.start().unwrap_or(0);
//     let end = x_range.end().unwrap_or(length);

//     assert!(end >= start);

//     let skip = start;
//     let take = end - start;

//     assert_eq!(take, data.len());

//     for (x_datum, &datum) in x.iter_mut().skip(skip).take(take).zip(data) {
//         *x_datum = datum;
//     }
// }

pub trait RangeArgument {

    fn start(&self) -> Option<usize> {
        None
    }

    fn end(&self) -> Option<usize> {
        None
    }
}

impl RangeArgument for Range<usize> {

    fn start(&self) -> Option<usize> {
        Some(self.start)
    }

    fn end(&self) -> Option<usize> {
        Some(self.end)
    }
}

impl RangeArgument for RangeFrom<usize> {

    fn start(&self) -> Option<usize> {
        Some(self.start)
    }
}

impl RangeArgument for RangeTo<usize> {

    fn end(&self) -> Option<usize> {
        Some(self.end)
    }
}

impl RangeArgument for RangeFull { }