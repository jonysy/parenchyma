# parenchyma

[![Join the chat](https://badges.gitter.im/lychee-eng/parenchyma.svg)](https://gitter.im/lychee-eng/parenchyma)
![Project Status](https://img.shields.io/badge/status-pre--alpha-green.svg)
[![](http://meritbadge.herokuapp.com/parenchyma)](https://crates.io/crates/parenchyma)
[![License](https://img.shields.io/crates/l/parenchyma.svg)](#license)
[![parenchyma](https://docs.rs/parenchyma/badge.svg)](https://docs.rs/parenchyma)

Parenchyma started off as a hard fork of [Collenchyma][collenchyma-repo] (hence the name), an 
extensible HPC framework developed by the [Autumn team] as well as an amazing group 
of [contributors][collenchyma-contributors]. Aside from the name and overall design, the two 
libraries are quite dissimilar to each other (e.g., auto-sync (thanks 
to [@alexandermorozov](/../../issues/2)) and the fallback mechanism). Therefore, before migrating 
over, one should go through the documentation carefully as to not make the mistake of misusing 
the framework. Not doing so may result in unintended behavior for which Parenchyma 
developers/contributors are not responsible.

Many of the original comments used for documentation purposes remain in the code base along with 
a few necessary additions/modifications.

> Disclaimer: Parenchyma is currently undergoing extensive refactoring and improvement. Therefore, 
> it is likely that many of the features available in the original Collenchyma project may not yet 
> be available in the Parenchyma project. It is also likely that certain features may never be 
> available in the Parenchyma project, as the different approaches that are currently being 
> considered may prove to be better than the original approach.

## Example

Parenchyma comes without any extension packages. The following example therefore assumes that
you have add both `parenchyma` and the Parenchyma ExtensionPackage `parenchyma-nn` to your
Cargo manifest.

```rust
extern crate parenchyma as pa;
extern crate parenchyma_nn as pann;

use pa::{Backend, Native, OpenCL, SharedTensor};

fn main() {
    let ref native: Backend = Backend::new::<Native>().unwrap();
    // Initialize an OpenCL or CUDA backend packaged with the NN extension.
    let ref backend = pann::Backend::new::<OpenCL>().unwrap();

    // Initialize two `SharedTensor`s.
    let shape = 1;
    let ref x = SharedTensor::<f32>::with(backend, shape, vec![3.5]).unwrap();
    let ref mut result = SharedTensor::<f32>::new(shape);

    // Run the sigmoid operation, provided by the NN extension, on 
    // your OpenCL/CUDA enabled GPU (or CPU, which is possible through OpenCL)
    backend.sigmoid(x, result).unwrap();

    // Print the result: `[0.97068775] shape=[1], strides=[1]`
    println!("{:?}", result.read(native).unwrap().as_native().unwrap());
}
```

## License

Dual licensed under
  * Apache License, Version 2.0 ([LICENSE-APACHE] or http://www.apache.org/licenses/LICENSE-2.0)
  * MIT license ([LICENSE-MIT] or http://opensource.org/licenses/MIT)

[Autumn team]: https://github.com/autumnai
[collenchyma-repo]: https://github.com/autumnai/collenchyma
[collenchyma-contributors]: https://github.com/autumnai/collenchyma/graphs/contributors
[LICENSE-APACHE]: ../../../license/blob/master/LICENSE-APACHE
[LICENSE-MIT]: ../../../license/blob/master/LICENSE-MIT
