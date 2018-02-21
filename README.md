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
to [@alexandermorozov](/../../issues/2)), async transfers, the fallback mechanism, etc.). Therefore, before migrating 
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

### Tensor creation

The easiest way to create a tensor is to use the `array` macro: 

```rust
#[macro_use(array)]
extern crate parenchyma;

use parenchyma::SharedTensor;

let c: SharedTensor<i32> = array![
    [
        [1,2,3],
        [4,5,6]
    ],
    [
        [11,22,33],
        [44,55,66]
    ],
    [
        [111,222,333],
        [444,555,666]
    ],
    [
        [1111,2222,3333],
        [4444,5555,6666]
    ]
].into_tensor();

let t = c.tensor_ref();

println!("{:?}", t);

// shape=[4, 2, 3], strides=[6, 3, 1], layout=C (0x1), type=i32
//
// [[[1, 2, 3],
//   [4, 5, 6]],
//  [[11, 22, 33],
//   [44, 55, 66]],
//  [[111, 222, 333],
//   [444, 555, 666]],
//  [[1111, 2222, 3333],
//   [4444, 5555, 6666]]]
```

### Synchronizing data across multiple `Device`s and `Backend`s

```rust
#[macro_use(array)]
extern crate parenchyma;

use parenchyma::prelude::*;

let ref cuda: Backend = {
    let framework = Cuda::new()?;
    let hardware = framework.available_hardware();
    Backend::with(framework, hardware)?
};

let t = array![[1.5, 2.3, 3.7], [4.8, 5.2, 6.9]].into_tensor();

t.synchronize(cuda)?;
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