# parenchyma

[![Join the chat](https://badges.gitter.im/lychee-eng/parenchyma.svg)](https://gitter.im/lychee-eng/parenchyma)
[![](https://tokei.rs/b1/github/lychee-eng/parenchyma)](https://github.com/lychee-eng/parenchyma)

Parenchyma started off as a hard fork of [Collenchyma][collenchyma-repo] (hence the name), an 
extensible HPC framework developed by the [Autumn team] as well as an amazing group 
of [contributors][collenchyma-contributors]. Aside from the name and design, the two libraries are 
almost completely different (e.g., auto-sync thanks to [@alexandermorozov](/../../issues/2)). 
Therefore, before migrating over, one should go through the documentation carefully as to not make 
the mistake of misusing the framework. Not doing so may result in unintended behavior for 
which Parenchyma devs/contributors are not responsible.

## Usage

```toml
[dependencies]
parenchyma = "0.1.0"
parenchyma-native = "0.1.0"
```

```rust
extern crate parenchyma;
extern crate parenchyma_native;

use parenchyma::Backend;
use parenchyma_native::Native;

let backend: Backend<Native> = Backend::default()?;
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
