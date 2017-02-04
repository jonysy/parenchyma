# parenchyma

[![Join the chat at https://gitter.im/lychee-eng/parenchyma](https://badges.gitter.im/lychee-eng/parenchyma.svg)](https://gitter.im/lychee-eng/parenchyma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Parenchyma is a hard fork of [Collenchyma][collenchyma-repo], an extendable HPC-Framework originally 
developed by the [Autumn team] as well as an [amazing group of contributors][collenchyma-contributors].

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

fn main() {
	let backend: Backend<Native> = Backend::default().expect("something went wrong!");
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
