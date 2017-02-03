# parenchyma

Parenchyma is a hard fork of [Collenchyma][collenchyma-repo], an extendable HPC-Framework originally 
developed by the [Autumnai team], as well as an [amazing group of contributors][collenchyma-contributors].

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
	let backend = Backend::<Native>::default().expect("Something went wrong!");
}
```

[Autumnai team]: https://github.com/autumnai
[collenchyma-repo]: https://github.com/autumnai/collenchyma
[collenchyma-contributors]: https://github.com/autumnai/collenchyma/graphs/contributors