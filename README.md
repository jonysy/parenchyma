# parenchyma

Parenchyma is a hard fork of [Collenchyma][collenchyma-repo], an extendable HPC-Framework for 
CUDA, OpenCL and common CPU.

## Usage

```toml
[dependencies]
parenchyma = "0.1.0"
```

```rust
extern crate parenchyma as chyma;

use chyma::api::*;
use chyma::frameworks::*;

fn main() {
	// Construct a new framework.
	let framework = Native::new();

	// Available devices can be obtained through the framework.
	let select = |devices: &[_]| devices.to_vec();

	// Create a ready to go `Backend` from the framework.
	let backend = Backend::new(framework, select).expect("Something went wrong!");
}
```

[collenchyma-repo]: https://github.com/autumnai/collenchyma