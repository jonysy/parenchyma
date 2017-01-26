use chyma::api::{Backend, Framework};
use chyma::frameworks::Native;

#[test]
fn it_can_create_default_backend() {
	assert!(Backend::new(Native::new(), |devices| devices.to_vec()).is_ok());
}