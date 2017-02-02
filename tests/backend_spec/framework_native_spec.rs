use chyma::api::Framework;
use chyma::frameworks::Native;

#[test]
fn it_works() {
	assert_eq!(Native::new().devices().len(), 1);
}