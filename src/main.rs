#![allow(warnings)]

extern crate futures;

use futures::{future, Future};

pub trait Foo {

    fn connect(& self) -> Box<Future<Item=(), Error=()> + 'static> {
        Box::new(future::ok::<_, ()>(()))
    }
    fn disconnect(& self) -> Box<Future<Item=(), Error=()> + 'static>;
    fn bar(& mut self) {}
}

fn main() {
    let foo1: &mut Foo = unimplemented!();
    let foo2: &mut Foo = unimplemented!();

    let _ = foo1.connect().join(foo2.connect()).and_then(|_| {
        future::ok::<_, ()>(foo1.bar())
            //.and_then(move |_| foo1.disconnect())
    });
}