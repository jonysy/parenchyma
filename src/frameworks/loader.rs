#[cfg(unix)]
pub type Sym<T> = ::lib::os::unix::Symbol<T>;

#[cfg(windows)]
pub type Sym<T> = ::lib::os::windows::Symbol<T>;

// borrowing: 
// https://www.reddit.com/r/rust/comments/3hrzsk/optionalruntime_loading_of_system_libraries/cuadtbc/
macro_rules! dynamic_extern {
    (@as_item $i:item) => {$i};
    
    (#[link=$library:tt] extern $linkage:tt {$(

        pub fn $function_name:ident($($argument_name:ident: $argument_type:ty),*) $(-> $ret_ty:ty)*;
    )*}) => {$(

        dynamic_extern! {
            @as_item
            pub unsafe fn $function_name($($argument_name: $argument_type),*) $(-> $ret_ty)* {
                #![allow(dead_code)]
                type FnPtr = unsafe extern $linkage fn($($argument_type),*) $(-> $ret_ty)*;

                lazy_static! {
                    static ref FN_PTR: ::frameworks::loader::Sym<FnPtr> = {
                        unsafe {
                            use lib::Library;
                            use std::path::Path;

                            let lib = Library::new(Path::new($library)).unwrap();
                            let name = stringify!($function_name);
                            let bytes = name.as_bytes();
                            let ptr = lib.get::<FnPtr>(bytes).unwrap();
                            ptr.into_raw()
                        }
                    };
                }

                (*FN_PTR)($($argument_name),*)
            }
        }
    )*}
}