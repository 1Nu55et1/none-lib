#[cfg(feature = "math")]
pub mod math;

#[cfg(feature = "regex")]
pub mod regex;

#[cfg(feature = "graphics")]
pub mod graphics;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
