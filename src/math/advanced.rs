use std::f64::consts::PI;

pub fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

pub fn fibonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 | 2 => 1,
        _ => {
            let mut a = 0;
            let mut b = 1;
            for _ in 2..=n {
                let temp = a + b;
                a = b;
                b = temp;
            }
            b
        }
    }
}

pub fn sin(x: f64) -> f64 {
    x.sin()
}

pub fn cos(x: f64) -> f64 {
    x.cos()
}

pub fn tan(x: f64) -> f64 {
    x.tan()
}

pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

pub fn radians_to_degrees(radians: f64) -> f64 {
    radians * 180.0 / PI
}

pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        0
    } else {
        (a * b) / gcd(a, b)
    }
}

#[derive(Debug, PartialEq)]
pub struct Fraction {
    numerator: i64,
    denominator: i64,
}

impl Fraction {
    pub fn new(numerator: i64, denominator: i64) -> Result<Self, &'static str> {
        if denominator == 0 {
            Err("El denominador no puede ser cero")
        } else {
            let mut frac = Fraction { numerator, denominator };
            frac.simplify();
            Ok(frac)
        }
    }

    pub fn simplify(&mut self) {
        let gcd = gcd(self.numerator.abs() as u64, self.denominator.abs() as u64) as i64;
        self.numerator /= gcd;
        self.denominator /= gcd;
        if self.denominator < 0 {
            self.numerator = -self.numerator;
            self.denominator = -self.denominator;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(10), 55);
    }

    #[test]
    fn test_trigonometric() {
        let x = PI / 4.0;
        assert!((sin(x) - 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);
        assert!((cos(x) - 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);
        assert!((tan(x) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_angle_conversion() {
        assert!((degrees_to_radians(180.0) - PI).abs() < 1e-10);
        assert!((radians_to_degrees(PI) - 180.0).abs() < 1e-10);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(100, 75), 25);
    }

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(4, 6), 12);
        assert_eq!(lcm(21, 6), 42);
    }

    #[test]
    fn test_fraction() {
        let frac = Fraction::new(4, 6).unwrap();
        assert_eq!(frac, Fraction { numerator: 2, denominator: 3 });

        assert!(Fraction::new(1, 0).is_err());
    }
}
