pub fn add(a: f64, b: f64) -> f64 {
    a + b
}

pub fn substract(a: f64, b: f64) -> f64 {
    a - b
}

pub fn multiplication(a: f64, b: f64) -> f64 {
    a * b
}

pub fn division(a: f64, b: f64) -> Result<f64, &'static str> {
    if b == 0.0 {
        Err("No se puede dividir por cero")
    } else {
        Ok(a / b)
    }
}

pub fn power(base: f64, exponente: i32) -> f64 {
    base.powi(exponente)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2.0, 3.0), 5.0);
    }

    #[test]
    fn test_substract() {
        assert_eq!(substract(5.0, 3.0), 2.0);
    }

    #[test]
    fn test_multiplication() {
        assert_eq!(multiplication(2.0, 3.0), 6.0);
    }

    #[test]
    fn test_division() {
        assert_eq!(division(6.0, 2.0), Ok(3.0));
        assert_eq!(division(1.0, 0.0), Err("No se puede dividir por cero"));
    }

    #[test]
    fn test_power() {
        assert_eq!(power(2.0, 3), 8.0);
    }
}
