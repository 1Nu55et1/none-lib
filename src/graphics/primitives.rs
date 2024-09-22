#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Point { x, y }
    }

    pub fn distance(&self, other: &Point) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Color { r, g, b, a }
    }

    pub fn black() -> Self {
        Color::new(0, 0, 0, 255)
    }

    pub fn white() -> Self {
        Color::new(255, 255, 255, 255)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    pub start: Point,
    pub end: Point,
    pub color: Color,
}

impl Line {
    pub fn new(start: Point, end: Point, color: Color) -> Self {
        Line { start, end, color }
    }

    pub fn length(&self) -> f32 {
        self.start.distance(&self.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);
        assert_eq!(p1.distance(&p2), 5.0);
    }

    #[test]
    fn test_color() {
        let black = Color::black();
        assert_eq!(black, Color::new(0, 0, 0, 255));

        let white = Color::white();
        assert_eq!(white, Color::new(255, 255, 255, 255));
    }

    #[test]
    fn test_line() {
        let start = Point::new(0.0, 0.0);
        let end = Point::new(3.0, 4.0);
        let color = Color::black();
        let line = Line::new(start, end, color);
        assert_eq!(line.length(), 5.0);
    }
}
