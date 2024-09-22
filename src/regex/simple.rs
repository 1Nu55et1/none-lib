pub struct SimpleRegex {
    pattern: String,
}

impl SimpleRegex {
    pub fn new(pattern: &str) -> Self {
        SimpleRegex {
            pattern: pattern.to_string(),
        }
    }

    pub fn matches(&self, text: &str) -> bool {
        text.contains(&self.pattern)
    }

    pub fn find(&self, text: &str) -> Option<(usize, usize)> {
        text.find(&self.pattern).map(|start| (start, start + self.pattern.len()))
    }

    pub fn replace(&self, text: &str, replacement: &str) -> String {
        text.replace(&self.pattern, replacement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matches() {
        let regex = SimpleRegex::new("hello");
        assert!(regex.matches("hello world"));
        assert!(!regex.matches("world"));
    }

    #[test]
    fn test_find() {
        let regex = SimpleRegex::new("world");
        assert_eq!(regex.find("hello world"), Some((6, 11)));
        assert_eq!(regex.find("hello"), None);
    }

    #[test]
    fn test_replace() {
        let regex = SimpleRegex::new("world");
        assert_eq!(regex.replace("hello world", "universe"), "hello universe");
    }
}
