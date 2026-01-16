#[derive(Debug, Clone, PartialEq)]
pub struct AtToken {
    pub range: (usize, usize), // byte range in the input string
    pub query: String,
}

/// Find the active @ token at the cursor position
pub fn find_active_at_token(input: &str, cursor: usize) -> Option<AtToken> {
    if cursor > input.len() {
        return None;
    }

    // Find the start of the current token by looking backward from cursor
    let mut token_start = cursor;
    
    // Look backward to find the start of the current word/token
    while token_start > 0 {
        let prev_char = input.chars().nth(char_index_at_byte_pos(input, token_start - 1)?)?;
        if prev_char.is_whitespace() {
            break;
        }
        token_start -= prev_char.len_utf8();
    }
    
    // Check if we have an @ at the token start
    if token_start >= input.len() {
        return None;
    }
    
    let char_at_start = input.chars().nth(char_index_at_byte_pos(input, token_start)?)?;
    if char_at_start != '@' {
        return None;
    }
    
    // Make sure the @ is actually at a token boundary (not inside an email)
    if token_start > 0 {
        let prev_char = input.chars().nth(char_index_at_byte_pos(input, token_start - 1)?)?;
        if !prev_char.is_whitespace() && prev_char != '\n' && prev_char != '\t' {
            return None; // @ is not at word boundary, likely inside email
        }
    }
    
    // Find the end of the token
    let mut token_end = token_start + 1; // Skip the @
    while token_end < input.len() {
        let ch = input.chars().nth(char_index_at_byte_pos(input, token_end)?)?;
        if ch.is_whitespace() {
            break;
        }
        token_end += ch.len_utf8();
    }
    
    // Extract the query part (everything after @)
    let query_start = token_start + 1; // Skip the @
    let query = if query_start < token_end {
        input[query_start..token_end].to_string()
    } else {
        String::new()
    };
    
    Some(AtToken {
        range: (token_start, token_end),
        query,
    })
}

/// Convert byte position to character index
fn char_index_at_byte_pos(s: &str, byte_pos: usize) -> Option<usize> {
    s.char_indices()
        .position(|(pos, _)| pos == byte_pos)
        .or_else(|| {
            // If exact match not found, get the last valid position before byte_pos
            s.char_indices()
                .take_while(|(pos, _)| *pos < byte_pos)
                .count()
                .into()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_at_start_of_line() {
        let input = "@";
        let token = find_active_at_token(input, 1).unwrap();
        assert_eq!(token.range, (0, 1));
        assert_eq!(token.query, "");
    }

    #[test]
    fn test_at_with_query() {
        let input = "@src";
        let token = find_active_at_token(input, 4).unwrap();
        assert_eq!(token.range, (0, 4));
        assert_eq!(token.query, "src");
    }

    #[test]
    fn test_at_with_partial_query() {
        let input = "@sr";
        let token = find_active_at_token(input, 3).unwrap();
        assert_eq!(token.range, (0, 3));
        assert_eq!(token.query, "sr");
    }

    #[test]
    fn test_at_after_whitespace() {
        let input = "hello @src world";
        let token = find_active_at_token(input, 10).unwrap(); // cursor after @src
        assert_eq!(token.range, (6, 10));
        assert_eq!(token.query, "src");
    }

    #[test]
    fn test_cursor_in_middle_of_token() {
        let input = "hello @src world";
        let token = find_active_at_token(input, 8).unwrap(); // cursor at 's' in @src
        assert_eq!(token.range, (6, 10));
        assert_eq!(token.query, "src");
    }

    #[test]
    fn test_email_not_triggered() {
        let input = "test@example.com";
        let token = find_active_at_token(input, 5); // cursor at @
        assert_eq!(token, None);
    }

    #[test]
    fn test_email_not_triggered_cursor_after() {
        let input = "test@example.com";
        let token = find_active_at_token(input, 16); // cursor at end
        assert_eq!(token, None);
    }

    #[test]
    fn test_at_token_after_email() {
        let input = "email test@example.com @src";
        let token = find_active_at_token(input, 27).unwrap(); // cursor after @src
        assert_eq!(token.range, (23, 27));
        assert_eq!(token.query, "src");
    }

    #[test]
    fn test_no_token_when_cursor_outside() {
        let input = "hello @src world";
        let token = find_active_at_token(input, 0); // cursor at start
        assert_eq!(token, None);
        
        let token = find_active_at_token(input, 16); // cursor at end
        assert_eq!(token, None);
    }

    #[test]
    fn test_multiple_at_tokens() {
        let input = "@first @second";
        
        let token1 = find_active_at_token(input, 6).unwrap(); // cursor at @first
        assert_eq!(token1.range, (0, 6));
        assert_eq!(token1.query, "first");
        
        let token2 = find_active_at_token(input, 14).unwrap(); // cursor at @second
        assert_eq!(token2.range, (7, 14));
        assert_eq!(token2.query, "second");
    }
}