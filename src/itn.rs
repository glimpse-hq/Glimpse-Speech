#[derive(Debug, Clone)]
struct TokenPart {
    leading: String,
    core: String,
    trailing: String,
}

pub fn apply_simple_english_itn(text: &str) -> String {
    if text.trim().is_empty() {
        return String::new();
    }

    let tokens: Vec<TokenPart> = text.split_whitespace().map(split_token).collect();
    let mut output: Vec<String> = Vec::new();

    let mut idx = 0usize;
    while idx < tokens.len() {
        if let Some((consumed, number)) = parse_number_phrase(&tokens[idx..]) {
            if consumed > 0 {
                let last = idx + consumed - 1;
                output.push(format!(
                    "{}{}{}",
                    tokens[idx].leading, number, tokens[last].trailing
                ));
                idx += consumed;
                continue;
            }
        }

        output.push(format!(
            "{}{}{}",
            tokens[idx].leading, tokens[idx].core, tokens[idx].trailing
        ));
        idx += 1;
    }

    output.join(" ")
}

fn parse_number_phrase(tokens: &[TokenPart]) -> Option<(usize, String)> {
    if tokens.is_empty() {
        return None;
    }

    let mut idx = 0usize;
    let mut negative = false;

    if let Some(word) = normalized_single_word(&tokens[idx]) {
        if word == "minus" || word == "negative" {
            negative = true;
            idx += 1;
        }
    }

    if idx >= tokens.len() {
        return None;
    }

    let mut total: i64 = 0;
    let mut group: i64 = 0;
    let mut saw_numeric = false;
    let mut saw_complex = false;
    let mut numeric_word_count = 0usize;
    let mut consumed = idx;

    while idx < tokens.len() {
        let words = normalized_words(&tokens[idx]);
        if words.is_empty() {
            break;
        }

        if words.len() == 1 && words[0] == "and" && saw_numeric {
            idx += 1;
            consumed = idx;
            continue;
        }

        if words.len() == 1 && words[0] == "point" && saw_numeric {
            if let Some((decimal_consumed, decimal_digits)) = parse_decimal_part(&tokens[idx + 1..])
            {
                let integer = total + group;
                let mut rendered = if negative {
                    format!("-{integer}")
                } else {
                    integer.to_string()
                };
                rendered.push('.');
                rendered.push_str(&decimal_digits);
                return Some((idx + 1 + decimal_consumed, rendered));
            }
            break;
        }

        let mut token_supported = true;

        for word in words {
            if let Some(value) = digit_value(&word) {
                group += i64::from(value);
                saw_numeric = true;
                numeric_word_count += 1;
                continue;
            }

            if let Some(value) = teen_or_tens_value(&word) {
                group += i64::from(value);
                saw_numeric = true;
                saw_complex = true;
                numeric_word_count += 1;
                continue;
            }

            if word == "hundred" {
                if group == 0 {
                    group = 1;
                }
                group *= 100;
                saw_numeric = true;
                saw_complex = true;
                numeric_word_count += 1;
                continue;
            }

            if let Some(multiplier) = magnitude_value(&word) {
                if group == 0 {
                    group = 1;
                }
                total += group * multiplier;
                group = 0;
                saw_numeric = true;
                saw_complex = true;
                numeric_word_count += 1;
                continue;
            }

            token_supported = false;
            break;
        }

        if !token_supported {
            break;
        }

        idx += 1;
        consumed = idx;
    }

    if !saw_numeric {
        return None;
    }

    let should_convert = saw_complex || numeric_word_count >= 2 || negative;
    if !should_convert {
        return None;
    }

    let mut rendered = (total + group).to_string();
    if negative {
        rendered.insert(0, '-');
    }

    Some((consumed, rendered))
}

fn parse_decimal_part(tokens: &[TokenPart]) -> Option<(usize, String)> {
    let mut digits = String::new();
    let mut consumed = 0usize;

    for token in tokens {
        let Some(word) = normalized_single_word(token) else {
            break;
        };
        let Some(value) = decimal_digit_value(&word) else {
            break;
        };
        digits.push(char::from(b'0' + value as u8));
        consumed += 1;
    }

    if digits.is_empty() {
        None
    } else {
        Some((consumed, digits))
    }
}

fn split_token(raw: &str) -> TokenPart {
    let Some(start) = raw.find(|ch: char| ch.is_alphanumeric()) else {
        return TokenPart {
            leading: raw.to_string(),
            core: String::new(),
            trailing: String::new(),
        };
    };

    let end = raw
        .rfind(|ch: char| ch.is_alphanumeric())
        .map(|index| index + 1)
        .unwrap_or(start);

    TokenPart {
        leading: raw[..start].to_string(),
        core: raw[start..end].to_string(),
        trailing: raw[end..].to_string(),
    }
}

fn normalized_single_word(token: &TokenPart) -> Option<String> {
    let words = normalized_words(token);
    if words.len() == 1 {
        Some(words[0].clone())
    } else {
        None
    }
}

fn normalized_words(token: &TokenPart) -> Vec<String> {
    token
        .core
        .split('-')
        .filter_map(|word| {
            let trimmed = word.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_ascii_lowercase())
            }
        })
        .collect()
}

fn digit_value(word: &str) -> Option<u32> {
    match word {
        "zero" | "oh" => Some(0),
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        _ => None,
    }
}

fn teen_or_tens_value(word: &str) -> Option<u32> {
    match word {
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "thirteen" => Some(13),
        "fourteen" => Some(14),
        "fifteen" => Some(15),
        "sixteen" => Some(16),
        "seventeen" => Some(17),
        "eighteen" => Some(18),
        "nineteen" => Some(19),
        "twenty" => Some(20),
        "thirty" => Some(30),
        "forty" => Some(40),
        "fifty" => Some(50),
        "sixty" => Some(60),
        "seventy" => Some(70),
        "eighty" => Some(80),
        "ninety" => Some(90),
        _ => None,
    }
}

fn magnitude_value(word: &str) -> Option<i64> {
    match word {
        "thousand" => Some(1_000),
        "million" => Some(1_000_000),
        "billion" => Some(1_000_000_000),
        _ => None,
    }
}

fn decimal_digit_value(word: &str) -> Option<u32> {
    digit_value(word)
}

#[cfg(test)]
mod tests {
    use super::apply_simple_english_itn;

    #[test]
    fn converts_compound_numbers() {
        let text = "I have five hundred sixty three files.";
        assert_eq!(apply_simple_english_itn(text), "I have 563 files.");
    }

    #[test]
    fn converts_hyphenated_numbers() {
        let text = "Room thirty-two is ready.";
        assert_eq!(apply_simple_english_itn(text), "Room 32 is ready.");
    }

    #[test]
    fn converts_decimal_numbers() {
        let text = "Version one point five is stable.";
        assert_eq!(apply_simple_english_itn(text), "Version 1.5 is stable.");
    }

    #[test]
    fn keeps_single_simple_number_words() {
        let text = "I want one apple.";
        assert_eq!(apply_simple_english_itn(text), "I want one apple.");
    }

    #[test]
    fn converts_negative_numbers() {
        let text = "temperature is negative forty two.";
        assert_eq!(apply_simple_english_itn(text), "temperature is -42.");
    }
}
