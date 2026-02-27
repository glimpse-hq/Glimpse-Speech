use std::collections::HashSet;

const MAX_DICTIONARY_ENTRIES: usize = 64;
const MAX_DICTIONARY_TERM_CHARS: usize = 160;
const MAX_PROMPT_BYTES: usize = 600;

pub fn sanitize_dictionary_entries(entries: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut cleaned = Vec::new();

    for raw in entries {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }

        let normalized = trimmed.to_lowercase();
        if !seen.insert(normalized) {
            continue;
        }

        let capped: String = trimmed.chars().take(MAX_DICTIONARY_TERM_CHARS).collect();
        let capped = capped.trim_end().to_string();
        cleaned.push(capped);

        if cleaned.len() >= MAX_DICTIONARY_ENTRIES {
            break;
        }
    }

    cleaned
}

pub fn build_dictionary_prompt(entries: &[String]) -> Option<String> {
    let cleaned = sanitize_dictionary_entries(entries);
    if cleaned.is_empty() {
        return None;
    }

    let mut prompt = String::new();
    let mut added_any = false;

    for (idx, term) in cleaned.iter().enumerate() {
        let separator = if idx > 0 { ", " } else { "" };
        let would_len = prompt.len() + separator.len() + term.len() + 1;

        if would_len > MAX_PROMPT_BYTES {
            break;
        }

        prompt.push_str(separator);
        prompt.push_str(term);
        added_any = true;
    }

    if !added_any {
        return None;
    }

    prompt.push('.');
    Some(prompt)
}

#[cfg(test)]
mod tests {
    use super::{build_dictionary_prompt, sanitize_dictionary_entries};

    #[test]
    fn sanitize_dictionary_entries_deduplicates_case_insensitively() {
        let cleaned = sanitize_dictionary_entries(&[
            "  Glimpse ".to_string(),
            "glimpse".to_string(),
            "  ".to_string(),
            "Speech".to_string(),
        ]);

        assert_eq!(cleaned, vec!["Glimpse".to_string(), "Speech".to_string()]);
    }

    #[test]
    fn build_dictionary_prompt_joins_terms() {
        let prompt = build_dictionary_prompt(&["alpha".to_string(), "beta".to_string()]);
        assert_eq!(prompt.as_deref(), Some("alpha, beta."));
    }
}
