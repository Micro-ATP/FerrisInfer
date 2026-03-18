use std::collections::{HashMap, HashSet};

use ferrisinfer_core::{ErrorKind, FerrisError, Result};

const DEFAULT_QWEN_SYSTEM_PROMPT: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    BytePair,
    SentencePiece,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant, content)
    }

    pub fn tool(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Tool, content)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplateKind {
    Qwen2,
}

#[derive(Debug, Clone)]
pub struct TokenizerAsset {
    pub kind: TokenizerKind,
    pub vocab_size: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub chat_template: Option<String>,
    model: Option<TokenizerModelAsset>,
}

#[derive(Debug, Clone)]
pub enum TokenizerModelAsset {
    BytePair(BytePairTokenizerModel),
}

#[derive(Debug, Clone)]
pub struct BytePairTokenizerModel {
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    merge_ranks: HashMap<(String, String), usize>,
    added_token_to_id: HashMap<String, u32>,
    added_token_ids: HashSet<u32>,
    added_tokens_sorted: Vec<String>,
    byte_to_char: Vec<char>,
    char_to_byte: HashMap<char, u8>,
    add_prefix_space: bool,
}

impl TokenizerAsset {
    pub fn new(
        kind: TokenizerKind,
        vocab_size: usize,
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
        unk_token_id: Option<u32>,
    ) -> Self {
        Self {
            kind,
            vocab_size,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            chat_template: None,
            model: None,
        }
    }

    pub fn with_chat_template(mut self, chat_template: Option<String>) -> Self {
        self.chat_template = chat_template;
        self
    }

    pub fn with_model(mut self, model: TokenizerModelAsset) -> Self {
        self.model = Some(model);
        self
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    pub fn chat_template_kind(&self) -> Option<ChatTemplateKind> {
        let template = self.chat_template()?;
        if template.contains("<|im_start|>") && template.contains("<|im_end|>") {
            Some(ChatTemplateKind::Qwen2)
        } else {
            None
        }
    }

    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        match self.chat_template_kind() {
            Some(ChatTemplateKind::Qwen2) => Ok(render_qwen2_chat(messages, add_generation_prompt)),
            None => Err(FerrisError::unsupported(
                "chat template rendering is not implemented for this tokenizer",
            )),
        }
    }

    fn byte_pair_model(&self) -> Result<&BytePairTokenizerModel> {
        match &self.model {
            Some(TokenizerModelAsset::BytePair(model)) => Ok(model),
            None => Err(FerrisError::unsupported(
                "tokenizer model data is not loaded",
            )),
        }
    }
}

impl BytePairTokenizerModel {
    pub fn new(
        token_to_id: HashMap<String, u32>,
        merge_ranks: HashMap<(String, String), usize>,
        added_token_to_id: HashMap<String, u32>,
        add_prefix_space: bool,
    ) -> Self {
        let mut id_to_token = HashMap::with_capacity(token_to_id.len() + added_token_to_id.len());
        for (token, id) in &token_to_id {
            id_to_token.insert(*id, token.clone());
        }

        let mut added_token_ids = HashSet::with_capacity(added_token_to_id.len());
        for (token, id) in &added_token_to_id {
            id_to_token.insert(*id, token.clone());
            added_token_ids.insert(*id);
        }

        let mut added_tokens_sorted = added_token_to_id.keys().cloned().collect::<Vec<_>>();
        added_tokens_sorted
            .sort_by(|left, right| right.len().cmp(&left.len()).then_with(|| left.cmp(right)));

        let (byte_to_char, char_to_byte) = bytes_to_unicode_tables();

        Self {
            token_to_id,
            id_to_token,
            merge_ranks,
            added_token_to_id,
            added_token_ids,
            added_tokens_sorted,
            byte_to_char,
            char_to_byte,
            add_prefix_space,
        }
    }

    fn encode(
        &self,
        text: &str,
        bos_token_id: Option<u32>,
        unk_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        if let Some(id) = bos_token_id {
            tokens.push(id);
        }

        let normalized = if self.add_prefix_space
            && !text.is_empty()
            && !text.chars().next().is_some_and(char::is_whitespace)
        {
            format!(" {text}")
        } else {
            text.to_string()
        };

        for segment in self.split_added_tokens(&normalized) {
            match segment {
                TokenSegment::AddedToken(token) => {
                    let id = self.added_token_to_id.get(token).copied().ok_or_else(|| {
                        FerrisError::new(ErrorKind::Parse, format!("unknown added token '{token}'"))
                    })?;
                    tokens.push(id);
                }
                TokenSegment::Text(text) => {
                    self.encode_text_segment(text, unk_token_id, &mut tokens)?
                }
            }
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut output = String::new();
        let mut regular_bytes = Vec::new();

        for token_id in tokens {
            let token = self.id_to_token.get(token_id).ok_or_else(|| {
                FerrisError::new(ErrorKind::Parse, format!("unknown token id {token_id}"))
            })?;

            if self.added_token_ids.contains(token_id) {
                flush_regular_bytes(&mut output, &mut regular_bytes);
                output.push_str(token);
                continue;
            }

            for ch in token.chars() {
                let byte = self.char_to_byte.get(&ch).copied().ok_or_else(|| {
                    FerrisError::new(
                        ErrorKind::Parse,
                        format!("token contains unmapped byte-level character '{ch}'"),
                    )
                })?;
                regular_bytes.push(byte);
            }
        }

        flush_regular_bytes(&mut output, &mut regular_bytes);
        Ok(output)
    }

    fn encode_text_segment(
        &self,
        text: &str,
        unk_token_id: Option<u32>,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        let mut cursor = 0usize;
        while cursor < text.len() {
            let end = next_pretokenized_boundary(text, cursor);
            let piece = &text[cursor..end];
            let byte_encoded = self.byte_encode(piece);
            for bpe_token in self.byte_pair_encode(&byte_encoded) {
                match self.token_to_id.get(&bpe_token).copied() {
                    Some(id) => out.push(id),
                    None => {
                        if let Some(id) = unk_token_id {
                            out.push(id);
                        } else {
                            return Err(FerrisError::new(
                                ErrorKind::Parse,
                                format!("byte pair token '{bpe_token}' is missing from vocabulary"),
                            ));
                        }
                    }
                }
            }
            cursor = end;
        }

        Ok(())
    }

    fn byte_encode(&self, text: &str) -> String {
        let mut encoded = String::new();
        for byte in text.as_bytes() {
            encoded.push(self.byte_to_char[*byte as usize]);
        }
        encoded
    }

    fn byte_pair_encode(&self, token: &str) -> Vec<String> {
        let mut pieces = token.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
        if pieces.len() <= 1 {
            return pieces;
        }

        loop {
            let mut best_rank = usize::MAX;
            let mut best_index = None;

            for index in 0..pieces.len() - 1 {
                let pair = (pieces[index].clone(), pieces[index + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_index = Some(index);
                    }
                }
            }

            let Some(index) = best_index else {
                break;
            };

            let merged = format!("{}{}", pieces[index], pieces[index + 1]);
            pieces[index] = merged;
            pieces.remove(index + 1);

            if pieces.len() <= 1 {
                break;
            }
        }

        pieces
    }

    fn split_added_tokens<'a>(&'a self, text: &'a str) -> Vec<TokenSegment<'a>> {
        if self.added_tokens_sorted.is_empty() {
            return vec![TokenSegment::Text(text)];
        }

        let mut segments = Vec::new();
        let mut text_start = 0usize;
        let mut cursor = 0usize;

        while cursor < text.len() {
            let remainder = &text[cursor..];
            let mut matched = None;

            for token in &self.added_tokens_sorted {
                if remainder.starts_with(token) {
                    matched = Some(token.as_str());
                    break;
                }
            }

            if let Some(token) = matched {
                if text_start < cursor {
                    segments.push(TokenSegment::Text(&text[text_start..cursor]));
                }
                segments.push(TokenSegment::AddedToken(token));
                cursor += token.len();
                text_start = cursor;
                continue;
            }

            cursor += text[cursor..]
                .chars()
                .next()
                .expect("cursor always points to a char boundary")
                .len_utf8();
        }

        if text_start < text.len() {
            segments.push(TokenSegment::Text(&text[text_start..]));
        }

        segments
    }
}

pub trait Tokenizer: Send + Sync {
    fn kind(&self) -> TokenizerKind;
    fn vocab_size(&self) -> usize;
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
}

#[derive(Debug, Clone)]
pub struct VocabularyTokenizer {
    asset: TokenizerAsset,
}

impl VocabularyTokenizer {
    pub fn new(asset: TokenizerAsset) -> Self {
        Self { asset }
    }

    pub fn asset(&self) -> &TokenizerAsset {
        &self.asset
    }

    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        self.asset.render_chat(messages, add_generation_prompt)
    }
}

impl Tokenizer for VocabularyTokenizer {
    fn kind(&self) -> TokenizerKind {
        self.asset.kind
    }

    fn vocab_size(&self) -> usize {
        self.asset.vocab_size
    }

    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        match self.asset.kind {
            TokenizerKind::BytePair => self.asset.byte_pair_model()?.encode(
                text,
                if add_bos {
                    self.asset.bos_token_id
                } else {
                    None
                },
                self.asset.unk_token_id,
            ),
            TokenizerKind::SentencePiece => Err(FerrisError::unsupported(
                "SentencePiece tokenizer encode is not implemented yet",
            )),
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.asset.kind {
            TokenizerKind::BytePair => self.asset.byte_pair_model()?.decode(tokens),
            TokenizerKind::SentencePiece => Err(FerrisError::unsupported(
                "SentencePiece tokenizer decode is not implemented yet",
            )),
        }
    }
}

enum TokenSegment<'a> {
    AddedToken(&'a str),
    Text(&'a str),
}

fn render_qwen2_chat(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut rendered = String::new();
    let mut cursor = 0usize;

    if let Some(first) = messages.first() {
        if first.role == ChatRole::System {
            push_qwen2_message(&mut rendered, ChatRole::System, &first.content);
            cursor = 1;
        } else {
            push_qwen2_message(&mut rendered, ChatRole::System, DEFAULT_QWEN_SYSTEM_PROMPT);
        }
    } else {
        push_qwen2_message(&mut rendered, ChatRole::System, DEFAULT_QWEN_SYSTEM_PROMPT);
    }

    while cursor < messages.len() {
        match messages[cursor].role {
            ChatRole::Tool => {
                rendered.push_str("<|im_start|>user");
                while cursor < messages.len() && messages[cursor].role == ChatRole::Tool {
                    rendered.push_str("\n<tool_response>\n");
                    rendered.push_str(&messages[cursor].content);
                    rendered.push_str("\n</tool_response>");
                    cursor += 1;
                }
                rendered.push_str("<|im_end|>\n");
            }
            role => {
                push_qwen2_message(&mut rendered, role, &messages[cursor].content);
                cursor += 1;
            }
        }
    }

    if add_generation_prompt {
        rendered.push_str("<|im_start|>assistant\n");
    }

    rendered
}

fn push_qwen2_message(rendered: &mut String, role: ChatRole, content: &str) {
    rendered.push_str("<|im_start|>");
    rendered.push_str(role.as_str());
    rendered.push('\n');
    rendered.push_str(content);
    rendered.push_str("<|im_end|>\n");
}

fn flush_regular_bytes(output: &mut String, regular_bytes: &mut Vec<u8>) {
    if regular_bytes.is_empty() {
        return;
    }

    output.push_str(&String::from_utf8_lossy(regular_bytes));
    regular_bytes.clear();
}

fn bytes_to_unicode_tables() -> (Vec<char>, HashMap<char, u8>) {
    let mut bs = Vec::new();
    bs.extend(33u16..=126u16);
    bs.extend(161u16..=172u16);
    bs.extend(174u16..=255u16);

    let mut present = [false; 256];
    for &value in &bs {
        present[value as usize] = true;
    }

    let mut cs = bs.clone();
    let mut extra = 0u16;
    for byte in 0u16..=255u16 {
        if !present[byte as usize] {
            bs.push(byte);
            cs.push(256 + extra);
            extra += 1;
        }
    }

    let mut byte_to_char = vec!['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);
    for (byte, codepoint) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(*codepoint as u32).expect("generated byte mapping is valid");
        byte_to_char[*byte as usize] = ch;
        char_to_byte.insert(ch, *byte as u8);
    }

    (byte_to_char, char_to_byte)
}

fn next_pretokenized_boundary(text: &str, start: usize) -> usize {
    match_contraction(text, start)
        .or_else(|| match_letter_sequence(text, start))
        .or_else(|| match_number(text, start))
        .or_else(|| match_symbol_chunk(text, start))
        .or_else(|| match_newline_whitespace(text, start))
        .or_else(|| match_trailing_whitespace(text, start))
        .or_else(|| match_whitespace(text, start))
        .unwrap_or_else(|| next_char_boundary(text, start))
}

fn match_contraction(text: &str, start: usize) -> Option<usize> {
    const PATTERNS: [&str; 7] = ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"];
    let remainder = &text[start..];

    for pattern in PATTERNS {
        if let Some(candidate) = remainder.get(..pattern.len()) {
            if candidate.eq_ignore_ascii_case(pattern) {
                return Some(start + pattern.len());
            }
        }
    }

    None
}

fn match_letter_sequence(text: &str, start: usize) -> Option<usize> {
    let (first, next) = next_char(text, start)?;

    if is_letter(first) {
        return Some(consume_while(text, next, is_letter));
    }

    if first != '\r' && first != '\n' && !is_letter(first) && !is_number(first) {
        let (second, second_next) = next_char(text, next)?;
        if is_letter(second) {
            return Some(consume_while(text, second_next, is_letter));
        }
    }

    None
}

fn match_number(text: &str, start: usize) -> Option<usize> {
    let (first, next) = next_char(text, start)?;
    if is_number(first) {
        Some(next)
    } else {
        None
    }
}

fn match_symbol_chunk(text: &str, start: usize) -> Option<usize> {
    let (first, next) = next_char(text, start)?;

    let symbol_start = if first == ' ' {
        let (second, second_next) = next_char(text, next)?;
        if is_symbol(second) {
            let after_symbols = consume_while(text, second_next, is_symbol);
            return Some(consume_while(text, after_symbols, is_crlf));
        }
        return None;
    } else if is_symbol(first) {
        next
    } else {
        return None;
    };

    let after_symbols = consume_while(text, symbol_start, is_symbol);
    Some(consume_while(text, after_symbols, is_crlf))
}

fn match_newline_whitespace(text: &str, start: usize) -> Option<usize> {
    let (first, _) = next_char(text, start)?;
    if !first.is_whitespace() {
        return None;
    }

    let before_newline = consume_while(text, start, |ch| ch.is_whitespace() && !is_crlf(ch));
    let after_newline = consume_while(text, before_newline, is_crlf);
    if after_newline > before_newline {
        Some(after_newline)
    } else {
        None
    }
}

fn match_trailing_whitespace(text: &str, start: usize) -> Option<usize> {
    let (first, _) = next_char(text, start)?;
    if !first.is_whitespace() {
        return None;
    }

    if text[start..].chars().all(char::is_whitespace) {
        Some(text.len())
    } else {
        None
    }
}

fn match_whitespace(text: &str, start: usize) -> Option<usize> {
    let (first, next) = next_char(text, start)?;
    if first.is_whitespace() {
        Some(consume_while(text, next, char::is_whitespace))
    } else {
        None
    }
}

fn consume_while(text: &str, mut cursor: usize, predicate: impl Fn(char) -> bool) -> usize {
    while let Some((ch, next)) = next_char(text, cursor) {
        if !predicate(ch) {
            break;
        }
        cursor = next;
    }
    cursor
}

fn next_char(text: &str, start: usize) -> Option<(char, usize)> {
    let ch = text[start..].chars().next()?;
    Some((ch, start + ch.len_utf8()))
}

fn next_char_boundary(text: &str, start: usize) -> usize {
    start
        + text[start..]
            .chars()
            .next()
            .expect("start must point to a char boundary")
            .len_utf8()
}

fn is_letter(ch: char) -> bool {
    ch.is_alphabetic()
}

fn is_number(ch: char) -> bool {
    ch.is_numeric()
}

fn is_crlf(ch: char) -> bool {
    ch == '\r' || ch == '\n'
}

fn is_symbol(ch: char) -> bool {
    !ch.is_whitespace() && !is_letter(ch) && !is_number(ch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_tables_round_trip_all_bytes() {
        let (byte_to_char, char_to_byte) = bytes_to_unicode_tables();

        for byte in 0u8..=255u8 {
            let ch = byte_to_char[byte as usize];
            assert_eq!(char_to_byte.get(&ch).copied(), Some(byte));
        }
    }

    #[test]
    fn vocabulary_tokenizer_encodes_and_decodes_byte_pair_text() {
        let tokenizer = VocabularyTokenizer::new(test_asset(false));

        let tokens = tokenizer.encode("hello world", false).unwrap();
        assert_eq!(tokens, vec![8, 2, 3, 13]);
        assert_eq!(tokenizer.decode(&tokens).unwrap(), "hello world");
    }

    #[test]
    fn vocabulary_tokenizer_handles_added_tokens_and_bos() {
        let tokenizer = VocabularyTokenizer::new(test_asset(true));

        let tokens = tokenizer.encode("<|im_start|>hello", true).unwrap();
        assert_eq!(tokens, vec![99, 100, 8, 2, 3]);
        assert_eq!(tokenizer.decode(&tokens).unwrap(), "<bos><|im_start|>hello");
    }

    #[test]
    fn vocabulary_tokenizer_renders_qwen_chat_template() {
        let tokenizer = VocabularyTokenizer::new(test_asset(false));
        let rendered = tokenizer
            .render_chat(&[ChatMessage::user("你好")], true)
            .unwrap();

        assert_eq!(
            rendered,
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn vocabulary_tokenizer_renders_tool_messages_as_grouped_user_block() {
        let tokenizer = VocabularyTokenizer::new(test_asset(false));
        let rendered = tokenizer
            .render_chat(
                &[
                    ChatMessage::system("sys"),
                    ChatMessage::user("u"),
                    ChatMessage::tool("tool-a"),
                    ChatMessage::tool("tool-b"),
                ],
                false,
            )
            .unwrap();

        assert_eq!(
            rendered,
            "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>user\n<tool_response>\ntool-a\n</tool_response>\n<tool_response>\ntool-b\n</tool_response><|im_end|>\n"
        );
    }

    fn test_asset(with_bos: bool) -> TokenizerAsset {
        let (byte_to_char, _) = bytes_to_unicode_tables();

        let space = byte_to_char[b' ' as usize].to_string();
        let h = byte_to_char[b'h' as usize].to_string();
        let e = byte_to_char[b'e' as usize].to_string();
        let l = byte_to_char[b'l' as usize].to_string();
        let o = byte_to_char[b'o' as usize].to_string();
        let w = byte_to_char[b'w' as usize].to_string();
        let r = byte_to_char[b'r' as usize].to_string();
        let d = byte_to_char[b'd' as usize].to_string();

        let mut token_to_id = HashMap::new();
        token_to_id.insert(h.clone(), 0);
        token_to_id.insert(e.clone(), 1);
        token_to_id.insert(l.clone(), 2);
        token_to_id.insert(o.clone(), 3);
        token_to_id.insert(w.clone(), 4);
        token_to_id.insert(r.clone(), 5);
        token_to_id.insert(d.clone(), 6);
        token_to_id.insert(format!("{h}{e}"), 7);
        token_to_id.insert(format!("{}{}{}", h, e, l), 8);
        token_to_id.insert(format!("{l}{l}"), 9);
        token_to_id.insert(format!("{}{}", space, w), 10);
        token_to_id.insert(format!("{}{}{}{}{}", space, w, o, r, l), 11);
        token_to_id.insert(
            "hello"
                .bytes()
                .map(|byte| byte_to_char[byte as usize])
                .collect(),
            12,
        );
        token_to_id.insert(
            " world"
                .bytes()
                .map(|byte| byte_to_char[byte as usize])
                .collect(),
            13,
        );

        let mut merges = HashMap::new();
        merges.insert((h.clone(), e.clone()), 0);
        merges.insert((format!("{h}{e}"), l.clone()), 1);
        merges.insert((l.clone(), l.clone()), 2);
        merges.insert((format!("{}{}{}", h, e, l), format!("{l}{l}")), 3);
        merges.insert((format!("{}{}{}{}", h, e, l, l), o.clone()), 4);
        merges.insert((space.clone(), w.clone()), 5);
        merges.insert((format!("{}{}", space, w), o.clone()), 6);
        merges.insert((format!("{}{}{}", space, w, o), r.clone()), 7);
        merges.insert((format!("{}{}{}{}", space, w, o, r), l.clone()), 8);
        merges.insert((format!("{}{}{}{}{}", space, w, o, r, l), d.clone()), 9);

        let mut added = HashMap::new();
        added.insert("<bos>".to_string(), 99);
        added.insert("<|im_start|>".to_string(), 100);

        TokenizerAsset::new(
            TokenizerKind::BytePair,
            101,
            if with_bos { Some(99) } else { None },
            None,
            None,
        )
        .with_chat_template(Some("<|im_start|>...<|im_end|>".to_string()))
        .with_model(TokenizerModelAsset::BytePair(BytePairTokenizerModel::new(
            token_to_id,
            merges,
            added,
            false,
        )))
    }
}
