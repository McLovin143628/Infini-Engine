//! The `.infini` lexer: bytes → spanned tokens, hand-rolled, no dependencies.
//!
//! # Two rules that are load-bearing rather than stylistic
//!
//! **Line endings are normalised before anything else.** A `.infini` file's IR
//! must be a pure function of its bytes, and a Windows checkout under
//! `core.autocrlf = true` hands the lexer `\r\n` where a Linux one hands it
//! `\n`. Normalising here — rather than hoping every consumer remembers — is the
//! same lesson `.rs is read by TESTS, so it needs text eol=lf` taught the
//! trig gate, met from the other side. The determinism gate asserts a CRLF file
//! and an LF file lower to the same IR hash.
//!
//! **Numbers keep their source text.** `-9223372036854775808` is `i64::MIN`, and
//! its digits alone do not fit an `i64`; a lexer that parsed the magnitude
//! eagerly would refuse the one literal a round-trip is most likely to meet at
//! the boundary. The token carries the digits, and [`crate::parse`] folds a
//! leading `-` into them before parsing once.
//!
//! # Long brackets
//!
//! `[[ … ]]`, `[=[ … ]=]`, `[==[ … ]==]` — Lua's long-string form, at any level.
//! They carry `rust` escape blocks ([`inf_blueprint::Stmt::Snippet`]), whose
//! contents are opaque Rust that must survive verbatim, so the *level* is chosen
//! by the emitter to be one the content does not contain. A single newline
//! immediately after the opening bracket is skipped (Lua's rule), which is what
//! lets the emitter write `[[\n<content>]]` and get `<content>` back even when
//! the content itself starts with a newline.

use std::fmt;

/// Where a token sits in the source. 1-based, in characters rather than bytes,
/// because it is shown to a human and pointed at by an editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Span {
    pub line: u32,
    pub col: u32,
    /// Length in characters. Zero at end of input.
    pub len: u32,
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

/// The reserved words. A reserved word is never an identifier, so a script
/// cannot name a variable `end` and produce a file whose meaning depends on
/// where the parser happened to be.
///
/// **`var` is deliberately not one of them.** A declaration reads `var speed:
/// float = 0` and the escape hatch for an awkward variable name reads
/// `var.get("hit count")`, and the second is a *call* whose first segment would
/// have to be an identifier. Rather than special-case a keyword in two grammar
/// positions, `var` is contextual: the top level treats a leading `var` as a
/// declaration and everything else reads it as a name. `local var` and `local
/// nodestate` are refused by the parser so the two spellings can never collide.
pub const KEYWORDS: [&str; 20] = [
    "actor", "and", "do", "else", "elseif", "end", "exposed", "false", "for", "function", "if",
    "local", "not", "on", "or", "return", "rust", "then", "true", "while",
];

/// One token.
#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    /// A reserved word, as its `&'static str` from [`KEYWORDS`].
    Kw(&'static str),
    /// An identifier (never a keyword).
    Ident(String),
    /// An integer literal, as written (no sign).
    Int(String),
    /// A float literal, as written (no sign).
    Float(String),
    /// A `"…"` string, escapes already resolved.
    Str(String),
    /// A `[[…]]` long bracket, contents verbatim.
    Long(String),
    /// Punctuation, as one of the fixed spellings in [`SYMBOLS`].
    Sym(&'static str),
    /// End of input.
    Eof,
}

impl Tok {
    /// How this token is written back in a diagnostic ("expected `end`, found …").
    pub fn describe(&self) -> String {
        match self {
            Tok::Kw(k) => format!("`{k}`"),
            Tok::Ident(i) => format!("`{i}`"),
            Tok::Int(n) | Tok::Float(n) => format!("`{n}`"),
            Tok::Str(_) => "a string".into(),
            Tok::Long(_) => "a `[[…]]` block".into(),
            Tok::Sym(s) => format!("`{s}`"),
            Tok::Eof => "end of file".into(),
        }
    }
}

/// The punctuation, **longest first** so `==` is never read as two `=`.
pub const SYMBOLS: [&str; 16] = [
    "==", "~=", "<=", ">=", "->", "(", ")", ",", ".", ":", "=", "<", ">", "+", "-", "*",
];
/// The rest, which cannot prefix one another.
const SYMBOLS2: [&str; 3] = ["/", "%", ";"];

/// A token with its span.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned {
    pub tok: Tok,
    pub span: Span,
}

/// A lexing failure — always a *value*, never a panic (P21's law).
#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub span: Span,
    pub message: String,
}

/// Tokenise `source`. CRLF is normalised to LF first; every span is in
/// characters of the normalised text.
pub fn lex(source: &str) -> Result<Vec<Spanned>, LexError> {
    let text: String = normalize_newlines(source);
    Lexer {
        chars: text.chars().collect(),
        i: 0,
        line: 1,
        col: 1,
    }
    .run()
}

/// `\r\n` → `\n`, and a lone `\r` → `\n` too (an old-Mac file is rare and a
/// silent difference in a determinism claim is worse than a rare file).
pub fn normalize_newlines(source: &str) -> String {
    if !source.contains('\r') {
        return source.to_string();
    }
    let mut out = String::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            out.push('\n');
        } else {
            out.push(c);
        }
    }
    out
}

struct Lexer {
    chars: Vec<char>,
    i: usize,
    line: u32,
    col: u32,
}

impl Lexer {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }

    fn peek_at(&self, k: usize) -> Option<char> {
        self.chars.get(self.i + k).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.i).copied()?;
        self.i += 1;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn here(&self) -> Span {
        Span {
            line: self.line,
            col: self.col,
            len: 0,
        }
    }

    fn err(&self, span: Span, message: impl Into<String>) -> LexError {
        let _ = self;
        LexError {
            span,
            message: message.into(),
        }
    }

    fn run(mut self) -> Result<Vec<Spanned>, LexError> {
        let mut out = Vec::new();
        loop {
            self.skip_trivia();
            let start = self.here();
            let Some(c) = self.peek() else {
                out.push(Spanned {
                    tok: Tok::Eof,
                    span: start,
                });
                return Ok(out);
            };
            let tok = if c == '"' {
                self.string(start)?
            } else if c == '[' && self.long_bracket_level().is_some() {
                self.long(start)?
            } else if c.is_ascii_digit() {
                self.number()
            } else if c == '_' || c.is_alphabetic() {
                self.word()
            } else {
                self.symbol(start)?
            };
            let span = Span {
                len: (self.col.saturating_sub(start.col)).max(1),
                ..start
            };
            // A multi-line token (a long bracket) has no meaningful column
            // length; clamp it to the opening delimiter so an editor underlines
            // the start rather than the rest of the file.
            let span = if self.line != start.line {
                Span { len: 2, ..start }
            } else {
                span
            };
            out.push(Spanned { tok, span });
        }
    }

    /// Whitespace and `--` line comments.
    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => {
                    self.bump();
                }
                Some('-') if self.peek_at(1) == Some('-') => {
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                _ => return,
            }
        }
    }

    /// `[`, `n` `=`s, `[` → `Some(n)`. Anything else → `None` (so a bare `[`
    /// is a plain unexpected character, which is what it is).
    fn long_bracket_level(&self) -> Option<usize> {
        if self.peek() != Some('[') {
            return None;
        }
        let mut n = 0;
        while self.peek_at(1 + n) == Some('=') {
            n += 1;
        }
        (self.peek_at(1 + n) == Some('[')).then_some(n)
    }

    fn long(&mut self, start: Span) -> Result<Tok, LexError> {
        let level = self.long_bracket_level().expect("checked by the caller");
        for _ in 0..level + 2 {
            self.bump();
        }
        // Lua's rule: one newline straight after the opener is not content.
        if self.peek() == Some('\n') {
            self.bump();
        }
        let close: String = std::iter::once(']')
            .chain(std::iter::repeat('=').take(level))
            .chain(std::iter::once(']'))
            .collect();
        let mut body = String::new();
        loop {
            if self.peek().is_none() {
                return Err(self.err(
                    start,
                    format!("unterminated `{close}` block — it is never closed"),
                ));
            }
            if self.peek() == Some(']') && self.matches_ahead(&close) {
                for _ in 0..close.chars().count() {
                    self.bump();
                }
                return Ok(Tok::Long(body));
            }
            body.push(self.bump().expect("checked just above"));
        }
    }

    fn matches_ahead(&self, s: &str) -> bool {
        s.chars()
            .enumerate()
            .all(|(k, c)| self.peek_at(k) == Some(c))
    }

    fn string(&mut self, start: Span) -> Result<Tok, LexError> {
        self.bump(); // the opening quote
        let mut out = String::new();
        loop {
            let at = self.here();
            let Some(c) = self.bump() else {
                return Err(self.err(start, "unterminated string — no closing `\"`"));
            };
            match c {
                '"' => return Ok(Tok::Str(out)),
                '\n' => {
                    return Err(self.err(
                        start,
                        "unterminated string — a `\"…\"` string may not span lines \
                         (write `\\n`, or use a `[[…]]` block)",
                    ))
                }
                '\\' => {
                    let Some(e) = self.bump() else {
                        return Err(self.err(at, "a `\\` escape at end of file"));
                    };
                    match e {
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        '0' => out.push('\0'),
                        '\\' => out.push('\\'),
                        '"' => out.push('"'),
                        'u' => out.push(self.unicode_escape(at)?),
                        other => {
                            return Err(self.err(
                                at,
                                format!(
                                    "unknown escape `\\{other}` — the escapes are \
                                     `\\n` `\\r` `\\t` `\\0` `\\\\` `\\\"` `\\u{{…}}`"
                                ),
                            ))
                        }
                    }
                }
                other => out.push(other),
            }
        }
    }

    /// `\u{1F600}` — the Rust spelling, so a `char` a Rust `String` can hold has
    /// a literal form here.
    fn unicode_escape(&mut self, at: Span) -> Result<char, LexError> {
        if self.bump() != Some('{') {
            return Err(self.err(at, "a `\\u` escape needs braces: `\\u{1F600}`"));
        }
        let mut hex = String::new();
        loop {
            let Some(c) = self.bump() else {
                return Err(self.err(at, "unterminated `\\u{…}` escape"));
            };
            if c == '}' {
                break;
            }
            hex.push(c);
        }
        let code = u32::from_str_radix(&hex, 16)
            .map_err(|_| self.err(at, format!("`{hex}` is not hexadecimal")))?;
        char::from_u32(code).ok_or_else(|| {
            self.err(
                at,
                format!("`\\u{{{hex}}}` is not a character (a lone surrogate, or out of range)"),
            )
        })
    }

    /// A decimal number. A `.` or an exponent makes it a float; otherwise it is
    /// an integer. The digits are kept **as written**, unparsed.
    fn number(&mut self) -> Tok {
        let mut s = String::new();
        let mut float = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.bump();
            } else if c == '.' && !float && self.peek_at(1).is_some_and(|d| d.is_ascii_digit()) {
                float = true;
                s.push(c);
                self.bump();
            } else if (c == 'e' || c == 'E')
                && (self.peek_at(1).is_some_and(|d| d.is_ascii_digit())
                    || (matches!(self.peek_at(1), Some('+') | Some('-'))
                        && self.peek_at(2).is_some_and(|d| d.is_ascii_digit())))
            {
                float = true;
                s.push(c);
                self.bump();
                if matches!(self.peek(), Some('+') | Some('-')) {
                    s.push(self.peek().expect("just matched"));
                    self.bump();
                }
            } else {
                break;
            }
        }
        if float {
            Tok::Float(s)
        } else {
            Tok::Int(s)
        }
    }

    fn word(&mut self) -> Tok {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c == '_' || c.is_alphanumeric() {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        match KEYWORDS.iter().find(|k| **k == s) {
            Some(k) => Tok::Kw(k),
            None => Tok::Ident(s),
        }
    }

    fn symbol(&mut self, start: Span) -> Result<Tok, LexError> {
        for s in SYMBOLS.iter().chain(SYMBOLS2.iter()) {
            if self.matches_ahead(s) {
                for _ in 0..s.chars().count() {
                    self.bump();
                }
                return Ok(Tok::Sym(s));
            }
        }
        let c = self.bump().expect("the caller peeked one");
        Err(self.err(start, format!("unexpected character `{c}`")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(src: &str) -> Vec<Tok> {
        lex(src)
            .expect("lexes")
            .into_iter()
            .map(|s| s.tok)
            .collect()
    }

    #[test]
    fn words_split_into_keywords_and_identifiers() {
        assert_eq!(
            toks("on tick angle end"),
            vec![
                Tok::Kw("on"),
                // `tick` is an event *name*, not a reserved word: the keyword
                // list is short on purpose, so a script may still call a
                // variable `tick`.
                Tok::Ident("tick".into()),
                Tok::Ident("angle".into()),
                Tok::Kw("end"),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn numbers_keep_their_text_and_their_kind() {
        assert_eq!(
            toks("1 1.5 2e3 9223372036854775808"),
            vec![
                Tok::Int("1".into()),
                Tok::Float("1.5".into()),
                Tok::Float("2e3".into()),
                // Larger than i64::MAX, and lexed anyway — the parser folds the
                // sign in before it parses, so `i64::MIN` has a literal form.
                Tok::Int("9223372036854775808".into()),
                Tok::Eof
            ]
        );
        // A trailing `.` is not part of the number (`x.` is a field access).
        assert_eq!(
            toks("1.foo"),
            vec![
                Tok::Int("1".into()),
                Tok::Sym("."),
                Tok::Ident("foo".into()),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn comments_run_to_the_end_of_the_line() {
        assert_eq!(
            toks("a -- b c\nd"),
            vec![Tok::Ident("a".into()), Tok::Ident("d".into()), Tok::Eof]
        );
    }

    #[test]
    fn strings_resolve_their_escapes() {
        assert_eq!(
            toks(r#""a\nb\u{41}\\\"""#),
            vec![Tok::Str("a\nbA\\\"".into()), Tok::Eof]
        );
    }

    #[test]
    fn an_unterminated_string_names_its_line() {
        let e = lex("local x = \"oops\nlocal y = 1").unwrap_err();
        assert_eq!(e.span.line, 1);
        assert!(e.message.contains("may not span lines"), "{}", e.message);
    }

    /// The long bracket takes its content verbatim, at whatever level closes it.
    #[test]
    fn long_brackets_nest_by_level() {
        assert_eq!(toks("[[hello]]"), vec![Tok::Long("hello".into()), Tok::Eof]);
        // A `]]` inside the content is reachable at level 1.
        assert_eq!(
            toks("[=[v[[1,2][0]]]=]"),
            vec![Tok::Long("v[[1,2][0]]".into()), Tok::Eof]
        );
        // One newline after the opener is skipped; the rest is content.
        assert_eq!(
            toks("[[\nline1\nline2\n]]"),
            vec![Tok::Long("line1\nline2\n".into()), Tok::Eof]
        );
    }

    /// **CRLF and LF lex identically.** The determinism law's first half, in the
    /// one place it can be enforced for everything downstream.
    #[test]
    fn crlf_and_lf_lex_to_the_same_tokens() {
        let lf = "on tick(dt)\n  debug.print(\"x\")\nend\n";
        let crlf = lf.replace('\n', "\r\n");
        assert_ne!(lf, crlf, "the two inputs really do differ in bytes");
        assert_eq!(toks(lf), toks(&crlf));
        // …and so do their spans, which is the half a naive normalisation misses.
        assert_eq!(lex(lf).unwrap(), lex(&crlf).unwrap());
    }

    /// A `[[…]]` block that is never closed is a refusal with a line, not a hang
    /// and not a panic.
    #[test]
    fn an_unclosed_long_bracket_refuses() {
        let e = lex("rust [[\nfn oops() {\n").unwrap_err();
        assert_eq!(e.span.line, 1);
        assert!(e.message.contains("unterminated"), "{}", e.message);
    }

    #[test]
    fn symbols_prefer_the_longer_spelling() {
        assert_eq!(
            toks("== = ~= <= < ->"),
            vec![
                Tok::Sym("=="),
                Tok::Sym("="),
                Tok::Sym("~="),
                Tok::Sym("<="),
                Tok::Sym("<"),
                Tok::Sym("->"),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn an_unexpected_character_names_itself() {
        let e = lex("local x = @").unwrap_err();
        assert!(e.message.contains('@'), "{}", e.message);
        assert_eq!((e.span.line, e.span.col), (1, 11));
    }
}
