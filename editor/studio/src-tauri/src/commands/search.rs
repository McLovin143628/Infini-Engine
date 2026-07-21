//! Global workspace search (P5.4): a gitignore-aware, ripgrep-style content
//! search over the project. Literal or regex; results are capped so a broad
//! query stays responsive.

use ignore::WalkBuilder;
use inf_editor_core::ipc::{SearchHitDto, SearchOptsDto};
use regex::RegexBuilder;

/// Max hits returned (a broad query is capped rather than streamed).
const MAX_HITS: usize = 500;
/// Skip files larger than this (bytes) — likely generated/binary.
const MAX_FILE: u64 = 2 * 1024 * 1024;
/// Trim very long lines in the result payload.
const MAX_LINE: usize = 400;

/// Search `root` for `query`. Returns up to [`MAX_HITS`] hits.
#[tauri::command]
pub async fn search_workspace(
    root: String,
    query: String,
    opts: SearchOptsDto,
) -> Result<Vec<SearchHitDto>, String> {
    if query.is_empty() {
        return Ok(vec![]);
    }
    let pattern = if opts.regex {
        query.clone()
    } else {
        regex::escape(&query)
    };
    let re = RegexBuilder::new(&pattern)
        .case_insensitive(!opts.case_sensitive)
        .build()
        .map_err(|e| format!("bad pattern: {e}"))?;

    let root_path = std::path::Path::new(&root);
    if !root_path.is_dir() {
        return Err(format!("{root} is not a directory"));
    }

    let mut hits = Vec::new();
    let walker = WalkBuilder::new(root_path)
        .hidden(false)
        .git_ignore(true)
        .filter_entry(|e| e.file_name() != ".git")
        .build();

    'files: for dent in walker.flatten() {
        if hits.len() >= MAX_HITS {
            break;
        }
        if !dent.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = dent.path();
        if std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) > MAX_FILE {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(path) else {
            continue; // binary / non-UTF-8
        };
        let rel = path
            .strip_prefix(root_path)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");

        for (i, line) in content.lines().enumerate() {
            if let Some(m) = re.find(line) {
                let mut text = line.trim_end().to_string();
                if text.len() > MAX_LINE {
                    text.truncate(MAX_LINE);
                }
                hits.push(SearchHitDto {
                    path: rel.clone(),
                    line: (i + 1) as u32,
                    column: (m.start() + 1) as u32,
                    text,
                });
                if hits.len() >= MAX_HITS {
                    tracing::info!("search capped at {MAX_HITS} hits");
                    break 'files;
                }
            }
        }
    }
    Ok(hits)
}
