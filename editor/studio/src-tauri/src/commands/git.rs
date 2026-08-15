//! Git panel commands (P5.4): shell out to the `git` CLI.
//!
//! Ported from the CodeR reference approach — no libgit2 linkage, so there's no
//! native-build friction. Every call goes through [`run_git`] with args as a
//! list (no shell), a `--` separator before paths, and an exit-code whitelist.
//! (stash / blame / log / push / pull are deferred — see TODOs.)

use std::path::PathBuf;
use std::process::Command;

use inf_editor_core::ipc::{GitFileDto, GitStatusDto};
use tauri::{AppHandle, State};

use super::paths::{confine_existing, confine_under};
use super::project::ProjectState;

/// Resolve + confine a `repo` argument (L7.H6).
///
/// Every command in this module shells out to `git -C <repo>`, and `repo`
/// arrived from the webview as a bare absolute path — so `git_discard` could be
/// pointed at any repository on the machine and told to throw away its working
/// tree. The guard is `shell_reveal`'s, hoisted into `super::paths`.
fn confined_repo(app: &AppHandle, project: &ProjectState, repo: &str) -> Result<PathBuf, String> {
    confine_existing(app, project, repo)
}

/// Validate every path argument as living **inside** the confined repo, and hand
/// back the relative spellings git wants. A `..` or an absolute path is refused
/// rather than normalized: neither is something the panel produced.
fn confined_paths(root: &PathBuf, paths: &[String]) -> Result<Vec<String>, String> {
    for p in paths {
        confine_under(root, p)?;
    }
    Ok(paths.to_vec())
}

/// Run `git -C <repo> <args>`, returning stdout. Errors unless the exit code is
/// 0 or in `allowed`.
fn run_git(repo: &str, args: &[&str], allowed: &[i32]) -> Result<String, String> {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(repo).args(args);
    let out = cmd.output().map_err(|e| format!("spawn git: {e}"))?;
    let code = out.status.code().unwrap_or(-1);
    if code == 0 || allowed.contains(&code) {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        Err(format!(
            "git {} failed ({code}): {}",
            args.first().copied().unwrap_or(""),
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

fn is_repo(repo: &str) -> bool {
    run_git(repo, &["rev-parse", "--is-inside-work-tree"], &[128])
        .map(|s| s.trim() == "true")
        .unwrap_or(false)
}

/// Working-tree status via `--porcelain=v2 --branch`.
#[tauri::command]
pub async fn git_status(
    app: AppHandle,
    project: State<'_, ProjectState>,
    repo: String,
) -> Result<GitStatusDto, String> {
    let _root = confined_repo(&app, &project, &repo)?;
    // Shelling out to git can block for a long time on a big repo — keep the
    // blocking work off the async workers (mirrors package.rs).
    tauri::async_runtime::spawn_blocking(move || {
        if !is_repo(&repo) {
            return Ok(GitStatusDto {
                is_repo: false,
                branch: String::new(),
                ahead: 0,
                behind: 0,
                files: vec![],
            });
        }
        let out = run_git(&repo, &["status", "--porcelain=v2", "--branch"], &[])?;
        let mut status = GitStatusDto {
            is_repo: true,
            branch: "(detached)".into(),
            ahead: 0,
            behind: 0,
            files: vec![],
        };

        for line in out.lines() {
            if let Some(rest) = line.strip_prefix("# branch.head ") {
                status.branch = rest.to_string();
            } else if let Some(rest) = line.strip_prefix("# branch.ab ") {
                // "+<ahead> -<behind>"
                for tok in rest.split_whitespace() {
                    if let Some(a) = tok.strip_prefix('+') {
                        status.ahead = a.parse().unwrap_or(0);
                    } else if let Some(b) = tok.strip_prefix('-') {
                        status.behind = b.parse().unwrap_or(0);
                    }
                }
            } else if let Some(rest) = line.strip_prefix("1 ") {
                // ordinary: XY sub mH mI mW hH hI path
                push_xy(&mut status.files, rest, 8, false);
            } else if let Some(rest) = line.strip_prefix("2 ") {
                // rename/copy: XY sub mH mI mW hH hI Xscore path\torig
                push_xy(&mut status.files, rest, 9, true);
            } else if let Some(path) = line.strip_prefix("? ") {
                status.files.push(GitFileDto {
                    path: path.replace('\\', "/"),
                    status: "?".into(),
                    staged: false,
                });
            } else if let Some(rest) = line.strip_prefix("u ") {
                // unmerged (conflict): XY ... path
                if let Some(path) = rest.rsplit(' ').next() {
                    status.files.push(GitFileDto {
                        path: path.replace('\\', "/"),
                        status: "U".into(),
                        staged: false,
                    });
                }
            }
        }
        Ok(status)
    })
    .await
    .map_err(|e| format!("git_status task failed to run: {e}"))?
}

/// Parse an ordinary/rename porcelain-v2 entry into staged/unstaged file rows.
fn push_xy(files: &mut Vec<GitFileDto>, rest: &str, field_count: usize, rename: bool) {
    let parts: Vec<&str> = rest.splitn(field_count, ' ').collect();
    if parts.len() < field_count {
        return;
    }
    let xy = parts[0];
    let mut x = xy.chars().next().unwrap_or('.');
    let y = xy.chars().nth(1).unwrap_or('.');
    // For renames X is 'R'; keep it.
    if rename && x == '.' {
        x = 'R';
    }
    // The path is the last field; for a rename it's "new\torig".
    let raw = parts[field_count - 1];
    let path = raw.split('\t').next().unwrap_or(raw).replace('\\', "/");

    if x != '.' {
        files.push(GitFileDto {
            path: path.clone(),
            status: x.to_string(),
            staged: true,
        });
    }
    if y != '.' {
        files.push(GitFileDto {
            path,
            status: y.to_string(),
            staged: false,
        });
    }
}

#[tauri::command]
pub async fn git_stage(
    app: AppHandle,
    project: State<'_, ProjectState>,
    repo: String,
    paths: Vec<String>,
) -> Result<(), String> {
    let root = confined_repo(&app, &project, &repo)?;
    let paths = confined_paths(&root, &paths)?;
    tauri::async_runtime::spawn_blocking(move || {
        let mut args = vec!["add", "--"];
        args.extend(paths.iter().map(String::as_str));
        run_git(&repo, &args, &[]).map(|_| ())
    })
    .await
    .map_err(|e| format!("git_stage task failed to run: {e}"))?
}

#[tauri::command]
pub async fn git_unstage(
    app: AppHandle,
    project: State<'_, ProjectState>,
    repo: String,
    paths: Vec<String>,
) -> Result<(), String> {
    let root = confined_repo(&app, &project, &repo)?;
    let paths = confined_paths(&root, &paths)?;
    tauri::async_runtime::spawn_blocking(move || {
        let mut args = vec!["restore", "--staged", "--"];
        args.extend(paths.iter().map(String::as_str));
        run_git(&repo, &args, &[]).map(|_| ())
    })
    .await
    .map_err(|e| format!("git_unstage task failed to run: {e}"))?
}

/// Discard working-tree changes (unstage + restore from HEAD). Untracked files
/// are removed.
#[tauri::command]
pub async fn git_discard(
    app: AppHandle,
    project: State<'_, ProjectState>,
    repo: String,
    paths: Vec<String>,
) -> Result<(), String> {
    // The one door here that DESTROYS work, and the one the finding named.
    let root = confined_repo(&app, &project, &repo)?;
    let paths = confined_paths(&root, &paths)?;
    tauri::async_runtime::spawn_blocking(move || {
        let mut args = vec!["restore", "--staged", "--worktree", "--"];
        args.extend(paths.iter().map(String::as_str));
        // Ignore code 1 (a purely-untracked path isn't known to restore).
        run_git(&repo, &args, &[1])?;
        // Sweep any still-present untracked paths.
        let mut clean = vec!["clean", "-fq", "--"];
        clean.extend(paths.iter().map(String::as_str));
        run_git(&repo, &clean, &[]).map(|_| ())
    })
    .await
    .map_err(|e| format!("git_discard task failed to run: {e}"))?
}

#[tauri::command]
pub async fn git_commit(
    app: AppHandle,
    project: State<'_, ProjectState>,
    repo: String,
    message: String,
) -> Result<String, String> {
    let _root = confined_repo(&app, &project, &repo)?;
    tauri::async_runtime::spawn_blocking(move || run_git(&repo, &["commit", "-m", &message], &[]))
        .await
        .map_err(|e| format!("git_commit task failed to run: {e}"))?
}

/// Unified diff of a single file (worktree vs index, or index vs HEAD if staged).
#[tauri::command]
pub async fn git_file_diff(
    app: AppHandle,
    project: State<'_, ProjectState>,
    repo: String,
    path: String,
    staged: bool,
) -> Result<String, String> {
    let root = confined_repo(&app, &project, &repo)?;
    confine_under(&root, &path)?;
    tauri::async_runtime::spawn_blocking(move || {
        let mut args = vec!["diff"];
        if staged {
            args.push("--cached");
        }
        args.push("--");
        args.push(&path);
        run_git(&repo, &args, &[])
    })
    .await
    .map_err(|e| format!("git_file_diff task failed to run: {e}"))?
}

#[tauri::command]
pub async fn git_branches(
    app: AppHandle,
    project: State<'_, ProjectState>,
    repo: String,
) -> Result<Vec<String>, String> {
    let _root = confined_repo(&app, &project, &repo)?;
    tauri::async_runtime::spawn_blocking(move || {
        let out = run_git(&repo, &["branch", "--format=%(refname:short)"], &[])?;
        Ok(out
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect())
    })
    .await
    .map_err(|e| format!("git_branches task failed to run: {e}"))?
}

#[tauri::command]
pub async fn git_init(
    app: AppHandle,
    project: State<'_, ProjectState>,
    repo: String,
) -> Result<(), String> {
    // `git init` creates the repo, so the DIRECTORY must exist and be confined —
    // the `.git` it makes inside is the new part.
    let _root = confined_repo(&app, &project, &repo)?;
    tauri::async_runtime::spawn_blocking(move || run_git(&repo, &["init"], &[]).map(|_| ()))
        .await
        .map_err(|e| format!("git_init task failed to run: {e}"))?
}
