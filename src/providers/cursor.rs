//! Cursor headless non-interactive CLI provider.
//!
//! Integrates with Cursor's headless CLI mode, spawning the `cursor` binary
//! as a subprocess for each inference request. This allows using Cursor's AI
//! models without an interactive UI session.
//!
//! # Usage
//!
//! The `cursor` binary must be available in `PATH`, or its location must be
//! set via the `CURSOR_PATH` environment variable.
//!
//! Cursor is invoked as:
//! ```text
//! cursor --headless --model <model> <message>
//! ```
//!
//! If the model argument is `"default"` or empty, the `--model` flag is omitted
//! and Cursor's own default model is used.
//!
//! # Limitations
//!
//! - **Conversation history**: Only the system prompt (if present) and the last
//!   user message are forwarded. Full multi-turn history is not preserved because
//!   Cursor's headless CLI accepts a single prompt per invocation.
//! - **System prompt**: The system prompt is prepended to the user message with a
//!   blank-line separator, as the headless CLI does not provide a dedicated
//!   system-prompt flag.
//! - **Temperature**: Cursor's headless CLI does not expose a temperature parameter.
//!   Any temperature value supplied by the caller is silently ignored.
//!
//! # Authentication
//!
//! Authentication is handled by Cursor itself (its own credential store).
//! No explicit API key is required by this provider.
//!
//! # Environment variables
//!
//! - `CURSOR_PATH` — override the path to the `cursor` binary (default: `"cursor"`)

use crate::providers::traits::{ChatRequest, ChatResponse, Provider, TokenUsage};
use async_trait::async_trait;
use std::path::PathBuf;
use tokio::process::Command;

/// Environment variable for overriding the path to the `cursor` binary.
pub const CURSOR_PATH_ENV: &str = "CURSOR_PATH";

/// Default `cursor` binary name (resolved via `PATH`).
const DEFAULT_CURSOR_BINARY: &str = "cursor";

/// Model name used to signal "use Cursor's own default model".
const DEFAULT_MODEL_MARKER: &str = "default";

/// Provider that invokes the Cursor headless CLI as a subprocess.
///
/// Each inference request spawns a fresh `cursor` process. This is the
/// non-interactive approach: Cursor processes the prompt and exits.
pub struct CursorProvider {
    /// Path to the `cursor` binary.
    cursor_path: PathBuf,
}

impl CursorProvider {
    /// Create a new `CursorProvider`.
    ///
    /// The binary path is resolved from `CURSOR_PATH` env var if set,
    /// otherwise defaults to `"cursor"` (found via `PATH`).
    pub fn new() -> Self {
        let cursor_path = std::env::var(CURSOR_PATH_ENV)
            .ok()
            .filter(|path| !path.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_CURSOR_BINARY));

        Self { cursor_path }
    }

    /// Returns true if the model argument should be forwarded to cursor.
    fn should_forward_model(model: &str) -> bool {
        let trimmed = model.trim();
        !trimmed.is_empty() && trimmed != DEFAULT_MODEL_MARKER
    }

    /// Invoke the cursor binary with the given prompt and optional model.
    /// Returns the trimmed stdout output as the assistant response.
    async fn invoke_cursor(&self, message: &str, model: &str) -> anyhow::Result<String> {
        let mut cmd = Command::new(&self.cursor_path);
        cmd.arg("--headless");

        if Self::should_forward_model(model) {
            cmd.arg("--model").arg(model);
        }

        cmd.arg(message);

        // Inherit stderr so that Cursor's diagnostic output is visible to the
        // operator, but capture stdout for the model response.
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::inherit());

        let output = cmd.output().await.map_err(|err| {
            anyhow::anyhow!(
                "Failed to spawn Cursor binary at {:?}: {err}. \
                 Ensure `cursor` is installed and in PATH, or set CURSOR_PATH.",
                self.cursor_path
            )
        })?;

        if !output.status.success() {
            let code = output.status.code().unwrap_or(-1);
            anyhow::bail!(
                "Cursor exited with non-zero status {code}. \
                 Check that Cursor is authenticated and the headless CLI is supported."
            );
        }

        let text = String::from_utf8(output.stdout).map_err(|err| {
            anyhow::anyhow!("Cursor produced non-UTF-8 output: {err}")
        })?;

        Ok(text.trim().to_string())
    }
}

impl Default for CursorProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for CursorProvider {
    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        // Cursor's headless CLI does not support a temperature parameter; ignored.
        _temperature: f64,
    ) -> anyhow::Result<String> {
        // Prepend the system prompt to the user message with a blank-line separator.
        // Cursor's headless CLI does not expose a dedicated system-prompt flag.
        let full_message = match system_prompt {
            Some(system) if !system.is_empty() => {
                format!("{system}\n\n{message}")
            }
            _ => message.to_string(),
        };

        self.invoke_cursor(&full_message, model).await
    }

    async fn chat(
        &self,
        request: ChatRequest<'_>,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ChatResponse> {
        let text = self
            .chat_with_history(request.messages, model, temperature)
            .await?;

        Ok(ChatResponse {
            text: Some(text),
            tool_calls: Vec::new(),
            usage: Some(TokenUsage::default()),
            reasoning_content: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env lock poisoned")
    }

    #[test]
    fn new_uses_env_override() {
        let _guard = env_lock();
        let orig = std::env::var(CURSOR_PATH_ENV).ok();
        std::env::set_var(CURSOR_PATH_ENV, "/usr/local/bin/cursor");
        let provider = CursorProvider::new();
        assert_eq!(provider.cursor_path, PathBuf::from("/usr/local/bin/cursor"));
        match orig {
            Some(v) => std::env::set_var(CURSOR_PATH_ENV, v),
            None => std::env::remove_var(CURSOR_PATH_ENV),
        }
    }

    #[test]
    fn new_defaults_to_cursor() {
        let _guard = env_lock();
        let orig = std::env::var(CURSOR_PATH_ENV).ok();
        std::env::remove_var(CURSOR_PATH_ENV);
        let provider = CursorProvider::new();
        assert_eq!(provider.cursor_path, PathBuf::from("cursor"));
        if let Some(v) = orig {
            std::env::set_var(CURSOR_PATH_ENV, v);
        }
    }

    #[test]
    fn new_ignores_blank_env_override() {
        let _guard = env_lock();
        let orig = std::env::var(CURSOR_PATH_ENV).ok();
        std::env::set_var(CURSOR_PATH_ENV, "   ");
        let provider = CursorProvider::new();
        assert_eq!(provider.cursor_path, PathBuf::from("cursor"));
        match orig {
            Some(v) => std::env::set_var(CURSOR_PATH_ENV, v),
            None => std::env::remove_var(CURSOR_PATH_ENV),
        }
    }

    #[test]
    fn should_forward_model_standard() {
        assert!(CursorProvider::should_forward_model("claude-3.5-sonnet"));
        assert!(CursorProvider::should_forward_model("gpt-4o"));
    }

    #[test]
    fn should_not_forward_default_model() {
        assert!(!CursorProvider::should_forward_model(DEFAULT_MODEL_MARKER));
        assert!(!CursorProvider::should_forward_model(""));
        assert!(!CursorProvider::should_forward_model("   "));
    }

    #[tokio::test]
    async fn invoke_missing_binary_returns_error() {
        let provider = CursorProvider {
            cursor_path: PathBuf::from("/nonexistent/path/to/cursor"),
        };
        let result = provider.invoke_cursor("hello", "gpt-4o").await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Failed to spawn Cursor binary"),
            "unexpected error message: {msg}"
        );
    }
}
