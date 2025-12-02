//! Shared HTTP client with retry logic for all providers.

use std::time::Duration;

use serde::{Serialize, de::DeserializeOwned};
use tracing::{debug, warn};

use super::error::LlmError;

/// Configuration for HTTP client resilience
#[derive(Debug, Clone)]
pub struct HttpClientConfig {
    pub timeout: Duration,
    pub max_retries: u32,
    /// Base duration for exponential backoff
    pub initial_retry_delay: Duration,
    /// Cap on the backoff duration
    pub max_retry_delay: Duration,
}

impl Default for HttpClientConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            max_retries: 3,
            initial_retry_delay: Duration::from_millis(500),
            max_retry_delay: Duration::from_secs(10),
        }
    }
}

/// Shared HTTP client with retry logic and exponential backoff.
pub struct HttpClient {
    client: reqwest::Client,
    config: HttpClientConfig,
}

impl HttpClient {
    /// Create a new HTTP client with the given configuration.
    pub fn new(config: HttpClientConfig, user_agent: Option<&str>) -> Result<Self, LlmError> {
        let default_ua = format!("rsai/{}", env!("CARGO_PKG_VERSION"));
        let ua = user_agent.unwrap_or(&default_ua);

        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .user_agent(ua)
            .build()
            .map_err(|e| {
                LlmError::ProviderConfiguration(format!("Failed to build reqwest client: {e}"))
            })?;

        Ok(Self { client, config })
    }

    /// Make a POST request with JSON body and retry logic.
    ///
    /// Retries on 429 (rate limit) and 5xx errors with exponential backoff.
    /// Fails immediately on 4xx errors (except 429).
    #[tracing::instrument(
        name = "http_post_json",
        skip(self, headers, body),
        fields(url = %url),
        err
    )]
    pub async fn post_json<Req, Res>(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Req,
    ) -> Result<Res, LlmError>
    where
        Req: Serialize,
        Res: DeserializeOwned,
    {
        let mut last_error: Option<LlmError> = None;

        for attempt in 0..=self.config.max_retries {
            // Build request (must be rebuilt each attempt since .send() consumes it)
            let mut req_builder = self.client.post(url).json(body);

            // Add headers
            for (name, value) in headers {
                req_builder = req_builder.header(name, value);
            }

            match req_builder.send().await {
                Err(e) => {
                    warn!(attempt, error = %e, "HTTP request failed, retrying");
                    last_error = Some(LlmError::Network {
                        message: format!(
                            "Request failed (attempt {}/{})",
                            attempt + 1,
                            self.config.max_retries + 1
                        ),
                        source: Box::new(e),
                    });
                }
                Ok(res) => {
                    let status = res.status();

                    // Success
                    if status.is_success() {
                        debug!(status = %status, "HTTP request successful");
                        return res.json().await.map_err(|e| LlmError::Parse {
                            message: "Failed to parse API response".to_string(),
                            source: Box::new(e),
                        });
                    }

                    warn!(attempt, status = %status, "API returned error status");

                    let is_retryable = status == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || status.is_server_error();
                    let error_text = res
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());

                    if !is_retryable {
                        // Fatal errors - don't retry
                        return Err(LlmError::Api {
                            message: format!("Fatal API Error: {error_text}"),
                            status_code: Some(status.as_u16()),
                            source: None,
                        });
                    }

                    // Retryable error - capture and continue
                    last_error = Some(LlmError::Api {
                        message: format!("Transient API error ({}): {}", status, error_text),
                        status_code: Some(status.as_u16()),
                        source: None,
                    });
                }
            }

            // Exponential backoff with jitter
            if attempt < self.config.max_retries {
                let base_delay =
                    self.config.initial_retry_delay.as_millis() as f64 * 2_f64.powi(attempt as i32);

                // +/- 10% jitter (0.9 to 1.1)
                let jitter_factor = rand::random::<f64>() * 0.2 + 0.9;
                let delay_ms = (base_delay * jitter_factor) as u64;

                // Cap delay at max
                let delay =
                    std::time::Duration::from_millis(delay_ms).min(self.config.max_retry_delay);

                tokio::time::sleep(delay).await;
            }
        }

        Err(last_error.unwrap_or_else(|| LlmError::Api {
            message: format!(
                "Request failed after max retries ({}) with unknown error",
                self.config.max_retries
            ),
            status_code: None,
            source: None,
        }))
    }

    /// Get the underlying configuration
    pub fn config(&self) -> &HttpClientConfig {
        &self.config
    }
}
