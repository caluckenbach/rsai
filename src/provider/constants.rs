pub mod openai {
    pub const DEFAULT_MODEL: &str = "gpt-4";
    pub const API_BASE: &str = "https://api.openai.com/v1";
    pub const RESPONSES_ENDPOINT: &str = "/responses";
    pub const API_KEY_ENV_VAR: &str = "OPENAI_API_KEY";
}
