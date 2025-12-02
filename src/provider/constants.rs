pub mod openai {
    pub const API_BASE: &str = "https://api.openai.com/v1";
    pub const RESPONSES_ENDPOINT: &str = "/responses";
    pub const API_KEY_ENV_VAR: &str = "OPENAI_API_KEY";
}

pub mod openrouter {
    pub const API_BASE: &str = "https://openrouter.ai/api/v1";
    pub const RESPONSES_ENDPOINT: &str = "/responses";
    pub const API_KEY_ENV_VAR: &str = "OPENROUTER_API_KEY";
}

pub mod gemini {
    pub const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";
    pub const API_KEY_ENV_VAR: &str = "GEMINI_API_KEY";
}
