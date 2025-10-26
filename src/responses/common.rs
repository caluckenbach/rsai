use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionToolCall {
    #[serde(rename = "type")]
    pub r#type: String,
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}
