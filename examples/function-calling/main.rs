use rsai::{ApiKey, ChatRole, Message, completion_schema, llm};
use rsai_macros::{tool, toolset};
use dotenv::dotenv;

#[tool]
/// Get current weather for a city
/// city: The city to get weather for
/// unit: Temperature unit (celsius or fahrenheit)
fn get_weather(city: String, unit: Option<String>) -> String {
    let temp_unit = unit.unwrap_or_else(|| "celsius".to_string());
    match city.to_lowercase().as_str() {
        "london" => format!(
            "Weather in {}: 15°{}",
            city,
            if temp_unit == "fahrenheit" { "F" } else { "C" }
        ),
        "tokyo" => format!(
            "Weather in {}: 22°{}",
            city,
            if temp_unit == "fahrenheit" { "F" } else { "C" }
        ),
        "new york" => format!(
            "Weather in {}: 18°{}",
            city,
            if temp_unit == "fahrenheit" { "F" } else { "C" }
        ),
        _ => format!(
            "Weather in {}: 20°{}",
            city,
            if temp_unit == "fahrenheit" { "F" } else { "C" }
        ),
    }
}

#[tool]
/// Calculate distance between two cities
/// from: Starting city
/// to: Destination city
fn calculate_distance(from: String, to: String) -> f64 {
    match (from.to_lowercase().as_str(), to.to_lowercase().as_str()) {
        ("london", "tokyo") | ("tokyo", "london") => 9600.0,
        ("london", "new york") | ("new york", "london") => 5500.0,
        ("tokyo", "new york") | ("new york", "tokyo") => 10800.0,
        _ => 1000.0, // Default distance
    }
}

#[completion_schema]
struct Weather {
    city: String,
    temperature: f64,
    description: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let tools = toolset![get_weather, calculate_distance];

    let messages = vec![
        Message {
            role: ChatRole::System,
            content: "You are a helpful assistant. Use the available tools to gather information, then provide a structured Weather response for the requested city.".to_string(),
        },
        Message {
            role: ChatRole::User,
            content: "What's the weather like in Tokyo?".to_string(),
        }
    ];

    let response = llm::with("openai")?
        .api_key(ApiKey::Default)?
        .model("gpt-4o-mini")
        .messages(messages)
        .tools(tools)
        .complete::<Weather>()
        .await?;

    let weather = response.content;
    println!(
        "Weather in {}: {}°, {}",
        weather.city, weather.temperature, weather.description
    );

    Ok(())
}
