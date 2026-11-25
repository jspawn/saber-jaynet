use saber_model::template::{ChatRequest, Renderer};

const TEMPLATE: &str = include_str!("../../../tools/templates/daisyrose_trailing_system.jinja");

fn render(messages: Vec<serde_json::Value>, tools: Option<Vec<serde_json::Value>>) -> String {
    let renderer = Renderer::from_source(TEMPLATE).unwrap();
    let mut extra = serde_json::Map::new();
    extra.insert("enable_thinking".into(), serde_json::Value::Bool(false));
    renderer
        .render(&ChatRequest {
            messages,
            tools,
            documents: None,
            add_generation_prompt: true,
            extra,
        })
        .unwrap()
}

#[test]
fn leading_system_rendering_is_unchanged() {
    let rendered = render(
        vec![
            serde_json::json!({"role": "system", "content": "base instructions"}),
            serde_json::json!({"role": "user", "content": "work"}),
        ],
        None,
    );
    assert_eq!(
        rendered,
        "<|im_start|>system\nbase instructions<|im_end|>\n\
         <|im_start|>user\nwork<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn trailing_system_messages_render_after_tool_history() {
    let rendered = render(
        vec![
            serde_json::json!({"role": "system", "content": "base instructions"}),
            serde_json::json!({"role": "user", "content": "weather?"}),
            serde_json::json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{"function": {
                    "name": "get_weather",
                    "arguments": {"city": "Berlin"}
                }}]
            }),
            serde_json::json!({"role": "tool", "content": "18 C"}),
            serde_json::json!({"role": "system", "content": "budget: iteration 2/8"}),
        ],
        None,
    );
    assert!(rendered.starts_with(
        "<|im_start|>system\nbase instructions<|im_end|>\n\
         <|im_start|>user\nweather?<|im_end|>\n"
    ));
    assert!(rendered.contains(
        "<|im_start|>user\n<tool_response>\n18 C\n</tool_response><|im_end|>\n\
         <|im_start|>system\nbudget: iteration 2/8<|im_end|>\n\
         <|im_start|>assistant\n"
    ));
}

#[test]
fn tool_prefix_stays_stable_when_only_the_trailing_anchor_changes() {
    let messages = vec![
        serde_json::json!({"role": "system", "content": "base instructions"}),
        serde_json::json!({"role": "user", "content": "work"}),
    ];
    let tools = Some(vec![serde_json::json!({
        "type": "function",
        "function": {"name": "code.run", "parameters": {"type": "object"}}
    })]);
    let first = render(messages.clone(), tools.clone());
    let mut anchored_messages = messages;
    anchored_messages.push(serde_json::json!({
        "role": "system", "content": "budget: iteration 1/8"
    }));
    let anchored = render(anchored_messages, tools);
    let prefix = first
        .strip_suffix("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        .unwrap();
    assert_eq!(
        anchored,
        format!(
            "{prefix}<|im_start|>system\nbudget: iteration 1/8<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    );
}
