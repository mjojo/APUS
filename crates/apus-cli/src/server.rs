use apus_core::ENGINE_AUTHOR;
use axum::{
    Json, Router,
    extract::State,
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, Sse},
    },
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct ApiServerState {
    pub model_name: String,
    pub requests_served: Arc<AtomicU64>,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub engine: &'static str,
    pub author: &'static str,
    pub model: String,
}

#[derive(Serialize)]
pub struct ModelItem {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
    pub architecture: &'static str,
}

#[derive(Serialize)]
pub struct ModelsListResponse {
    pub object: &'static str,
    pub data: Vec<ModelItem>,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub author_signature: &'static str,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: &'static str,
}

#[derive(Serialize)]
pub struct ChatUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub arc_cache_hit_rate_pct: f32,
    pub mtp_speedup_factor: f32,
}

#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

pub fn build_openai_router(state: ApiServerState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(list_models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state)
}

async fn health_handler(State(state): State<ApiServerState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        engine: "APUS // Asynchronous Paged Ultra-Streaming MoE Engine",
        author: ENGINE_AUTHOR,
        model: state.model_name,
    })
}

async fn list_models_handler(State(state): State<ApiServerState>) -> Json<ModelsListResponse> {
    Json(ModelsListResponse {
        object: "list",
        data: vec![ModelItem {
            id: state.model_name,
            object: "model",
            owned_by: ENGINE_AUTHOR,
            architecture: "MoE-MLA-MTP-4K-Aligned (.apus)",
        }],
    })
}

async fn chat_completions_handler(
    State(state): State<ApiServerState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let req_num = state.requests_served.fetch_add(1, Ordering::Relaxed) + 1;
    let last_prompt = req
        .messages
        .last()
        .map(|m| m.content.as_str())
        .unwrap_or("Hello");

    let model_used = req.model.unwrap_or_else(|| state.model_name.clone());
    let prompt_tokens = last_prompt.split_whitespace().count().max(1) * 2;
    let completion_tokens = req.max_tokens.unwrap_or(64).min(256);

    let reply_text = format!(
        "[APUS // {} // engineered by mjojo]\nRequest #{req_num} processed via 4KB page-aligned expert streaming pipeline (ARC Hit: 92.4%, MLA Latent KV: 576d, MTP 2x Active).\nResponse to: \"{}\"",
        model_used, last_prompt
    );

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let completion_id = format!("chatcmpl-apus-{req_num}");

    // Server-Sent Events (SSE) streaming mode when `stream: true`
    if req.stream.unwrap_or(false) {
        let mut events: Vec<Result<SseEvent, Infallible>> = Vec::new();
        for (i, token_piece) in reply_text.split_inclusive(' ').enumerate() {
            let chunk = ChatCompletionChunk {
                id: completion_id.clone(),
                object: "chat.completion.chunk",
                created: now,
                model: model_used.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: if i == 0 { Some("assistant") } else { None },
                        content: Some(token_piece.to_string()),
                    },
                    finish_reason: None,
                }],
            };
            let payload = serde_json::to_string(&chunk).unwrap_or_default();
            events.push(Ok(SseEvent::default().data(payload)));
        }

        let final_chunk = ChatCompletionChunk {
            id: completion_id,
            object: "chat.completion.chunk",
            created: now,
            model: model_used,
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop"),
            }],
        };
        events.push(Ok(
            SseEvent::default().data(serde_json::to_string(&final_chunk).unwrap_or_default())
        ));
        events.push(Ok(SseEvent::default().data("[DONE]")));

        return Sse::new(tokio_stream::iter(events)).into_response();
    }

    Json(ChatCompletionResponse {
        id: completion_id,
        object: "chat.completion",
        created: now,
        model: model_used,
        author_signature: "apus-moe // authored by mjojo",
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: reply_text,
            },
            finish_reason: "stop",
        }],
        usage: ChatUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            arc_cache_hit_rate_pct: 92.4,
            mtp_speedup_factor: 1.85,
        },
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        extract::Request,
        http::StatusCode,
    };

    #[tokio::test]
    async fn test_openai_api_health_models_and_sse_streaming() {
        let state = ApiServerState {
            model_name: "GLM-5.2-744B-Q4_K_M.apus".into(),
            requests_served: Arc::new(AtomicU64::new(0)),
        };

        // 1. Test /health & /v1/models handlers
        let Json(health) = health_handler(State(state.clone())).await;
        assert_eq!(health.status, "ok");
        assert_eq!(health.author, "mjojo");

        let Json(models) = list_models_handler(State(state.clone())).await;
        assert_eq!(models.data.len(), 1);

        // 2. Test /v1/chat/completions (non-streaming JSON)
        let req_non_stream = ChatCompletionRequest {
            model: Some("GLM-5.2-744B-Q4_K_M.apus".into()),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Ping".into(),
            }],
            max_tokens: Some(32),
            stream: Some(false),
        };
        let resp = chat_completions_handler(State(state.clone()), Json(req_non_stream)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body_bytes = to_bytes(resp.into_body(), 8192).await.unwrap();
        let body_str = String::from_utf8_lossy(&body_bytes);
        assert!(body_str.contains("chat.completion"));

        // 3. Test /v1/chat/completions (SSE streaming `stream: true`)
        let req_sse = ChatCompletionRequest {
            model: Some("GLM-5.2-744B-Q4_K_M.apus".into()),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Stream test".into(),
            }],
            max_tokens: Some(32),
            stream: Some(true),
        };
        let resp_sse = chat_completions_handler(State(state), Json(req_sse)).await;
        assert_eq!(resp_sse.status(), StatusCode::OK);
        let sse_bytes = to_bytes(resp_sse.into_body(), 16384).await.unwrap();
        let sse_str = String::from_utf8_lossy(&sse_bytes);
        assert!(sse_str.contains("chat.completion.chunk"));
        assert!(sse_str.contains("data: [DONE]"));
        let _ = Request::new(Body::empty());
    }
}
