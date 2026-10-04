// tests/common/mock_mcp_server.rs - Mock MCP server for testing
use std::{collections::HashMap, convert::Infallible, sync::Arc};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json,
};
use futures::{channel::mpsc, Stream, StreamExt};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    service::RequestContext,
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpService,
    },
    ErrorData as McpError, RoleServer, ServerHandler, ServiceExt,
};
use tokio::net::TcpListener;

/// Mock MCP server that returns hardcoded responses for testing
pub struct MockMCPServer {
    pub port: u16,
    pub server_handle: Option<tokio::task::JoinHandle<()>>,
    path: &'static str,
}

// Legacy SSE sessions: sessionId to the sender feeding that session's rmcp server.
type SseSessions =
    Arc<parking_lot::Mutex<HashMap<String, mpsc::UnboundedSender<ClientJsonRpcMessage>>>>;

/// Simple test server with mock search tools
#[derive(Clone)]
pub struct MockSearchServer {
    tool_router: ToolRouter<MockSearchServer>,
}

impl Default for MockSearchServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_router]
impl MockSearchServer {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Mock web search tool")]
    fn brave_web_search(
        &self,
        Parameters(params): Parameters<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<CallToolResult, McpError> {
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("test");
        Ok(CallToolResult::success(vec![ContentBlock::text(format!(
            "Mock search results for: {}",
            query
        ))]))
    }

    #[tool(description = "Mock local search tool")]
    fn brave_local_search(
        &self,
        Parameters(_params): Parameters<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![ContentBlock::text(
            "Mock local search results",
        )]))
    }
}

#[tool_handler]
impl ServerHandler for MockSearchServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::V_2024_11_05)
            .with_instructions("Mock server for testing")
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, McpError> {
        Ok(self.get_info())
    }
}

impl MockMCPServer {
    /// Start a mock MCP server on an available port
    pub async fn start() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Find an available port
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();

        // Create the MCP service using rmcp's StreamableHttpService
        let service = StreamableHttpService::new(
            || Ok(MockSearchServer::new()),
            LocalSessionManager::default().into(),
            Default::default(),
        );

        let app = axum::Router::new().nest_service("/mcp", service);

        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("Mock MCP server failed to start");
        });

        // Give the server a moment to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(MockMCPServer {
            port,
            server_handle: Some(server_handle),
            path: "/mcp",
        })
    }

    /// Start a mock server speaking the legacy HTTP+SSE transport (protocol 2024-11-05):
    /// `GET /sse` opens the event stream, `POST /message?sessionId=..` carries requests.
    pub async fn start_sse() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();

        let sessions: SseSessions = Arc::default();
        let app = axum::Router::new()
            .route("/sse", get(sse_connect))
            .route("/message", post(sse_post_message))
            .with_state(sessions);

        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("Mock SSE MCP server failed to start");
        });

        Ok(MockMCPServer {
            port,
            server_handle: Some(server_handle),
            path: "/sse",
        })
    }

    /// Get the full URL for this mock server
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}{}", self.port, self.path)
    }

    /// Stop the mock server
    pub async fn stop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
            // Wait a moment for cleanup
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
    }
}

async fn sse_connect(
    State(sessions): State<SseSessions>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let session_id = uuid::Uuid::new_v4().to_string();
    let (to_server, from_client) = mpsc::unbounded::<ClientJsonRpcMessage>();
    let (to_client, from_server) = mpsc::unbounded::<ServerJsonRpcMessage>();
    sessions.lock().insert(session_id.clone(), to_server);

    tokio::spawn(async move {
        if let Ok(running) = MockSearchServer::new()
            .serve((to_client, from_client))
            .await
        {
            let _ = running.waiting().await;
        }
    });

    let endpoint = Event::default()
        .event("endpoint")
        .data(format!("/message?sessionId={session_id}"));
    let messages = from_server.map(|message| {
        Event::default()
            .event("message")
            .data(serde_json::to_string(&message).expect("serialize server message"))
    });
    Sse::new(
        futures::stream::once(async { endpoint })
            .chain(messages)
            .map(Ok),
    )
}

async fn sse_post_message(
    State(sessions): State<SseSessions>,
    Query(query): Query<HashMap<String, String>>,
    Json(message): Json<ClientJsonRpcMessage>,
) -> StatusCode {
    let sender = query
        .get("sessionId")
        .and_then(|id| sessions.lock().get(id).cloned());
    match sender {
        Some(sender) if sender.unbounded_send(message).is_ok() => StatusCode::ACCEPTED,
        _ => StatusCode::NOT_FOUND,
    }
}

impl Drop for MockMCPServer {
    fn drop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::MockMCPServer;

    #[tokio::test]
    async fn test_mock_server_startup() {
        let mut server = MockMCPServer::start().await.unwrap();
        assert!(server.port > 0);
        assert!(server.url().contains(&server.port.to_string()));
        server.stop().await;
    }

    #[tokio::test]
    async fn test_mock_server_with_rmcp_client() {
        let mut server = MockMCPServer::start().await.unwrap();

        use rmcp::{transport::StreamableHttpClientTransport, ServiceExt};

        let transport = StreamableHttpClientTransport::from_uri(server.url().as_str());
        let client = ().serve(transport).await;

        assert!(client.is_ok(), "Should be able to connect to mock server");

        if let Ok(client) = client {
            let tools = client.peer().list_all_tools().await;
            assert!(tools.is_ok(), "Should be able to list tools");

            if let Ok(tools) = tools {
                assert_eq!(tools.len(), 2, "Should have 2 tools");
                assert!(tools.iter().any(|t| t.name == "brave_web_search"));
                assert!(tools.iter().any(|t| t.name == "brave_local_search"));
            }

            // Shutdown by dropping the client
            drop(client);
        }

        server.stop().await;
    }
}
