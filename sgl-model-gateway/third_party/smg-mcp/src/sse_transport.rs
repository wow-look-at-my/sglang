//! Client side of the MCP HTTP+SSE transport (protocol revision).

use std::time::Duration;

use futures::{stream::BoxStream, StreamExt};
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use rmcp::{
    model::{ClientJsonRpcMessage, ServerJsonRpcMessage},
    transport::Transport,
    RoleClient,
};
use sse_stream::{Sse, SseStream};
use url::Url;

const EVENT_STREAM_MIME_TYPE: &str = "text/event-stream";
const HEADER_LAST_EVENT_ID: &str = "Last-Event-ID";
const MIN_RECONNECT_INTERVAL: Duration = Duration::from_secs(1);

type EventStream = BoxStream<'static, Result<Sse, sse_stream::Error>>;

#[derive(Debug, thiserror::Error)]
pub enum SseTransportError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("SSE error: {0}")]
    Sse(#[from] sse_stream::Error),
    #[error("unexpected content type: {0:?}")]
    UnexpectedContentType(Option<String>),
    #[error("SSE stream ended before the server sent its endpoint event")]
    MissingEndpoint,
    #[error("invalid URL {0:?}: {1}")]
    InvalidUrl(String, url::ParseError),
}

/// An rmcp client [`Transport`] over the legacy HTTP+SSE protocol.
pub struct SseClientTransport {
    client: reqwest::Client,
    sse_url: Url,
    // Refreshed by `endpoint` events, which servers resend when a stream reconnects.
    message_endpoint: Url,
    // `None` after a stream error until the next `receive` reopens it.
    stream: Option<EventStream>,
    last_event_id: Option<String>,
    server_retry: Option<Duration>,
    closed: bool,
}

impl SseClientTransport {
    /// Opens the event stream at `sse_url` and waits for the server's `endpoint` event.
    ///
    /// Authentication and proxying are taken from `client` (default headers, proxy config).
    pub async fn start(client: reqwest::Client, sse_url: &str) -> Result<Self, SseTransportError> {
        let sse_url = Url::parse(sse_url)
            .map_err(|e| SseTransportError::InvalidUrl(sse_url.to_string(), e))?;
        let mut stream = open_stream(&client, &sse_url, None).await?;
        let mut last_event_id = None;
        let mut server_retry = None;
        let endpoint = loop {
            let event = stream
                .next()
                .await
                .ok_or(SseTransportError::MissingEndpoint)??;
            record_event_metadata(&event, &mut last_event_id, &mut server_retry);
            if event.event.as_deref() == Some("endpoint") {
                break resolve_message_endpoint(&sse_url, event.data.as_deref().unwrap_or(""))?;
            }
        };
        Ok(Self {
            client,
            sse_url,
            message_endpoint: endpoint,
            stream: Some(stream),
            last_event_id,
            server_retry,
            closed: false,
        })
    }

    async fn reconnect(&mut self) {
        loop {
            match open_stream(&self.client, &self.sse_url, self.last_event_id.as_deref()).await {
                Ok(stream) => {
                    self.stream = Some(stream);
                    return;
                }
                Err(e) => {
                    let interval = self
                        .server_retry
                        .map_or(MIN_RECONNECT_INTERVAL, |r| r.max(MIN_RECONNECT_INTERVAL));
                    tracing::debug!(url = %self.sse_url, "SSE reconnect failed: {e}; retrying in {interval:?}");
                    tokio::time::sleep(interval).await;
                }
            }
        }
    }

    fn handle_control_event(&mut self, event: &Sse) {
        if event.event.as_deref() != Some("endpoint") {
            return;
        }
        let Some(data) = event.data.as_deref() else {
            return;
        };
        match resolve_message_endpoint(&self.sse_url, data) {
            Ok(endpoint) => self.message_endpoint = endpoint,
            Err(e) => tracing::warn!(url = %self.sse_url, "ignoring SSE endpoint event: {e}"),
        }
    }
}

impl Transport<RoleClient> for SseClientTransport {
    type Error = SseTransportError;

    fn send(
        &mut self,
        item: ClientJsonRpcMessage,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send + 'static {
        let client = self.client.clone();
        let endpoint = self.message_endpoint.clone();
        async move {
            client
                .post(endpoint)
                .json(&item)
                .send()
                .await?
                .error_for_status()?;
            Ok(())
        }
    }

    async fn receive(&mut self) -> Option<ServerJsonRpcMessage> {
        loop {
            if self.closed {
                return None;
            }
            let Some(stream) = self.stream.as_mut() else {
                self.reconnect().await;
                continue;
            };
            let event = match stream.next().await {
                Some(Ok(event)) => event,
                Some(Err(e)) => {
                    tracing::warn!(
                        url = %self.sse_url,
                        last_event_id = self.last_event_id.as_deref().unwrap_or(""),
                        "SSE stream error: {e}"
                    );
                    self.stream = None;
                    continue;
                }
                None => {
                    tracing::debug!(url = %self.sse_url, "SSE stream terminated");
                    self.closed = true;
                    return None;
                }
            };
            record_event_metadata(&event, &mut self.last_event_id, &mut self.server_retry);
            if !matches!(event.event.as_deref(), None | Some("") | Some("message")) {
                self.handle_control_event(&event);
                continue;
            }
            let Some(data) = event.data else {
                continue;
            };
            match serde_json::from_str::<ServerJsonRpcMessage>(&data) {
                Ok(message) => return Some(message),
                Err(e) => tracing::debug!(
                    last_event_id = self.last_event_id.as_deref().unwrap_or(""),
                    "failed to deserialize server message: {e}"
                ),
            }
        }
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        self.closed = true;
        self.stream = None;
        Ok(())
    }
}

async fn open_stream(
    client: &reqwest::Client,
    url: &Url,
    last_event_id: Option<&str>,
) -> Result<EventStream, SseTransportError> {
    let mut request = client
        .get(url.clone())
        .header(ACCEPT, EVENT_STREAM_MIME_TYPE);
    if let Some(id) = last_event_id {
        request = request.header(HEADER_LAST_EVENT_ID, id);
    }
    let response = request.send().await?.error_for_status()?;
    let content_type = response.headers().get(CONTENT_TYPE);
    if !content_type.is_some_and(|ct| ct.as_bytes().starts_with(EVENT_STREAM_MIME_TYPE.as_bytes()))
    {
        return Err(SseTransportError::UnexpectedContentType(
            content_type.map(|ct| String::from_utf8_lossy(ct.as_bytes()).into_owned()),
        ));
    }
    Ok(SseStream::from_bytes_stream(response.bytes_stream()).boxed())
}

fn record_event_metadata(
    event: &Sse,
    last_event_id: &mut Option<String>,
    server_retry: &mut Option<Duration>,
) {
    if let Some(id) = &event.id {
        *last_event_id = Some(id.clone());
    }
    if let Some(retry_ms) = event.retry {
        *server_retry = Some(Duration::from_millis(retry_ms));
    }
}

/// replaces path and query on the SSE URL's origin.
fn resolve_message_endpoint(sse_url: &Url, endpoint: &str) -> Result<Url, SseTransportError> {
    let invalid = |e| SseTransportError::InvalidUrl(endpoint.to_string(), e);
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        return Url::parse(endpoint).map_err(invalid);
    }
    let mut url = sse_url.clone();
    url.set_fragment(None);
    if let Some(query) = endpoint.strip_prefix('?') {
        url.set_query(Some(query));
        return Ok(url);
    }
    let (path, query) = match endpoint.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (endpoint, None),
    };
    if path.starts_with('/') {
        url.set_path(path);
    } else {
        url.set_path(&format!("/{path}"));
    }
    url.set_query(query);
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_message_endpoint() {
        let base = Url::parse("https://localhost/sse").unwrap();
        let resolve = |ep: &str| resolve_message_endpoint(&base, ep).unwrap().to_string();

        assert_eq!(resolve("?sessionId=x"), "https://localhost/sse?sessionId=x");
        assert_eq!(
            resolve("mypath?sessionId=x"),
            "https://localhost/mypath?sessionId=x"
        );
        assert_eq!(
            resolve("/xxx?sessionId=x"),
            "https://localhost/xxx?sessionId=x"
        );
        assert_eq!(
            resolve("http://example.com/xxx?sessionId=x"),
            "http://example.com/xxx?sessionId=x"
        );
        // A scheme-relative payload must not move the session to another host.
        assert_eq!(
            resolve("//evil.example/x"),
            "https://localhost//evil.example/x"
        );
    }
}
