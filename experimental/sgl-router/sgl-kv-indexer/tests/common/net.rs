// SPDX-FileCopyrightText: Copyright (c) The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;

/// Reserves an ephemeral loopback port.
pub fn free_addr() -> SocketAddr {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}
