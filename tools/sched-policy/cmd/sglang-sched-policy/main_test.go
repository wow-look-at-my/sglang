package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	ipc "github.com/wow-look-at-my/go-ipc"
	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/schedpolicy"
)

// The ranks connect before the process serves and after; each hears the
// same failure as a call error, and the process exits once they leave.
func TestRanksReachThePolicyAsAService(t *testing.T) {
	name := fmt.Sprintf("sched-policy-test-%d", os.Getpid())
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	connected := make(chan *ipc.Client, 1)
	go func() {
		c, err := ipc.Connect(ctx, name)
		assert.Nil(t, err)

		connected <- c
	}()
	done := make(chan error, 1)
	go func() { done <- run(name, 2, time.Second) }()
	early := <-connected
	require.NotNil(t, early)
	late, err := ipc.Connect(ctx, name)
	require.NoError(t, err)
	ranks := []*ipc.Client{early, late}

	helloed := make(chan error, 2)
	for rank, c := range ranks {
		go func() {
			var ok schedpolicy.HelloOk
			hello := &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, Rank: uint32(rank), World: 2}
			helloed <- c.CallTyped(ctx, hello, &ok)
		}()
	}
	require.NoError(t, <-helloed)
	require.NoError(t, <-helloed)

	failed := make(chan error, 2)
	go func() {
		failed <- ranks[0].CallTyped(ctx, &schedpolicy.Init{BurstTokens: 4096, Throttle: true}, &schedpolicy.Ack{})
	}()
	go func() { failed <- ranks[1].CallTyped(ctx, &schedpolicy.BeginPass{}, &schedpolicy.Ack{}) }()
	for range ranks {
		err := <-failed
		var ce *ipc.CallError
		require.True(t, errors.As(err, &ce), "%v", err)
		require.Contains(t, ce.Message, "ranks diverged")
	}
	for _, c := range ranks {
		require.NoError(t, c.Close())
	}
	require.ErrorContains(t, <-done, "ranks diverged")
}

func TestRunRejectsABadServiceName(t *testing.T) {
	require.Error(t, run("no/slashes", 1, time.Second))
}

// The Python client speaks the protocol version this binary speaks.
func TestPythonClientSpeaksTheSameProtocol(t *testing.T) {
	src, err := os.ReadFile("../../../../python/sglang/srt/managers/scheduler_components/sched_policy.py")
	require.NoError(t, err)
	m := regexp.MustCompile(`(?m)^PROTOCOL_VERSION = (\d+)$`).FindSubmatch(src)
	require.NotNil(t, m, "sched_policy.py has no PROTOCOL_VERSION line")
	v, err := strconv.ParseUint(string(m[1]), 10, 32)
	require.NoError(t, err)
	require.Equal(t, schedpolicy.ProtocolVersion, uint32(v))
}
