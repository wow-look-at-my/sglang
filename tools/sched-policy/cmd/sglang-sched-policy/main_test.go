package main

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	ipc "github.com/wow-look-at-my/go-ipc"
	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/schedpolicy"
)

func TestRanksReachThePolicyOverChannels(t *testing.T) {
	name := fmt.Sprintf("sched-policy-test-%d", os.Getpid())
	done := make(chan error, 1)
	go func() { done <- run(name, 2, time.Second) }()

	var ranks []*ipc.Channel
	for rank := range 2 {
		var ch *ipc.Channel
		var err error
		for ch == nil {
			ch, err = ipc.OpenChannel(fmt.Sprintf("%s.r%d", name, rank))
			if err != nil {
				select {
				case serveErr := <-done:
					t.Fatalf("policy process exited: %v (open: %v)", serveErr, err)
				case <-time.After(10 * time.Millisecond):
				}
			}
		}
		defer ch.Close()
		ranks = append(ranks, ch)
	}

	ctx := context.Background()
	for rank, ch := range ranks {
		hello := schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, Rank: uint32(rank), World: 2}
		payload, err := hello.MarshalBinary()
		require.NoError(t, err)
		require.NoError(t, ch.SendTyped(ctx, schedpolicy.HelloType, payload))
	}
	for _, ch := range ranks {
		typ, _, err := ch.Recv(ctx)
		require.NoError(t, err)
		require.Equal(t, schedpolicy.HelloOkType, typ)
	}

	init := schedpolicy.Init{BurstTokens: 4096, Throttle: true}
	payload, err := init.MarshalBinary()
	require.NoError(t, err)
	require.NoError(t, ranks[0].SendTyped(ctx, schedpolicy.InitType, payload))
	pass := schedpolicy.BeginPass{}
	payload, err = pass.MarshalBinary()
	require.NoError(t, err)
	require.NoError(t, ranks[1].SendTyped(ctx, schedpolicy.BeginPassType, payload))
	for _, ch := range ranks {
		typ, body, err := ch.Recv(ctx)
		require.NoError(t, err)
		require.Equal(t, schedpolicy.ErrorType, typ)
		var e schedpolicy.Error
		require.NoError(t, e.UnmarshalBinary(body))
		require.Contains(t, e.Message, "ranks diverged")
	}
	require.ErrorContains(t, <-done, "ranks diverged")
}

func TestRunRejectsABadChannelName(t *testing.T) {
	require.Error(t, run("no/slashes", 1, time.Second))
}
