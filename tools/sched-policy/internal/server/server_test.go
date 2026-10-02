package server

import (
	"context"
	"encoding/binary"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/schedpolicy"
)

type frame struct {
	typ     uint32
	payload []byte
}

// pipe is an in-memory Endpoint: the test writes requests into in and reads
// replies from out.
type pipe struct {
	in  chan frame
	out chan frame
}

func newPipe() *pipe { return &pipe{in: make(chan frame, 16), out: make(chan frame, 16)} }

func (p *pipe) Recv(ctx context.Context) (uint32, []byte, error) {
	select {
	case f := <-p.in:
		return f.typ, f.payload, nil
	case <-ctx.Done():
		return 0, nil, ctx.Err()
	}
}

func (p *pipe) SendTyped(ctx context.Context, typ uint32, payload []byte) error {
	p.out <- frame{typ, payload}
	return nil
}

type rig struct {
	t     *testing.T
	ranks []*pipe
	done  chan error
}

func start(t *testing.T, world int, timeout time.Duration) *rig {
	r := &rig{t: t, done: make(chan error, 1)}
	eps := make([]Endpoint, world)
	for i := range world {
		p := newPipe()
		r.ranks = append(r.ranks, p)
		eps[i] = p
	}
	go func() { r.done <- New(eps, timeout).Serve(context.Background()) }()
	return r
}

func (r *rig) send(rank int, m encoder, typ uint32) {
	payload, err := m.MarshalBinary()
	if err != nil {
		r.t.Fatal(err)
	}
	r.ranks[rank].in <- frame{typ, payload}
}

func (r *rig) call(m encoder, typ uint32) frame {
	for rank := range r.ranks {
		r.send(rank, m, typ)
	}
	return r.replies()
}

func (r *rig) replies() frame {
	first := <-r.ranks[0].out
	for _, p := range r.ranks[1:] {
		f := <-p.out
		if f.typ != first.typ || string(f.payload) != string(first.payload) {
			r.t.Fatal("ranks received different replies")
		}
	}
	return first
}

func (r *rig) hello() {
	for rank := range r.ranks {
		r.send(rank, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, Rank: uint32(rank), World: uint32(len(r.ranks))}, schedpolicy.HelloType)
	}
	if f := r.replies(); f.typ != schedpolicy.HelloOkType {
		r.t.Fatalf("hello reply type %d", f.typ)
	}
}

func (r *rig) failure() string {
	f := r.replies()
	if f.typ != schedpolicy.ErrorType {
		r.t.Fatalf("reply type %d, want error", f.typ)
	}
	var e schedpolicy.Error
	if err := e.UnmarshalBinary(f.payload); err != nil {
		r.t.Fatal(err)
	}
	if err := <-r.done; err == nil || !strings.Contains(err.Error(), e.Message) {
		r.t.Fatalf("Serve returned %v, ranks were told %q", err, e.Message)
	}
	return e.Message
}

func boolReply(t *testing.T, f frame) bool {
	t.Helper()
	require.Equal(t, schedpolicy.BoolReplyType, f.typ)

	var b schedpolicy.BoolReply
	require.NoError(t, b.UnmarshalBinary(f.payload))

	return b.Value
}

func tokenBytes(from, n int) []byte {
	b := make([]byte, 4*n)
	for i := range n {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(from+i))
	}
	return b
}

func TestEveryOpRoundTripsAndRankZeroDecides(t *testing.T) {
	r := start(t, 2, time.Second)
	r.hello()
	f := r.call(&schedpolicy.Init{BurstTokens: 4096, DeviceTokens: 1000, HostTokens: 0, Throttle: true}, schedpolicy.InitType)
	require.Equal(t, schedpolicy.AckType, f.typ)

	require.False(t, boolReply(t, r.call(&schedpolicy.ShouldDeferPrefill{PrefillPending: true, DecodeRunnable: true}, schedpolicy.ShouldDeferPrefillType)))

	f = r.call(&schedpolicy.PrefillTokenBudget{}, schedpolicy.PrefillTokenBudgetType)
	var budget schedpolicy.OptIntReply
	err := budget.UnmarshalBinary(f.payload)
	require.False(t, err != nil || budget.Present)

	r.send(0, &schedpolicy.BatchLaunched{BatchClass: 1, Tokens: 1000, Now: 0}, schedpolicy.BatchLaunchedType)
	r.send(1, &schedpolicy.BatchLaunched{BatchClass: 1, Tokens: 1000, Now: 50}, schedpolicy.BatchLaunchedType)
	r.replies()
	r.send(0, &schedpolicy.BatchFinished{Now: 1}, schedpolicy.BatchFinishedType)
	r.send(1, &schedpolicy.BatchFinished{Now: 99}, schedpolicy.BatchFinishedType)
	r.replies()
	require.True(t, boolReply(t, r.call(&schedpolicy.ShouldDeferPrefill{PrefillPending: true, DecodeRunnable: true}, schedpolicy.ShouldDeferPrefillType)))

	f = r.call(&schedpolicy.PrefillTokenBudget{}, schedpolicy.PrefillTokenBudgetType)
	err = budget.UnmarshalBinary(f.payload)
	require.False(t, err != nil || !budget.Present || budget.Value != 4096-1000)

	r.call(&schedpolicy.RequestQueued{Rid: "a", Tokens: tokenBytes(0, 400), Now: 1}, schedpolicy.RequestQueuedType)
	r.call(&schedpolicy.RequestFinished{Rid: "a", Length: 410, Tail: tokenBytes(346, 64), Now: 2}, schedpolicy.RequestFinishedType)
	r.call(&schedpolicy.RequestQueued{Rid: "b", Tokens: tokenBytes(10000, 400), Now: 2}, schedpolicy.RequestQueuedType)
	r.call(&schedpolicy.RequestFinished{Rid: "b", Length: 410, Tail: tokenBytes(10346, 64), Now: 3}, schedpolicy.RequestFinishedType)
	r.call(&schedpolicy.RequestQueued{Rid: "a2", Tokens: tokenBytes(0, 420), Now: 3}, schedpolicy.RequestQueuedType)
	r.call(&schedpolicy.RequestQueued{Rid: "c", Tokens: tokenBytes(50000, 300), Now: 3}, schedpolicy.RequestQueuedType)
	r.call(&schedpolicy.BeginPass{Present: "a2\nc"}, schedpolicy.BeginPassType)
	r.send(0, &schedpolicy.ShouldHold{Rid: "c", InputLen: 300, TotalTokens: 350, WouldEvict: true, QueuedAt: 3, Now: 3}, schedpolicy.ShouldHoldType)
	r.send(1, &schedpolicy.ShouldHold{Rid: "c", InputLen: 300, TotalTokens: 350, WouldEvict: false, QueuedAt: 3, Now: 3}, schedpolicy.ShouldHoldType)
	require.True(t, boolReply(t, r.replies()))

	f = r.call(&schedpolicy.Admitted{Evicted: true, Now: 3}, schedpolicy.AdmittedType)
	require.Equal(t, schedpolicy.AckType, f.typ)

}

func TestInitWithoutThrottleRejectsShouldHold(t *testing.T) {
	r := start(t, 1, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{BurstTokens: -1}, schedpolicy.InitType)
	r.call(&schedpolicy.RequestQueued{Rid: "a"}, schedpolicy.RequestQueuedType)
	r.call(&schedpolicy.BeginPass{}, schedpolicy.BeginPassType)
	r.send(0, &schedpolicy.ShouldHold{Rid: "a"}, schedpolicy.ShouldHoldType)
	msg := r.failure()
	require.Contains(t, msg, "without an eviction throttle")

}

func TestProtocolMismatchIsRefused(t *testing.T) {
	r := start(t, 1, time.Second)
	r.send(0, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion + 1, World: 1}, schedpolicy.HelloType)
	msg := r.failure()
	require.Contains(t, msg, "speaks protocol")

}

func TestWrongRankOrWorldIsRefused(t *testing.T) {
	r := start(t, 1, time.Second)
	r.send(0, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, Rank: 1, World: 2}, schedpolicy.HelloType)
	msg := r.failure()
	require.Contains(t, msg, "connected as rank 1 of 2")

}

func TestRequestBeforeHelloIsRefused(t *testing.T) {
	r := start(t, 1, time.Second)
	r.send(0, &schedpolicy.Init{}, schedpolicy.InitType)
	msg := r.failure()
	require.Contains(t, msg, "before hello")

}

func TestRequestBeforeInitAndDoubleInitAreRefused(t *testing.T) {
	r := start(t, 1, time.Second)
	r.hello()
	r.send(0, &schedpolicy.BeginPass{}, schedpolicy.BeginPassType)
	msg := r.failure()
	require.Contains(t, msg, "before init")

	r = start(t, 1, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{}, schedpolicy.InitType)
	r.send(0, &schedpolicy.Init{}, schedpolicy.InitType)
	msg = r.failure()
	require.Contains(t, msg, "init sent twice")

}

func TestMalformedAndUnknownMessagesAreRefused(t *testing.T) {
	r := start(t, 1, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{}, schedpolicy.InitType)
	r.ranks[0].in <- frame{schedpolicy.BatchLaunchedType, []byte{1}}
	msg := r.failure()
	require.Contains(t, msg, "shorter than the fixed section")

	r = start(t, 1, time.Second)
	r.hello()
	r.ranks[0].in <- frame{schedpolicy.AckType, nil}
	msg = r.failure()
	require.Contains(t, msg, "not a request")

	r = start(t, 1, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{}, schedpolicy.InitType)
	r.send(0, &schedpolicy.BatchLaunched{BatchClass: 7}, schedpolicy.BatchLaunchedType)
	msg = r.failure()
	require.Contains(t, msg, "batch class 7")

	r = start(t, 1, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{}, schedpolicy.InitType)
	r.send(0, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, World: 1}, schedpolicy.HelloType)
	msg = r.failure()
	require.Contains(t, msg, "after hello")

}

func TestRanksThatDivergeAreToldSoInsteadOfHanging(t *testing.T) {
	r := start(t, 2, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{Throttle: true}, schedpolicy.InitType)
	r.send(0, &schedpolicy.BeginPass{Present: "a"}, schedpolicy.BeginPassType)
	r.send(1, &schedpolicy.ShouldHold{Rid: "a"}, schedpolicy.ShouldHoldType)
	msg := r.failure()
	require.Contains(t, msg, "ranks diverged")

	// Same op, different replicated argument.
	r = start(t, 2, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{Throttle: true}, schedpolicy.InitType)
	r.send(0, &schedpolicy.BeginPass{Present: "a"}, schedpolicy.BeginPassType)
	r.send(1, &schedpolicy.BeginPass{Present: "b"}, schedpolicy.BeginPassType)
	msg = r.failure()
	require.Contains(t, msg, "ranks diverged")

}

func TestARankThatNeverSendsTripsTheLockstepTimeout(t *testing.T) {
	r := start(t, 2, 50*time.Millisecond)
	r.send(0, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, World: 2}, schedpolicy.HelloType)
	msg := r.failure()
	require.Contains(t, msg, "lockstep timeout: rank(s) [1] did not send")

}

func TestAnEndpointErrorEndsServe(t *testing.T) {
	r := start(t, 1, time.Second)
	r.hello()
	close(r.ranks[0].in)
	// A closed inbox makes Recv return a zero frame forever; decode refuses it.
	msg := r.failure()
	require.Contains(t, msg, "not a request")

}
