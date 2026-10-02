package server

import (
	"encoding/binary"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	ipc "github.com/wow-look-at-my/go-ipc"
	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/schedpolicy"
)

// A frame is one reply as the handler returns it.
type frame struct {
	typ     uint32
	payload []byte
	err     error
}

// A rig drives the handler directly: each rank is a session whose calls run
// on goroutines, as the service layer runs them.
type rig struct {
	t        *testing.T
	s        *Server
	sessions []*ipc.Session
	// pending holds the reply of each rank's call in flight.
	pending []chan frame
}

func start(t *testing.T, world int, timeout time.Duration) *rig {
	r := &rig{t: t, s: New(world, timeout)}
	for range world {
		r.sessions = append(r.sessions, &ipc.Session{})
		r.pending = append(r.pending, make(chan frame, 1))
	}
	return r
}

// send makes rank's call on its own goroutine, as the service layer does.
func (r *rig) send(rank int, m schedpolicy.Message) {
	r.sendRaw(rank, m.TypeID(), r.encode(m))
}

func (r *rig) sendRaw(rank int, typ uint32, payload []byte) {
	go func() {
		rtyp, reply, err := r.s.Call(r.sessions[rank], typ, payload)
		r.pending[rank] <- frame{rtyp, reply, err}
	}()
}

func (r *rig) encode(m schedpolicy.Message) []byte {
	payload, err := m.MarshalBinary()
	if err != nil {
		r.t.Fatal(err)
	}
	return payload
}

func (r *rig) call(m schedpolicy.Message) frame {
	for rank := range r.sessions {
		r.send(rank, m)
	}
	return r.replies()
}

// replies collects every rank's reply and requires them identical.
func (r *rig) replies() frame {
	r.t.Helper()
	first := <-r.pending[0]
	for _, p := range r.pending[1:] {
		f := <-p
		if f.typ != first.typ || string(f.payload) != string(first.payload) || (f.err == nil) != (first.err == nil) {
			r.t.Fatal("ranks received different replies")
		}
	}
	return first
}

func (r *rig) hello() {
	r.t.Helper()
	for rank := range r.sessions {
		r.send(rank, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, Rank: uint32(rank), World: uint32(len(r.sessions))})
	}
	f := r.replies()
	require.NoError(r.t, f.err)
	require.Equal(r.t, schedpolicy.HelloOkType, f.typ)
}

// failure requires every rank's call in flight to fail with the group's
// error, and returns its text.
func (r *rig) failure() string {
	r.t.Helper()
	f := r.replies()
	require.Error(r.t, f.err)
	select {
	case <-r.s.Failed():
	default:
		r.t.Fatal("a call failed without failing the group")
	}
	require.Equal(r.t, r.s.Err().Error(), f.err.Error())
	return f.err.Error()
}

func boolReply(t *testing.T, f frame) bool {
	t.Helper()
	require.NoError(t, f.err)
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
	f := r.call(&schedpolicy.Init{BurstTokens: 4096, DeviceTokens: 1000, HostTokens: 0, Throttle: true})
	require.NoError(t, f.err)
	require.Equal(t, schedpolicy.AckType, f.typ)

	require.False(t, boolReply(t, r.call(&schedpolicy.ShouldDeferPrefill{PrefillPending: true, DecodeRunnable: true})))

	f = r.call(&schedpolicy.PrefillTokenBudget{})
	var budget schedpolicy.OptIntReply
	err := budget.UnmarshalBinary(f.payload)
	require.False(t, err != nil || budget.Present)

	r.send(0, &schedpolicy.BatchLaunched{BatchClass: 1, Tokens: 1000, Now: 0})
	r.send(1, &schedpolicy.BatchLaunched{BatchClass: 1, Tokens: 1000, Now: 50})
	r.replies()
	r.send(0, &schedpolicy.BatchFinished{Now: 1})
	r.send(1, &schedpolicy.BatchFinished{Now: 99})
	r.replies()
	require.True(t, boolReply(t, r.call(&schedpolicy.ShouldDeferPrefill{PrefillPending: true, DecodeRunnable: true, ContinuesChunk: true})))

	f = r.call(&schedpolicy.PrefillTokenBudget{})
	err = budget.UnmarshalBinary(f.payload)
	require.False(t, err != nil || !budget.Present || budget.Value != 4096-1000)

	r.call(&schedpolicy.RequestQueued{Rid: "a", Tokens: tokenBytes(0, 400), Now: 1})
	r.call(&schedpolicy.RequestFinished{Rid: "a", Length: 410, Tail: tokenBytes(346, 64), Now: 2})
	r.call(&schedpolicy.RequestQueued{Rid: "b", Tokens: tokenBytes(10000, 400), Now: 2})
	r.call(&schedpolicy.RequestFinished{Rid: "b", Length: 410, Tail: tokenBytes(10346, 64), Now: 3})
	r.call(&schedpolicy.RequestQueued{Rid: "a2", Tokens: tokenBytes(0, 420), Now: 3})
	r.call(&schedpolicy.RequestQueued{Rid: "c", Tokens: tokenBytes(50000, 300), Now: 3})
	r.call(&schedpolicy.BeginPass{Present: "a2\nc"})
	r.send(0, &schedpolicy.ShouldHold{Rid: "c", InputLen: 300, TotalTokens: 350, WouldEvict: true, QueuedAt: 3, Now: 3})
	r.send(1, &schedpolicy.ShouldHold{Rid: "c", InputLen: 300, TotalTokens: 350, WouldEvict: false, QueuedAt: 3, Now: 3})
	require.True(t, boolReply(t, r.replies()))

	f = r.call(&schedpolicy.Admitted{Evicted: true, Now: 3})
	require.NoError(t, f.err)
	require.Equal(t, schedpolicy.AckType, f.typ)
	select {
	case <-r.s.Failed():
		t.Fatal("the group failed")
	default:
	}
}

// Ranks connect in any order; the rank in the Hello decides, not the session.
func TestRanksMayConnectOutOfOrder(t *testing.T) {
	r := start(t, 2, time.Second)
	r.send(1, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, Rank: 1, World: 2})
	r.send(0, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, Rank: 0, World: 2})
	f := r.replies()
	require.NoError(t, f.err)
	require.Equal(t, schedpolicy.HelloOkType, f.typ)
	r.call(&schedpolicy.Init{BurstTokens: 100})
	r.send(1, &schedpolicy.BatchLaunched{BatchClass: 1, Tokens: 40, Now: 7})
	r.send(0, &schedpolicy.BatchLaunched{BatchClass: 1, Tokens: 40, Now: 0})
	f = r.replies()
	require.NoError(t, f.err)
	require.Equal(t, schedpolicy.AckType, f.typ)
}

func TestInitWithoutThrottleRejectsShouldHold(t *testing.T) {
	r := start(t, 1, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{BurstTokens: -1})
	r.call(&schedpolicy.RequestQueued{Rid: "a"})
	r.call(&schedpolicy.BeginPass{})
	r.send(0, &schedpolicy.ShouldHold{Rid: "a"})
	msg := r.failure()
	require.Contains(t, msg, "without an eviction throttle")
}

func TestProtocolMismatchIsRefused(t *testing.T) {
	r := start(t, 1, time.Second)
	r.send(0, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion + 1, World: 1})
	msg := r.failure()
	require.Contains(t, msg, "speaks protocol")
}

func TestWrongRankOrWorldIsRefused(t *testing.T) {
	r := start(t, 1, time.Second)
	r.send(0, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, Rank: 1, World: 2})
	msg := r.failure()
	require.Contains(t, msg, "connected as rank 1 of 2")

	r = start(t, 2, time.Second)
	r.send(0, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, Rank: 1, World: 2})
	r.send(1, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, Rank: 1, World: 2})
	msg = r.failure()
	require.Contains(t, msg, "rank 1 connected twice")
}

func TestRequestBeforeHelloIsRefused(t *testing.T) {
	r := start(t, 1, time.Second)
	r.send(0, &schedpolicy.Init{})
	msg := r.failure()
	require.Contains(t, msg, "before hello")
}

func TestRequestBeforeInitAndDoubleInitAreRefused(t *testing.T) {
	r := start(t, 1, time.Second)
	r.hello()
	r.send(0, &schedpolicy.BeginPass{})
	msg := r.failure()
	require.Contains(t, msg, "before init")

	r = start(t, 1, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{})
	r.send(0, &schedpolicy.Init{})
	msg = r.failure()
	require.Contains(t, msg, "init sent twice")
}

func TestMalformedAndUnknownMessagesAreRefused(t *testing.T) {
	r := start(t, 1, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{})
	r.sendRaw(0, schedpolicy.BatchLaunchedType, []byte{1})
	msg := r.failure()
	require.Contains(t, msg, "shorter than the fixed section")

	r = start(t, 1, time.Second)
	r.hello()
	r.sendRaw(0, schedpolicy.AckType, nil)
	msg = r.failure()
	require.Contains(t, msg, "not a request")

	r = start(t, 1, time.Second)
	r.hello()
	r.sendRaw(0, 77, nil)
	msg = r.failure()
	require.Contains(t, msg, "type 77 is not a request")

	r = start(t, 1, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{})
	r.send(0, &schedpolicy.BatchLaunched{BatchClass: 7})
	msg = r.failure()
	require.Contains(t, msg, "batch class 7")

	r = start(t, 1, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{})
	r.send(0, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, World: 1})
	msg = r.failure()
	require.Contains(t, msg, "hello twice")
}

func TestRanksThatDivergeAreToldSoInsteadOfHanging(t *testing.T) {
	r := start(t, 2, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{Throttle: true})
	r.send(0, &schedpolicy.BeginPass{Present: "a"})
	r.send(1, &schedpolicy.ShouldHold{Rid: "a"})
	msg := r.failure()
	require.Contains(t, msg, "ranks diverged")

	// Same op, different replicated argument.
	r = start(t, 2, time.Second)
	r.hello()
	r.call(&schedpolicy.Init{Throttle: true})
	r.send(0, &schedpolicy.BeginPass{Present: "a"})
	r.send(1, &schedpolicy.BeginPass{Present: "b"})
	msg = r.failure()
	require.Contains(t, msg, "ranks diverged")
}

func TestARankThatNeverSendsTripsTheLockstepTimeout(t *testing.T) {
	r := start(t, 2, 50*time.Millisecond)
	r.send(0, &schedpolicy.Hello{Protocol: schedpolicy.ProtocolVersion, World: 2})
	f := <-r.pending[0]
	require.ErrorIs(t, f.err, errTimeout)
	require.Contains(t, f.err.Error(), "lockstep timeout: rank(s) [1] did not send")
}

// A rank that leaves fails the group, and a call after the failure gets the
// same answer. Gone closes once the last rank has left.
func TestARankThatExitsFailsTheGroup(t *testing.T) {
	r := start(t, 2, time.Second)
	r.hello()
	r.send(0, &schedpolicy.Init{})
	r.s.Gone(r.sessions[1])
	f := <-r.pending[0]
	require.ErrorContains(t, f.err, "rank 1 exited")
	require.ErrorContains(t, r.s.Err(), "rank 1 exited")
	r.send(0, &schedpolicy.BeginPass{})
	require.ErrorContains(t, (<-r.pending[0]).err, "rank 1 exited")
	select {
	case <-r.s.Left():
		t.Fatal("gone closed while rank 0 is connected")
	default:
	}
	r.s.Gone(r.sessions[0])
	<-r.s.Left()
}

// A session that leaves before its hello fails the group too, because a
// scheduler that connected and died is a scheduler that died.
func TestASessionThatExitsBeforeHelloFailsTheGroup(t *testing.T) {
	r := start(t, 1, time.Second)
	r.s.Gone(&ipc.Session{})
	require.ErrorContains(t, r.s.Err(), "before hello")
	<-r.s.Left()
}
