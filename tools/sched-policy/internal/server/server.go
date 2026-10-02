// Package server runs the policy for one TP group. A rank that diverges, or
// that does not arrive within the lockstep timeout, is an error reported to
// every rank instead of a hang.
package server

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"time"

	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/policy"
	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/schedpolicy"
)

// Endpoint is one rank's connection. *ipc.Channel satisfies it.
type Endpoint interface {
	Recv(ctx context.Context) (uint32, []byte, error)
	SendTyped(ctx context.Context, typ uint32, payload []byte) error
}

type Server struct {
	ranks    []Endpoint
	timeout  time.Duration
	balancer *policy.Balancer
	throttle *policy.Throttle
}

func New(ranks []Endpoint, lockstepTimeout time.Duration) *Server {
	return &Server{ranks: ranks, timeout: lockstepTimeout}
}

type arrival struct {
	typ     uint32
	payload []byte
	err     error
}

// Serve runs until a rank's endpoint fails or the ranks diverge. It returns
// the error that ended it; every rank has been sent that error first.
func (s *Server) Serve(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	inbox := make([]chan arrival, len(s.ranks))
	for i, ep := range s.ranks {
		inbox[i] = make(chan arrival, 1)
		go read(ctx, ep, inbox[i])
	}
	if err := s.hello(ctx, inbox); err != nil {
		return s.fail(ctx, err)
	}
	for {
		round, err := s.collect(ctx, inbox)
		if err != nil {
			return s.fail(ctx, err)
		}
		typ, reply, err := s.dispatch(round[0].typ, round[0].payload)
		if err != nil {
			return s.fail(ctx, err)
		}
		if err := s.broadcast(ctx, typ, reply); err != nil {
			return err
		}
	}
}

func read(ctx context.Context, ep Endpoint, out chan<- arrival) {
	for {
		typ, payload, err := ep.Recv(ctx)
		select {
		case out <- arrival{typ: typ, payload: payload, err: err}:
		case <-ctx.Done():
			return
		}
		if err != nil {
			return
		}
	}
}

func (s *Server) hello(ctx context.Context, inbox []chan arrival) error {
	round, err := s.collect(ctx, inbox)
	if err != nil {
		return err
	}
	for rank, a := range round {
		if a.typ != schedpolicy.HelloType {
			return fmt.Errorf("rank %d sent message type %d before hello", rank, a.typ)
		}
		var hello schedpolicy.Hello
		if err := hello.UnmarshalBinary(a.payload); err != nil {
			return fmt.Errorf("rank %d hello: %w", rank, err)
		}
		if hello.Protocol != schedpolicy.ProtocolVersion {
			return fmt.Errorf("rank %d speaks protocol %d, policy process speaks %d",
				rank, hello.Protocol, schedpolicy.ProtocolVersion)
		}
		if int(hello.Rank) != rank || int(hello.World) != len(s.ranks) {
			return fmt.Errorf("rank %d of %d connected as rank %d of %d",
				rank, len(s.ranks), hello.Rank, hello.World)
		}
	}
	return s.broadcast(ctx, schedpolicy.HelloOkType, &schedpolicy.HelloOk{Protocol: schedpolicy.ProtocolVersion})
}

// collect takes one request from every rank. It waits indefinitely for the
// first; the rest must follow within the lockstep timeout.
func (s *Server) collect(ctx context.Context, inbox []chan arrival) ([]arrival, error) {
	round := make([]arrival, len(inbox))
	got := make([]bool, len(inbox))
	var deadline <-chan time.Time
	for received := 0; received < len(inbox); received++ {
		rank, a, err := next(ctx, inbox, got, deadline)
		if err != nil {
			return nil, err
		}
		if a.err != nil {
			return nil, fmt.Errorf("rank %d: %w", rank, a.err)
		}
		round[rank], got[rank] = a, true
		if deadline == nil {
			deadline = time.After(s.timeout)
		}
	}
	first, err := replicated(round[0].typ, round[0].payload)
	if err != nil {
		return nil, fmt.Errorf("rank 0: %w", err)
	}
	for rank := 1; rank < len(round); rank++ {
		other, err := replicated(round[rank].typ, round[rank].payload)
		if err != nil {
			return nil, fmt.Errorf("rank %d: %w", rank, err)
		}
		if round[rank].typ != round[0].typ || string(other) != string(first) {
			return nil, fmt.Errorf("ranks diverged: rank 0 sent message type %d %x, rank %d sent type %d %x",
				round[0].typ, first, rank, round[rank].typ, other)
		}
	}
	return round, nil
}

var errTimeout = errors.New("lockstep timeout")

// next waits for a message from any rank not yet heard from this round.
func next(ctx context.Context, inbox []chan arrival, got []bool, deadline <-chan time.Time) (int, arrival, error) {
	cases := []reflect.SelectCase{
		{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(ctx.Done())},
		{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(deadline)},
	}
	ranks := []int{-1, -1}
	for rank, ch := range inbox {
		if !got[rank] {
			cases = append(cases, reflect.SelectCase{Dir: reflect.SelectRecv, Chan: reflect.ValueOf(ch)})
			ranks = append(ranks, rank)
		}
	}
	chosen, value, _ := reflect.Select(cases)
	switch chosen {
	case 0:
		return 0, arrival{}, ctx.Err()
	case 1:
		return 0, arrival{}, fmt.Errorf("%w: rank(s) %v did not send the request rank(s) %v sent",
			errTimeout, ranksWhere(got, false), ranksWhere(got, true))
	}
	return ranks[chosen], value.Interface().(arrival), nil
}

func ranksWhere(got []bool, want bool) []int {
	var ranks []int
	for rank, g := range got {
		if g == want {
			ranks = append(ranks, rank)
		}
	}
	return ranks
}

type encoder interface {
	MarshalBinary() ([]byte, error)
}

func (s *Server) broadcast(ctx context.Context, typ uint32, reply encoder) error {
	payload, err := reply.MarshalBinary()
	if err != nil {
		return err
	}
	for rank, ep := range s.ranks {
		if err := ep.SendTyped(ctx, typ, payload); err != nil {
			return fmt.Errorf("reply to rank %d: %w", rank, err)
		}
	}
	return nil
}

func (s *Server) fail(ctx context.Context, cause error) error {
	if err := s.broadcast(ctx, schedpolicy.ErrorType, &schedpolicy.Error{Message: cause.Error()}); err != nil {
		return errors.Join(cause, err)
	}
	return cause
}
