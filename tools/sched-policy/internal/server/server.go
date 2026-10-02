// Package server runs the policy for one TP group as a go-ipc service. Every
// rank makes each call; a call is answered once every rank has made it, with
// the same reply for all. A rank that diverges, that exits, or that does not
// arrive within the lockstep timeout fails the group: every rank hears the
// reason as a call error instead of hanging.
package server

import (
	"errors"
	"fmt"
	"sync"
	"time"

	ipc "github.com/wow-look-at-my/go-ipc"
	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/policy"
	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/schedpolicy"
)

type Server struct {
	world   int
	timeout time.Duration

	mu    sync.Mutex
	ranks map[*ipc.Session]int
	// taken marks the ranks that have said hello.
	taken []bool
	round *round
	// failure is the error that ended the group. Every call after it returns it.
	failure error
	failed  chan struct{}
	// gone closes once the group has failed and every rank has left.
	gone     chan struct{}
	live     int
	balancer *policy.Balancer
	throttle *policy.Throttle
}

func New(world int, lockstepTimeout time.Duration) *Server {
	return &Server{
		world: world, timeout: lockstepTimeout,
		ranks: make(map[*ipc.Session]int), taken: make([]bool, world),
		failed: make(chan struct{}), gone: make(chan struct{}),
	}
}

// A round is one call in lockstep: the request of every rank, and the reply
// they all get.
type round struct {
	requests []schedpolicy.Message
	got      []bool
	n        int
	timer    *time.Timer
	done     chan struct{}
	reply    schedpolicy.Message
	err      error
}

// Failed closes once the group has failed. Err returns the reason.
func (s *Server) Failed() <-chan struct{} { return s.failed }

// Left closes once the group has failed and every rank has left.
func (s *Server) Left() <-chan struct{} { return s.gone }

func (s *Server) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.failure
}

// Call answers one rank's call, after every rank has made the same call.
func (s *Server) Call(sess *ipc.Session, typ uint32, payload []byte) (uint32, []byte, error) {
	r, err := s.join(sess, typ, payload)
	if err != nil {
		return 0, nil, err
	}
	<-r.done
	if r.err != nil {
		return 0, nil, r.err
	}
	b, err := r.reply.MarshalBinary()
	if err != nil {
		return 0, nil, err
	}
	return r.reply.TypeID(), b, nil
}

// Gone fails the group when a rank leaves, and marks the last departure.
func (s *Server) Gone(sess *ipc.Session) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rank, seen := s.ranks[sess]
	if seen {
		delete(s.ranks, sess)
		s.live--
	}
	if s.failure == nil {
		if rank >= 0 && seen {
			s.fail(fmt.Errorf("rank %d exited", rank))
		} else {
			s.fail(errors.New("a rank exited before hello"))
		}
	}
	s.markGone()
}

// first records a session on its first call, so Gone knows it was here.
func (s *Server) first(sess *ipc.Session) {
	if _, seen := s.ranks[sess]; !seen {
		s.ranks[sess] = -1
		s.live++
	}
}

// markGone closes gone once the group has failed and no rank remains.
func (s *Server) markGone() {
	if s.failure == nil || s.live > 0 {
		return
	}
	select {
	case <-s.gone:
	default:
		close(s.gone)
	}
}

// join adds a request to the current round, and completes the round when
// the request is the last one.
func (s *Server) join(sess *ipc.Session, typ uint32, payload []byte) (*round, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.first(sess)
	if s.failure != nil {
		return nil, s.failure
	}
	m := schedpolicy.NewMessage(typ)
	if m == nil {
		return nil, s.fail(fmt.Errorf("message type %d is not a request", typ))
	}
	if err := m.UnmarshalBinary(payload); err != nil {
		return nil, s.fail(fmt.Errorf("message type %d: %w", typ, err))
	}
	rank, err := s.rankOf(sess, m)
	if err != nil {
		return nil, s.fail(err)
	}
	r := s.round
	if r == nil {
		r = &round{requests: make([]schedpolicy.Message, s.world), got: make([]bool, s.world), done: make(chan struct{})}
		r.timer = time.AfterFunc(s.timeout, func() { s.expire(r) })
		s.round = r
	}
	if r.got[rank] {
		return nil, s.fail(fmt.Errorf("rank %d sent message type %d twice in one call", rank, typ))
	}
	r.requests[rank], r.got[rank] = m, true
	r.n++
	if r.n == s.world {
		r.timer.Stop()
		s.round = nil
		s.complete(r)
	}
	return r, nil
}

// rankOf returns the rank of a session. A Hello names it; every later call
// finds it recorded.
func (s *Server) rankOf(sess *ipc.Session, m schedpolicy.Message) (int, error) {
	hello, isHello := m.(*schedpolicy.Hello)
	rank := s.ranks[sess]
	known := rank >= 0
	switch {
	case known && isHello:
		return 0, fmt.Errorf("rank %d sent hello twice", rank)
	case known:
		return rank, nil
	case !isHello:
		return 0, fmt.Errorf("a rank sent message type %d before hello", m.TypeID())
	}
	if hello.Protocol != schedpolicy.ProtocolVersion {
		return 0, fmt.Errorf("rank %d speaks protocol %d, policy process speaks %d",
			hello.Rank, hello.Protocol, schedpolicy.ProtocolVersion)
	}
	if int(hello.World) != s.world || int(hello.Rank) >= s.world {
		return 0, fmt.Errorf("a rank connected as rank %d of %d to a policy process for %d ranks",
			hello.Rank, hello.World, s.world)
	}
	rank = int(hello.Rank)
	if s.taken[rank] {
		return 0, fmt.Errorf("rank %d connected twice", rank)
	}
	s.taken[rank] = true
	s.ranks[sess] = rank
	return rank, nil
}

func (s *Server) complete(r *round) {
	defer close(r.done)
	first, err := replicated(r.requests[0])
	if err != nil {
		r.err = s.fail(fmt.Errorf("rank 0: %w", err))
		return
	}
	for rank := 1; rank < s.world; rank++ {
		other, err := replicated(r.requests[rank])
		if err != nil {
			r.err = s.fail(fmt.Errorf("rank %d: %w", rank, err))
			return
		}
		if r.requests[rank].TypeID() != r.requests[0].TypeID() || string(other) != string(first) {
			r.err = s.fail(fmt.Errorf("ranks diverged: rank 0 sent message type %d %x, rank %d sent type %d %x",
				r.requests[0].TypeID(), first, rank, r.requests[rank].TypeID(), other))
			return
		}
	}
	r.reply, err = s.dispatch(r.requests[0])
	if err != nil {
		r.err = s.fail(err)
	}
}

var errTimeout = errors.New("lockstep timeout")

// expire fails the group when a round is still open at the lockstep timeout.
func (s *Server) expire(r *round) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.round != r {
		return
	}
	s.fail(fmt.Errorf("%w: rank(s) %v did not send the request rank(s) %v sent",
		errTimeout, ranksWhere(r.got, false), ranksWhere(r.got, true)))
}

// fail records the first failure and releases the open round with it. It
// returns the error that stands.
func (s *Server) fail(err error) error {
	if s.failure != nil {
		return s.failure
	}
	s.failure = err
	close(s.failed)
	if r := s.round; r != nil {
		s.round = nil
		r.timer.Stop()
		r.err = err
		close(r.done)
	}
	s.markGone()
	return err
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
