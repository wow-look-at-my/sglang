package server

import (
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/policy"
	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/schedpolicy"
)

// replicated encodes a copy of a request with its rank-local fields zeroed:
// the bytes every rank must agree on. Clocks and allocator state are
// rank-local.
func replicated(req schedpolicy.Message) ([]byte, error) {
	b, err := req.MarshalBinary()
	if err != nil {
		return nil, err
	}
	m := schedpolicy.NewMessage(req.TypeID())
	if err := m.UnmarshalBinary(b); err != nil {
		return nil, err
	}
	switch r := m.(type) {
	case *schedpolicy.Hello:
		r.Rank = 0
	case *schedpolicy.BatchLaunched:
		r.Now = 0
	case *schedpolicy.BatchFinished:
		r.Now = 0
	case *schedpolicy.RequestQueued:
		r.Now = 0
	case *schedpolicy.RequestFinished:
		r.Now = 0
	case *schedpolicy.ShouldHold:
		r.Now, r.QueuedAt = 0, 0
		r.WouldEvict, r.DeviceHit, r.TotalTokens = false, 0, 0
	case *schedpolicy.Admitted:
		r.Now = 0
	}
	return m.MarshalBinary()
}

// dispatch answers the request every rank agreed on. A Hello answers with
// HelloOk; an Init builds the policy; the rest need the policy built.
func (s *Server) dispatch(m schedpolicy.Message) (schedpolicy.Message, error) {
	switch r := m.(type) {
	case *schedpolicy.Hello:
		return &schedpolicy.HelloOk{Protocol: schedpolicy.ProtocolVersion}, nil
	case *schedpolicy.Init:
		return s.init(r)
	case *schedpolicy.ShouldDeferPrefill, *schedpolicy.PrefillTokenBudget, *schedpolicy.BatchLaunched,
		*schedpolicy.BatchFinished, *schedpolicy.RequestQueued, *schedpolicy.RequestFinished,
		*schedpolicy.BeginPass, *schedpolicy.ShouldHold, *schedpolicy.Admitted:
	default:
		return nil, fmt.Errorf("message type %d is not a request", m.TypeID())
	}
	if s.balancer == nil {
		return nil, fmt.Errorf("message type %d before init", m.TypeID())
	}
	switch r := m.(type) {
	case *schedpolicy.ShouldDeferPrefill:
		defer_ := s.balancer.ShouldDeferPrefill(r.PrefillPending, r.DecodeRunnable, r.ContinuesChunk)
		return &schedpolicy.BoolReply{Value: defer_}, nil
	case *schedpolicy.PrefillTokenBudget:
		budget := s.balancer.PrefillTokenBudget(r.ContinuesChunk)
		if budget == nil {
			return &schedpolicy.OptIntReply{}, nil
		}
		return &schedpolicy.OptIntReply{Present: true, Value: int64(*budget)}, nil
	case *schedpolicy.BatchLaunched:
		if r.BatchClass > uint8(policy.ClassDecode) {
			return nil, fmt.Errorf("batch class %d is not other, prefill or decode", r.BatchClass)
		}
		s.balancer.OnBatchLaunched(policy.BatchClass(r.BatchClass), int(r.Tokens), int(r.DecodeRows), r.Now)
	case *schedpolicy.BatchFinished:
		s.balancer.OnBatchFinished(r.Now)
	case *schedpolicy.RequestQueued:
		if s.throttle != nil {
			s.throttle.OnRequestQueued(r.Rid, tokens(r.Tokens), r.Now)
		}
	case *schedpolicy.RequestFinished:
		if s.throttle != nil {
			s.throttle.OnRequestFinished(r.Rid, int(r.Length), tokens(r.Tail), r.Now)
		}
	case *schedpolicy.BeginPass:
		if s.throttle != nil {
			s.throttle.BeginPass(strings.Split(r.Present, "\n"))
		}
	case *schedpolicy.ShouldHold:
		if s.throttle == nil {
			return nil, fmt.Errorf("should_hold without an eviction throttle")
		}
		hold := s.throttle.ShouldHold(r.Rid, int(r.InputLen), int(r.DeviceHit), int(r.TotalTokens),
			r.WouldEvict, r.QueuedAt, r.Now)
		return &schedpolicy.BoolReply{Value: hold}, nil
	case *schedpolicy.Admitted:
		if s.throttle != nil {
			s.throttle.OnAdmitted(r.Evicted, r.Now)
		}
	}
	return &schedpolicy.Ack{}, nil
}

func (s *Server) init(r *schedpolicy.Init) (schedpolicy.Message, error) {
	if s.balancer != nil {
		return nil, fmt.Errorf("init sent twice")
	}
	var burst *int
	if r.BurstTokens >= 0 {
		b := int(r.BurstTokens)
		burst = &b
	}
	s.balancer = policy.NewBalancer(burst)
	if r.Throttle {
		s.throttle = policy.NewThrottle(int(r.DeviceTokens), int(r.HostTokens), s.balancer.PrefillSecondsPerToken)
	}
	return &schedpolicy.Ack{}, nil
}

// tokens decodes little-endian int32 token ids.
func tokens(b []byte) []int32 {
	out := make([]int32, len(b)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}
