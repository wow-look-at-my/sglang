package server

import (
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/policy"
	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/schedpolicy"
)

type request interface {
	encoder
	UnmarshalBinary([]byte) error
}

// decode returns the request struct for a message type, or nil for a type
// that is not a request.
func decode(typ uint32, payload []byte) (request, error) {
	var m request
	switch typ {
	case schedpolicy.HelloType:
		m = &schedpolicy.Hello{}
	case schedpolicy.InitType:
		m = &schedpolicy.Init{}
	case schedpolicy.ShouldDeferPrefillType:
		m = &schedpolicy.ShouldDeferPrefill{}
	case schedpolicy.PrefillTokenBudgetType:
		m = &schedpolicy.PrefillTokenBudget{}
	case schedpolicy.BatchLaunchedType:
		m = &schedpolicy.BatchLaunched{}
	case schedpolicy.BatchFinishedType:
		m = &schedpolicy.BatchFinished{}
	case schedpolicy.RequestQueuedType:
		m = &schedpolicy.RequestQueued{}
	case schedpolicy.RequestFinishedType:
		m = &schedpolicy.RequestFinished{}
	case schedpolicy.BeginPassType:
		m = &schedpolicy.BeginPass{}
	case schedpolicy.ShouldHoldType:
		m = &schedpolicy.ShouldHold{}
	case schedpolicy.AdmittedType:
		m = &schedpolicy.Admitted{}
	default:
		return nil, fmt.Errorf("message type %d is not a request", typ)
	}
	if err := m.UnmarshalBinary(payload); err != nil {
		return nil, fmt.Errorf("message type %d: %w", typ, err)
	}
	return m, nil
}

// replicated re-encodes a request with its rank-local fields zeroed: the
// bytes every rank must agree on. Clocks and allocator state are rank-local.
func replicated(typ uint32, payload []byte) ([]byte, error) {
	m, err := decode(typ, payload)
	if err != nil {
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

func (s *Server) dispatch(typ uint32, payload []byte) (uint32, encoder, error) {
	m, err := decode(typ, payload)
	if err != nil {
		return 0, nil, err
	}
	if init, ok := m.(*schedpolicy.Init); ok {
		return s.init(init)
	}
	if s.balancer == nil {
		return 0, nil, fmt.Errorf("message type %d before init", typ)
	}
	switch r := m.(type) {
	case *schedpolicy.ShouldDeferPrefill:
		defer_ := s.balancer.ShouldDeferPrefill(r.PrefillPending, r.DecodeRunnable, r.ContinuesChunk)
		return schedpolicy.BoolReplyType, &schedpolicy.BoolReply{Value: defer_}, nil
	case *schedpolicy.PrefillTokenBudget:
		budget := s.balancer.PrefillTokenBudget(r.ContinuesChunk)
		if budget == nil {
			return schedpolicy.OptIntReplyType, &schedpolicy.OptIntReply{}, nil
		}
		return schedpolicy.OptIntReplyType, &schedpolicy.OptIntReply{Present: true, Value: int64(*budget)}, nil
	case *schedpolicy.BatchLaunched:
		if r.BatchClass > uint8(policy.ClassDecode) {
			return 0, nil, fmt.Errorf("batch class %d is not other, prefill or decode", r.BatchClass)
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
			return 0, nil, fmt.Errorf("should_hold without an eviction throttle")
		}
		hold := s.throttle.ShouldHold(r.Rid, int(r.InputLen), int(r.DeviceHit), int(r.TotalTokens),
			r.WouldEvict, r.QueuedAt, r.Now)
		return schedpolicy.BoolReplyType, &schedpolicy.BoolReply{Value: hold}, nil
	case *schedpolicy.Admitted:
		if s.throttle != nil {
			s.throttle.OnAdmitted(r.Evicted, r.Now)
		}
	default:
		return 0, nil, fmt.Errorf("message type %d after hello", typ)
	}
	return schedpolicy.AckType, &schedpolicy.Ack{}, nil
}

func (s *Server) init(r *schedpolicy.Init) (uint32, encoder, error) {
	if s.balancer != nil {
		return 0, nil, fmt.Errorf("init sent twice")
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
	return schedpolicy.AckType, &schedpolicy.Ack{}, nil
}

// tokens decodes little-endian int32 token ids.
func tokens(b []byte) []int32 {
	out := make([]int32, len(b)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}
