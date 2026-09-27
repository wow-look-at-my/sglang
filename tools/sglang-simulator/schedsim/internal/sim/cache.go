package sim

// Segment is one radix-tree node: a run of tokens a conversation appended.
type Segment struct {
	Tokens int
	Parent *Segment

	inTree     bool
	onDevice   bool
	onHost     bool
	lock       int
	lastAccess float64
	kids       int
	kidsDevice int
	index      int
}

type Cache struct {
	DeviceCap, HostCap int

	deviceUsed   int
	hostUsed     int
	lockedDevice int
	// reqTokens are device tokens requests hold outside the tree.
	reqTokens int
	segs      []*Segment
}

// NewCache returns an empty cache.
func NewCache(deviceCap, hostCap int) *Cache {
	return &Cache{DeviceCap: deviceCap, HostCap: hostCap}
}

// Free is the device tokens nobody holds.
func (c *Cache) Free() int { return c.DeviceCap - c.deviceUsed - c.reqTokens }

// Evictable is the cached device tokens no running request locks.
func (c *Cache) Evictable() int { return c.deviceUsed - c.lockedDevice }

// Match walks a chain: the device-resident prefix, then the host-only run.
func (c *Cache) Match(chain []*Segment) (dev, host []*Segment) {
	for _, s := range chain {
		if !s.inTree {
			break
		}
		if s.onDevice && len(host) == 0 {
			dev = append(dev, s)
			continue
		}
		if !s.onHost {
			break
		}
		host = append(host, s)
	}
	return dev, host
}

func tokensOf(segs []*Segment) int {
	n := 0
	for _, s := range segs {
		n += s.Tokens
	}
	return n
}

// Lock pins segments on device for a running request.
func (c *Cache) Lock(segs []*Segment, now float64) {
	for _, s := range segs {
		if s.lock == 0 && s.onDevice {
			c.lockedDevice += s.Tokens
		}
		s.lock++
		s.lastAccess = now
	}
}

// Unlock releases what Lock pinned.
func (c *Cache) Unlock(segs []*Segment, now float64) {
	for _, s := range segs {
		s.lock--
		if s.lock == 0 && s.onDevice {
			c.lockedDevice -= s.Tokens
		}
		s.lastAccess = now
	}
}

// Alloc takes n device tokens for a request, evicting if it must.
func (c *Cache) Alloc(n int) bool {
	if c.Free() < n && !c.evictDevice(n) {
		return false
	}
	c.reqTokens += n
	return true
}

// Release returns request-held device tokens.
func (c *Cache) Release(n int) { c.reqTokens -= n }

// LoadBack copies host-only segments to device. The caller locks their
// device-resident ancestors first so eviction cannot take them.
func (c *Cache) LoadBack(segs []*Segment) bool {
	need := tokensOf(segs)
	if c.Free() < need && !c.evictDevice(need) {
		return false
	}
	for _, s := range segs {
		s.onDevice = true
		c.deviceUsed += s.Tokens
		if s.lock > 0 {
			c.lockedDevice += s.Tokens
		}
		if s.Parent != nil {
			s.Parent.kidsDevice++
		}
	}
	return true
}

// Insert puts a finished request's segments on device, taking the tokens the
// request held. A segment another request already restored costs nothing.
func (c *Cache) Insert(segs []*Segment, held int, now float64) {
	for _, s := range segs {
		s.lastAccess = now
		if s.inTree && s.onDevice {
			continue
		}
		held -= s.Tokens
		c.reqTokens -= s.Tokens
		if !s.inTree {
			s.inTree = true
			s.index = len(c.segs)
			c.segs = append(c.segs, s)
			if s.Parent != nil {
				s.Parent.kids++
			}
		}
		s.onDevice = true
		c.deviceUsed += s.Tokens
		if s.Parent != nil {
			s.Parent.kidsDevice++
		}
		c.backup(s)
	}
	c.reqTokens -= held
}

// Seed places a segment directly, for state that exists before a run.
func (c *Cache) Seed(s *Segment, onDevice bool, now float64) {
	s.inTree, s.lastAccess = true, now
	s.index = len(c.segs)
	c.segs = append(c.segs, s)
	if s.Parent != nil {
		s.Parent.kids++
	}
	if onDevice && c.Free() >= s.Tokens {
		s.onDevice = true
		c.deviceUsed += s.Tokens
		if s.Parent != nil {
			s.Parent.kidsDevice++
		}
	}
	c.backup(s)
	if !s.onDevice && !s.onHost {
		c.remove(s)
	}
}

func (c *Cache) backup(s *Segment) {
	if c.HostCap == 0 || s.onHost {
		return
	}
	for c.hostUsed+s.Tokens > c.HostCap {
		if !c.evictHostOne() {
			return
		}
	}
	s.onHost = true
	c.hostUsed += s.Tokens
}

func (c *Cache) evictDevice(need int) bool {
	for c.Free() < need {
		var victim *Segment
		for _, s := range c.segs {
			if !s.onDevice || s.lock > 0 || s.kidsDevice > 0 || (!s.onHost && s.kids > 0) {
				continue
			}
			if victim == nil || s.lastAccess < victim.lastAccess {
				victim = s
			}
		}
		if victim == nil {
			return false
		}
		victim.onDevice = false
		c.deviceUsed -= victim.Tokens
		if victim.Parent != nil {
			victim.Parent.kidsDevice--
		}
		if !victim.onHost {
			c.remove(victim)
		}
	}
	return true
}

func (c *Cache) evictHostOne() bool {
	var victim *Segment
	for _, s := range c.segs {
		if !s.onHost || s.onDevice || s.kids > 0 {
			continue
		}
		if victim == nil || s.lastAccess < victim.lastAccess {
			victim = s
		}
	}
	if victim == nil {
		return false
	}
	c.remove(victim)
	return true
}

func (c *Cache) remove(s *Segment) {
	if s.onHost {
		c.hostUsed -= s.Tokens
		s.onHost = false
	}
	s.inTree = false
	if s.Parent != nil {
		s.Parent.kids--
	}
	last := c.segs[len(c.segs)-1]
	c.segs[s.index] = last
	last.index = s.index
	c.segs = c.segs[:len(c.segs)-1]
}
