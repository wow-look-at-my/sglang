// Command sglang-sched-policy decides prefill/decode time sharing and
// eviction throttling for one TP group of the SGLang scheduler. The ranks
// connect over go-ipc channels named <name>.r<rank>; the process prints
// "ready" once every channel exists and exits non-zero when the ranks diverge.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	ipc "github.com/wow-look-at-my/go-ipc"
	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/server"
)

func main() {
	name := flag.String("name", "", "channel name prefix shared with the ranks")
	ranks := flag.Int("ranks", 0, "number of scheduler ranks")
	timeout := flag.Duration("lockstep-timeout", 5*time.Minute,
		"how long the other ranks may lag the first rank's request")
	flag.Parse()
	if *name == "" || *ranks <= 0 {
		fmt.Fprintln(os.Stderr, "sglang-sched-policy: -name and -ranks are required")
		os.Exit(2)
	}
	if err := run(*name, *ranks, *timeout); err != nil {
		fmt.Fprintln(os.Stderr, "sglang-sched-policy:", err)
		os.Exit(1)
	}
}

func run(name string, ranks int, timeout time.Duration) error {
	endpoints := make([]server.Endpoint, ranks)
	for rank := range ranks {
		ch, err := ipc.CreateChannel(fmt.Sprintf("%s.r%d", name, rank))
		if err != nil {
			return err
		}
		defer ch.Unlink()
		defer ch.Close()
		endpoints[rank] = ch
	}
	fmt.Println("ready")
	return server.New(endpoints, timeout).Serve(context.Background())
}
