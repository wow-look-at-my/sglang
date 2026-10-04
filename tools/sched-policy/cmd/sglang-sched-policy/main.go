// Command sglang-sched-policy decides prefill/decode time sharing and
// eviction throttling for one TP group of the SGLang scheduler. The ranks
// connect to the go-ipc service of the given name; the process exits
// non-zero when the ranks diverge, once every rank has heard why.
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	ipc "github.com/wow-look-at-my/go-ipc"
	"github.com/wow-look-at-my/sglang/tools/sched-policy/internal/server"
)

func main() {
	name := flag.String("name", "", "service name shared with the ranks")
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

// run serves until the group fails. It then waits for the ranks to leave, so
// each has read the failure, and no longer than the lockstep timeout.
func run(name string, ranks int, timeout time.Duration) error {
	srv := server.New(ranks, timeout)
	svc, err := ipc.Serve(name, srv)
	if err != nil {
		return err
	}
	defer svc.Close()
	<-srv.Failed()
	select {
	case <-srv.Left():
	case <-time.After(timeout):
	}
	return srv.Err()
}
