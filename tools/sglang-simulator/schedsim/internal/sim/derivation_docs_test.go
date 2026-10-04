package sim

import (
	"os"
	"strings"
	"testing"
)

// Each losing cell is excused by exactly one derivation document, and
// boundName is the only table of which cell gets which. A bound whose document is
// missing, or a document no bound reads, means the excuse in the test and the
// argument on disk have drifted apart -- the test would keep passing while the
// reasoning it cites no longer exists. This checks the pair, in both directions.
var derivationDocs = map[string]string{
	boundWholeGPU:       "../../docs/derivation-cold-ttft-vs-old.md",
	boundOwnWork:        "../../docs/derivation-cold-ttft-vs-prev.md",
	boundMixedChunkTail: "../../docs/derivation-itl-percentiles-under-mixed-chunk.md",
	boundEvictionTail:   "../../docs/derivation-itl-percentiles-under-eviction.md",
	boundStreamRate:     "../../docs/derivation-stream-rate-under-mixed-chunk.md",
	boundAdmissionRate:  "../../docs/derivation-scenario-a-throughput.md",
}

func TestEveryBoundNamesACommittedDerivation(t *testing.T) {
	for name, path := range derivationDocs {
		body, err := os.ReadFile(path)
		if err != nil {
			t.Errorf("bound %q cites %s: %v", name, path, err)
			continue
		}
		if !strings.Contains(strings.ToLower(string(body)), strings.ToLower(name)) {
			t.Errorf("%s does not name the bound it is cited for, %q", path, name)
		}
	}
}

// boundName is consulted for every scenario the suite measures, so the set it can
// return is fixed by the table rather than by which cells happen to lose on this
// run. A bound name with no document would excuse a loss with nothing behind it,
// and a document no name returns is an argument nothing reads.
func TestBoundTableAndDocumentsCoverEachOther(t *testing.T) {
	claimed := map[string]bool{}
	for _, sc := range BaseScenarios(testEpisodePtr()) {
		for _, opp := range []string{"OLD", "PREV"} {
			for _, k := range ContractMetrics {
				if name := boundName(sc, k, opp); name != "" {
					claimed[name] = true
				}
			}
		}
	}
	for name := range claimed {
		if _, ok := derivationDocs[name]; !ok {
			t.Errorf("boundName returns %q for a measured scenario, but no derivation document is registered for it", name)
		}
	}
	for name, path := range derivationDocs {
		if !claimed[name] {
			t.Errorf("%s is registered for bound %q, which the table never returns", path, name)
		}
	}
}
