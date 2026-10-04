package sim

import (
	"github.com/stretchr/testify/assert"
	"github.com/wow-look-at-my/go-containers/set"
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
		assert.Nil(t, err)

		assert.Contains(t, strings.ToLower(string(body)), strings.ToLower(name))

	}
}

// boundName is consulted for every scenario the suite measures, so the set it can
// return is fixed by the table rather than by which cells happen to lose on this
// run. A bound name with no document would excuse a loss with nothing behind it,
// and a document no name returns is an argument nothing reads.
func TestBoundTableAndDocumentsCoverEachOther(t *testing.T) {
	claimed := set.New[string]()
	for _, sc := range BaseScenarios(testEpisodePtr()) {
		for _, opp := range []string{"OLD", "PREV"} {
			for _, k := range ContractMetrics {
				if name := boundName(sc, k, opp); name != "" {
					claimed.Add(name)
				}
			}
		}
	}
	for name := range claimed.All() {
		_, ok := derivationDocs[name]
		assert.True(t, ok)

	}
	for name, _ := range derivationDocs {
		assert.True(t, claimed.Contains(name))

	}
}
