package sim

import (
	"github.com/stretchr/testify/require"
	"math"
	"strings"
	"testing"
)

// The contract suite compares NEW against PREV. If PREV ever stopped differing
// from NEW -- a rule ported as a no-op, a feature flag wired to the wrong mode --
// every comparison against it would still pass while proving nothing, because a
// policy cannot lose to a copy of itself. TestFeaturesTablePinsTheModes checks the
// rule sets differ; this checks that the difference reaches the numbers.
func TestPrevAndNewReportDifferentNumbers(t *testing.T) {
	rows := suite(t)
	var disagreeing, firstDetail []string
	for _, sc := range BaseScenarios(testEpisodePtr()) {
		row, ok := rows[sc.Key]
		require.True(t, ok)

		prev, neu := row.Metrics[PolicyIndex(ModePrev)], row.Metrics[PolicyIndex(ModeNew)]
		var cells []string
		for _, k := range ContractMetrics {
			a, b := prev.Value(k), neu.Value(k)
			if math.IsNaN(a) || math.IsNaN(b) {
				continue
			}
			tol := 1e-9 + 1e-6*math.Max(math.Abs(a), math.Abs(b))
			if math.Abs(a-b) > tol {
				cells = append(cells, k.String())
				firstDetail = append(firstDetail, sc.Key+" "+k.String()+": PREV "+fmtVal(k, a)+" vs NEW "+fmtVal(k, b))
			}
		}
		if len(cells) > 0 {
			disagreeing = append(disagreeing, sc.Key+": "+strings.Join(cells, ", "))
		}
	}
	require.NotEqual(t, 0, len(disagreeing))

	t.Logf("PREV and NEW disagree on %d of %d scenario-metric cells, in %d scenarios:",
		len(firstDetail), len(BaseScenarios(testEpisodePtr()))*len(ContractMetrics), len(disagreeing))
	for _, d := range disagreeing {
		t.Logf("  %s", d)
	}
}
