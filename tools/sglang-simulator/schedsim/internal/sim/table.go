package sim

import (
	"fmt"
	"io"
	"math"
	"strings"
)

// ExtraRow is one of the supporting columns, reported beside the seven metrics
// but not part of the contract written over them.
type ExtraRow struct {
	Name string
	V    [NumModes]string
}

// WriteTables prints every scenario's three-policy comparison block.
func WriteTables(w io.Writer, rows []Row) {
	fmt.Fprintln(w, "Every cost the scheduler charges is the production log's own calibration;")
	fmt.Fprintln(w, "the metric values below are simulated from that model.")
	fmt.Fprintln(w, "A caret marks a cell where NEW is worse than the policy named: ^old, ^prev.")
	fmt.Fprintln(w)
	for _, r := range rows {
		fmt.Fprintf(w, "%s\n%s\n", r.Scenario.Name, strings.Repeat("-", 100))
		fmt.Fprintf(w, "  %s\n", r.Scenario.Note)
		fmt.Fprintf(w, "  seeds %v, window %.0f s, max_running %d, host tier %.1fx the device pool\n\n",
			r.Scenario.Seeds, r.Scenario.Window, r.Scenario.MaxRunning, r.Scenario.HostMul)
		fmt.Fprintf(w, "  %-32s %13s %13s %13s\n", "Metric", "OLD", "PREV", "NEW")
		for _, k := range ContractMetrics {
			var v [NumModes]float64
			for i := range Modes {
				v[i] = r.Metrics[i].Value(k)
			}
			mark := ""
			if !Better(k, v[2], v[0]) {
				mark += "^old"
			}
			if !Better(k, v[2], v[1]) {
				mark += "^prev"
			}
			fmt.Fprintf(w, "  %-32s %13s %13s %13s  %s\n", k.String()+" "+direction(k),
				format(k, v[0]), format(k, v[1]), format(k, v[2]), mark)
		}
		for _, e := range extras(r) {
			fmt.Fprintf(w, "  %-32s %13s %13s %13s\n", e.Name, e.V[0], e.V[1], e.V[2])
		}
		fmt.Fprintln(w)
	}
}

func direction(k MetricKey) string {
	if k.HigherIsBetter() {
		return "(up)"
	}
	return "(down)"
}

func format(k MetricKey, v float64) string {
	if math.IsNaN(v) {
		return "-"
	}
	switch k {
	case MStreamRate, MAgentRate, MThroughput:
		return fmt.Sprintf("%.1f tok/s", v)
	case MRecomputes:
		return fmt.Sprintf("%d", int(v))
	default:
		return seconds(v)
	}
}

// seconds renders a duration at the scale that argues it: a stall in seconds, an
// inter-token gap in milliseconds.
func seconds(v float64) string {
	if math.IsNaN(v) {
		return "-"
	}
	switch {
	case v < 0.001:
		return fmt.Sprintf("%.0f us", v*1e6)
	case v < 1:
		return fmt.Sprintf("%.1f ms", v*1e3)
	case v < 10:
		return fmt.Sprintf("%.2f s", v)
	}
	return fmt.Sprintf("%.1f s", v)
}

func pctText(v float64) string {
	if math.IsNaN(v) {
		return "-"
	}
	return fmt.Sprintf("%.1f%%", 100*v)
}

// extras lists the supporting columns for one scenario's three policies.
func extras(r Row) []ExtraRow {
	m := r.Metrics
	return []ExtraRow{
		{Name: "cold prompts served/arrived", V: [NumModes]string{
			fmt.Sprintf("%d/%d", m[0].ColdServed, m[0].ColdArrived),
			fmt.Sprintf("%d/%d", m[1].ColdServed, m[1].ColdArrived),
			fmt.Sprintf("%d/%d", m[2].ColdServed, m[2].ColdArrived)}},
		{Name: "stream time in stalls > 1 s", V: [NumModes]string{
			pctText(m[0].StallFrac1s), pctText(m[1].StallFrac1s), pctText(m[2].StallFrac1s)}},
		{Name: "ITL p50", V: [NumModes]string{seconds(m[0].ITLp50), seconds(m[1].ITLp50), seconds(m[2].ITLp50)}},
		{Name: "ITL p99 (chunk gaps)", V: [NumModes]string{
			seconds(m[0].ITLp99Raw), seconds(m[1].ITLp99Raw), seconds(m[2].ITLp99Raw)}},
		{Name: "turn TTFT p50", V: [NumModes]string{
			seconds(m[0].TurnTTFTp50), seconds(m[1].TurnTTFTp50), seconds(m[2].TurnTTFTp50)}},
		{Name: "turn TTFT p99", V: [NumModes]string{
			seconds(m[0].TurnTTFTp99), seconds(m[1].TurnTTFTp99), seconds(m[2].TurnTTFTp99)}},
		{Name: "short TTFT p50", V: [NumModes]string{
			seconds(m[0].ShortTTFTp50), seconds(m[1].ShortTTFTp50), seconds(m[2].ShortTTFTp50)}},
		{Name: "short TTFT p99", V: [NumModes]string{
			seconds(m[0].ShortTTFTp99), seconds(m[1].ShortTTFTp99), seconds(m[2].ShortTTFTp99)}},
		{Name: "completed turns", V: [NumModes]string{
			itoa(m[0].CompletedTurns), itoa(m[1].CompletedTurns), itoa(m[2].CompletedTurns)}},
		{Name: "GPU busy share", V: [NumModes]string{
			pctText(m[0].GPUBusyShare), pctText(m[1].GPUBusyShare), pctText(m[2].GPUBusyShare)}},
		{Name: "decode share of GPU", V: [NumModes]string{
			pctText(m[0].DecodeShare), pctText(m[1].DecodeShare), pctText(m[2].DecodeShare)}},
		{Name: "decode lines < 25 tok/s", V: [NumModes]string{
			fmt.Sprintf("%d/%d", m[0].LowGenLines, m[0].DecodeLogLines),
			fmt.Sprintf("%d/%d", m[1].LowGenLines, m[1].DecodeLogLines),
			fmt.Sprintf("%d/%d", m[2].LowGenLines, m[2].DecodeLogLines)}},
	}
}
