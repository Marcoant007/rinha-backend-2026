package vectorize

import (
	"testing"
	"time"

	"github.com/Marcoant007/rinha-2026/internal/models"
)

// --- parseRFC3339 unit tests ---

func TestParseRFC3339_UTC(t *testing.T) {
	s := "2023-06-15T14:30:00Z"
	hour, wd, mins, ok := parseRFC3339(s)
	if !ok {
		t.Fatal("parseRFC3339 returned ok=false")
	}

	ref, _ := time.Parse(time.RFC3339, s)
	if hour != ref.UTC().Hour() {
		t.Errorf("hour: got %d, want %d", hour, ref.UTC().Hour())
	}

	refWd := (int(ref.UTC().Weekday()) + 6) % 7 // Mon=0, ..., Sun=6
	if wd != refWd {
		t.Errorf("weekday: got %d, want %d (day=%s)", wd, refWd, ref.UTC().Weekday())
	}

	// Verify minutes are consistent: two timestamps 90 minutes apart should differ by 90.
	s2 := "2023-06-15T16:00:00Z"
	_, _, mins2, _ := parseRFC3339(s2)
	diff := mins2 - mins
	if diff != 90 {
		t.Errorf("minute diff: got %d, want 90", diff)
	}
	_ = mins
}

func TestParseRFC3339_PosOffset(t *testing.T) {
	s := "2023-06-15T19:30:00+05:30"
	hour, _, _, ok := parseRFC3339(s)
	if !ok {
		t.Fatal("parseRFC3339 returned ok=false")
	}
	ref, _ := time.Parse(time.RFC3339, s)
	if hour != ref.UTC().Hour() {
		t.Errorf("hour with +05:30: got %d, want %d", hour, ref.UTC().Hour())
	}
}

func TestParseRFC3339_NegOffset(t *testing.T) {
	s := "2023-06-15T06:30:00-08:00"
	hour, _, _, ok := parseRFC3339(s)
	if !ok {
		t.Fatal("parseRFC3339 returned ok=false")
	}
	ref, _ := time.Parse(time.RFC3339, s)
	if hour != ref.UTC().Hour() {
		t.Errorf("hour with -08:00: got %d, want %d", hour, ref.UTC().Hour())
	}
}

func TestParseRFC3339_MinuteDiff(t *testing.T) {
	// Two timestamps 75 minutes apart, same day
	s1 := "2024-03-10T10:00:00Z"
	s2 := "2024-03-10T11:15:00Z"
	_, _, m1, _ := parseRFC3339(s1)
	_, _, m2, _ := parseRFC3339(s2)
	if m2-m1 != 75 {
		t.Errorf("minute diff: got %d, want 75", m2-m1)
	}
}

func TestParseRFC3339_CrossDay(t *testing.T) {
	// Timestamps 90 minutes apart across midnight
	s1 := "2024-01-01T23:15:00Z"
	s2 := "2024-01-02T00:45:00Z"
	_, _, m1, _ := parseRFC3339(s1)
	_, _, m2, _ := parseRFC3339(s2)
	if m2-m1 != 90 {
		t.Errorf("cross-day minute diff: got %d, want 90", m2-m1)
	}
}

// --- Vectorize smoke test ---

func sampleRequest() *models.TransactionRequest {
	return &models.TransactionRequest{
		Transaction: models.Transaction{
			Amount:       150.0,
			Installments: 3,
			RequestedAt:  "2024-06-10T14:30:00Z",
		},
		Customer: models.Customer{
			AvgAmount:      100.0,
			TxCount24h:     2,
			KnownMerchants: []string{"merchant-abc"},
		},
		Merchant: models.Merchant{
			ID:        "merchant-xyz",
			MCC:       "5999",
			AvgAmount: 200.0,
		},
		Terminal: models.Terminal{
			IsOnline:    true,
			CardPresent: false,
			KmFromHome:  50.0,
		},
		LastTx: &models.LastTx{
			Timestamp:     "2024-06-10T13:00:00Z",
			KmFromCurrent: 10.0,
		},
	}
}

func TestVectorize_Bounds(t *testing.T) {
	req := sampleRequest()
	vec := Vectorize(req)

	// All dimensions must be in [-1, 1]
	for i, v := range vec {
		if v < -1 || v > 1 {
			t.Errorf("vec[%d] = %f out of [-1, 1]", i, v)
		}
	}

	// vec[5] = minutes_since_last_tx: 90 minutes → clamp(90/1440)
	expected5 := 90.0 / 1440.0
	if abs(vec[5]-expected5) > 0.001 {
		t.Errorf("vec[5] (minutes_since_last_tx): got %f, want ~%f", vec[5], expected5)
	}
}

func TestVectorize_NoLastTx(t *testing.T) {
	req := sampleRequest()
	req.LastTx = nil
	vec := Vectorize(req)
	if vec[5] != -1 {
		t.Errorf("vec[5] with nil LastTx: got %f, want -1", vec[5])
	}
	if vec[6] != -1 {
		t.Errorf("vec[6] with nil LastTx: got %f, want -1", vec[6])
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// --- Benchmarks ---

func BenchmarkParseRFC3339(b *testing.B) {
	s := "2024-06-10T14:30:00Z"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parseRFC3339(s)
	}
}

func BenchmarkParseRFC3339_WithOffset(b *testing.B) {
	s := "2024-06-10T19:30:00+05:30"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parseRFC3339(s)
	}
}

func BenchmarkStdlibTimeParse(b *testing.B) {
	s := "2024-06-10T14:30:00Z"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		time.Parse(time.RFC3339, s)
	}
}

func BenchmarkVectorize(b *testing.B) {
	req := sampleRequest()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Vectorize(req)
	}
}

func BenchmarkVectorize_NoLastTx(b *testing.B) {
	req := sampleRequest()
	req.LastTx = nil
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Vectorize(req)
	}
}
