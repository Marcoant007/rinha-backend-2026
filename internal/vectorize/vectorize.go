package vectorize

import (
	"github.com/Marcoant007/rinha-2026/internal/models"
)

const (
	maxAmount            = 10000.0
	maxInstallments      = 12.0
	amountVsAvgRatio     = 10.0
	maxMinutes           = 1440.0
	maxKm                = 1000.0
	maxTxCount24h        = 20.0
	maxMerchantAvgAmount = 10000.0

	invMaxAmount            = 1.0 / maxAmount
	invMaxInstallments      = 1.0 / maxInstallments
	invAmountVsAvgRatio     = 1.0 / amountVsAvgRatio
	invMaxMinutes           = 1.0 / maxMinutes
	invMaxKm                = 1.0 / maxKm
	invMaxTxCount24h        = 1.0 / maxTxCount24h
	invMaxMerchantAvgAmount = 1.0 / maxMerchantAvgAmount
)

var mccRisk = map[string]float64{
	"5411": 0.15,
	"5812": 0.30,
	"5912": 0.20,
	"5944": 0.45,
	"7801": 0.80,
	"7802": 0.75,
	"7995": 0.85,
	"4511": 0.35,
	"5311": 0.25,
	"5999": 0.50,
}

func clamp(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// parseInt2 parses two ASCII digits from s[0:2] with no bounds check.
func parseInt2(s string) int {
	return int(s[0]-'0')*10 + int(s[1]-'0')
}

// parseInt4 parses four ASCII digits from s[0:4] with no bounds check.
func parseInt4(s string) int {
	return int(s[0]-'0')*1000 + int(s[1]-'0')*100 + int(s[2]-'0')*10 + int(s[3]-'0')
}

// julianDay computes the Julian Day Number for a Gregorian calendar date.
// Standard algorithm, works for any date after 1582.
func julianDay(y, m, d int) int {
	a := (14 - m) / 12
	yr := y + 4800 - a
	mo := m + 12*a - 3
	return d + (153*mo+2)/5 + 365*yr + yr/4 - yr/100 + yr/400 - 32045
}

// parseRFC3339 parses an RFC3339 timestamp into:
//   - hour: UTC hour (0–23)
//   - weekday: Mon=0 … Sun=6 (UTC)
//   - minutesUTC: total minutes from the Julian epoch (suitable for diffs)
//
// No allocations. Returns ok=false if the string is too short or malformed.
func parseRFC3339(s string) (hour, weekday int, minutesUTC int64, ok bool) {
	if len(s) < 19 {
		return
	}
	y := parseInt4(s[0:])
	mo := parseInt2(s[5:])
	d := parseInt2(s[8:])
	h := parseInt2(s[11:])
	mi := parseInt2(s[14:])

	// Timezone offset in minutes (+ means ahead of UTC)
	tzMins := 0
	if len(s) > 19 {
		switch s[19] {
		case '+':
			if len(s) >= 25 {
				tzMins = parseInt2(s[20:])*60 + parseInt2(s[23:])
			}
		case '-':
			if len(s) >= 25 {
				tzMins = -(parseInt2(s[20:])*60 + parseInt2(s[23:]))
			}
		// 'Z' → tzMins stays 0
		}
	}

	jdn := julianDay(y, mo, d)
	minutesUTC = int64(jdn)*1440 + int64(h)*60 + int64(mi) - int64(tzMins)

	// UTC hour (after applying timezone)
	totalH := int64(h)*60 + int64(mi) - int64(tzMins)
	// Normalise into [0, 24*60) range
	totalH = ((totalH % 1440) + 1440) % 1440
	hour = int(totalH / 60)

	// Weekday from Julian Day Number: JDN 0 = Monday
	// JDN mod 7: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
	// But we want Mon=0 … Sun=6 which is exactly (JDN % 7)
	// Adjust JDN for the UTC day (after timezone)
	utcJDN := jdn + int((int64(h)*60+int64(mi)-int64(tzMins))/1440)
	weekday = ((utcJDN % 7) + 7) % 7 // Mon=0 … Sun=6

	ok = true
	return
}

// Vectorize converts a transaction request into a 14-dimensional feature vector.
// No heap allocations.
func Vectorize(req *models.TransactionRequest) [14]float64 {
	var vec [14]float64

	vec[0] = clamp(req.Transaction.Amount * invMaxAmount)
	vec[1] = clamp(float64(req.Transaction.Installments) * invMaxInstallments)

	if req.Customer.AvgAmount > 0 {
		vec[2] = clamp((req.Transaction.Amount / req.Customer.AvgAmount) * invAmountVsAvgRatio)
	}

	txHour, txWeekday, txMinutes, txOK := parseRFC3339(req.Transaction.RequestedAt)
	if txOK {
		vec[3] = float64(txHour) / 23.0
		vec[4] = float64(txWeekday) / 6.0
	}

	if req.LastTx == nil {
		vec[5] = -1
		vec[6] = -1
	} else {
		if txOK {
			_, _, lastMinutes, lastOK := parseRFC3339(req.LastTx.Timestamp)
			if lastOK {
				minutes := float64(txMinutes - lastMinutes)
				vec[5] = clamp(minutes * invMaxMinutes)
			} else {
				vec[5] = -1
			}
		} else {
			vec[5] = -1
		}
		vec[6] = clamp(req.LastTx.KmFromCurrent * invMaxKm)
	}

	vec[7] = clamp(req.Terminal.KmFromHome * invMaxKm)
	vec[8] = clamp(float64(req.Customer.TxCount24h) * invMaxTxCount24h)

	if req.Terminal.IsOnline {
		vec[9] = 1
	}
	if req.Terminal.CardPresent {
		vec[10] = 1
	}

	vec[11] = 1
	for _, m := range req.Customer.KnownMerchants {
		if m == req.Merchant.ID {
			vec[11] = 0
			break
		}
	}

	if risk, ok := mccRisk[req.Merchant.MCC]; ok {
		vec[12] = risk
	} else {
		vec[12] = 0.5
	}

	vec[13] = clamp(req.Merchant.AvgAmount * invMaxMerchantAvgAmount)

	return vec
}
