package vectorize

import (
	"math"
	"time"

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
)

var MCCRisk = map[string]float64{
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
	return math.Max(0.0, math.Min(1.0, v))
}

func Vectorize(req *models.TransactionRequest) []float64 {
	vec := make([]float64, 14)

	vec[0] = clamp(req.Transaction.Amount / maxAmount)

	vec[1] = clamp(float64(req.Transaction.Installments) / maxInstallments)

	if req.Customer.AvgAmount > 0 {
		vec[2] = clamp((req.Transaction.Amount / req.Customer.AvgAmount) / amountVsAvgRatio)
	}

	t, err := time.Parse(time.RFC3339, req.Transaction.RequestedAt)
	if err == nil {
		vec[3] = float64(t.UTC().Hour()) / 23.0
		wd := int(t.UTC().Weekday())
		wd = (wd + 6) % 7
		vec[4] = float64(wd) / 6.0
	}

	if req.LastTx == nil {
		vec[5] = -1
		vec[6] = -1
	} else {
		lastTime, err := time.Parse(time.RFC3339, req.LastTx.Timestamp)
		if err == nil {
			minutes := t.Sub(lastTime).Minutes()
			vec[5] = clamp(minutes / maxMinutes)
		} else {
			vec[5] = -1
		}
		vec[6] = clamp(req.LastTx.KmFromCurrent / maxKm)
	}

	vec[7] = clamp(req.Terminal.KmFromHome / maxKm)
	vec[8] = clamp(float64(req.Customer.TxCount24h) / maxTxCount24h)

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

	if risk, ok := MCCRisk[req.Merchant.MCC]; ok {
		vec[12] = risk
	} else {
		vec[12] = 0.5
	}

	vec[13] = clamp(req.Merchant.AvgAmount / maxMerchantAvgAmount)

	return vec
}
